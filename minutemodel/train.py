from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .baselines import GroupMeanBaseline, RidgeDraftBaseline
from .config import PipelineConfig
from .data_loading import candidate_transformation_columns, filter_complete_games, load_raw_csv
from .evaluate import (
    add_duration_bucket,
    benchmarking_table,
    error_breakdown_table,
    regression_metrics_seconds,
    save_error_bar_plot,
    save_residual_plots,
)
from .feature_engineering import DraftFeatureBuilder, build_leakage_safe_rolling_priors, build_target
from .preprocessing import flatten_to_match_level
from .schema_inspection import format_schema_report, inspect_schema
from .utils import ensure_dir, seed_everything, write_json

LOGGER = logging.getLogger(__name__)

try:
    from catboost import CatBoostRegressor  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    CatBoostRegressor = None

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    shap = None


@dataclass
class SplitData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def chronological_split(match_df: pd.DataFrame, config: PipelineConfig) -> SplitData:
    work = match_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=False)
    work = work.sort_values(["date", "gameid"]).reset_index(drop=True)

    n_rows = len(work)
    if n_rows < 30:
        raise ValueError("Need at least 30 match rows for train/validation/test split.")

    train_end = int(n_rows * config.train_fraction)
    val_end = train_end + int(n_rows * config.validation_fraction)

    train_end = max(train_end, 1)
    val_end = max(val_end, train_end + 1)
    val_end = min(val_end, n_rows - 1)

    train_df = work.iloc[:train_end].copy()
    val_df = work.iloc[train_end:val_end].copy()
    test_df = work.iloc[val_end:].copy()

    LOGGER.info("Chronological split sizes -> train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))
    return SplitData(train=train_df, validation=val_df, test=test_df)


def build_preprocessor(categorical_cols: List[str], numeric_cols: List[str]) -> ColumnTransformer:
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipe, categorical_cols),
            ("numeric", numeric_pipe, numeric_cols),
        ],
        sparse_threshold=0.3,
    )


def tune_lightgbm(
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
    config: PipelineConfig,
) -> Tuple[Dict[str, float], lgb.LGBMRegressor, float]:
    best_params: Dict[str, float] = {}
    best_model: lgb.LGBMRegressor | None = None
    best_mae = float("inf")

    for idx, params in enumerate(config.lgbm_param_grid):
        candidate = lgb.LGBMRegressor(
            objective="regression",
            random_state=config.random_seed,
            n_jobs=-1,
            **params,
        )
        candidate.fit(X_train, y_train)
        val_pred = candidate.predict(X_val)
        val_mae = float(mean_absolute_error(y_val, val_pred))

        LOGGER.info("LightGBM candidate %d validation MAE (target units): %.4f", idx + 1, val_mae)
        if val_mae < best_mae:
            best_mae = val_mae
            best_params = params
            best_model = candidate

    if best_model is None:
        raise RuntimeError("LightGBM tuning failed to produce a valid model.")

    return best_params, best_model, best_mae


def _train_final_lightgbm(
    X_train_val,
    y_train_val: np.ndarray,
    best_params: Dict[str, float],
    config: PipelineConfig,
) -> lgb.LGBMRegressor:
    final_model = lgb.LGBMRegressor(
        objective="regression",
        random_state=config.random_seed,
        n_jobs=-1,
        **best_params,
    )
    final_model.fit(X_train_val, y_train_val)
    return final_model


def _prepare_catboost_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[int], List[str], List[str]]:
    """Prepare feature dataframe for CatBoost with native categorical support."""
    work = df.copy()

    categorical_cols = [col for col in work.columns if str(work[col].dtype) == "object"]
    numeric_cols = [col for col in work.columns if col not in categorical_cols]

    for col in categorical_cols:
        work[col] = work[col].astype(str).fillna("__MISSING__")

    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    cat_feature_indices = [work.columns.get_loc(col) for col in categorical_cols]
    return work, cat_feature_indices, categorical_cols, numeric_cols


def tune_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    config: PipelineConfig,
) -> Tuple[Dict[str, float], Any, float, int]:
    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost is not installed. Install `catboost` to use primary_model=catboost.")

    X_train_cb, cat_feature_indices, _, _ = _prepare_catboost_frame(X_train)
    X_val_cb, _, _, _ = _prepare_catboost_frame(X_val)

    best_params: Dict[str, float] = {}
    best_model: Any = None
    best_mae = float("inf")
    best_iteration = 0

    for idx, params in enumerate(config.catboost_param_grid):
        candidate = CatBoostRegressor(
            loss_function="MAE",
            eval_metric="MAE",
            iterations=config.catboost_iterations,
            random_seed=config.random_seed,
            verbose=False,
            **params,
        )
        candidate.fit(
            X_train_cb,
            y_train,
            cat_features=cat_feature_indices,
            eval_set=(X_val_cb, y_val),
            use_best_model=True,
            early_stopping_rounds=config.catboost_early_stopping_rounds,
        )
        val_pred = candidate.predict(X_val_cb)
        val_mae = float(mean_absolute_error(y_val, val_pred))

        LOGGER.info("CatBoost candidate %d validation MAE (target units): %.4f", idx + 1, val_mae)
        if val_mae < best_mae:
            best_mae = val_mae
            best_params = dict(params)
            best_model = candidate
            best_iteration = int(candidate.get_best_iteration())

    if best_model is None:
        raise RuntimeError("CatBoost tuning failed to produce a valid model.")

    if best_iteration <= 0:
        best_iteration = config.catboost_iterations

    return best_params, best_model, best_mae, best_iteration


def _train_final_catboost(
    X_train_val: pd.DataFrame,
    y_train_val: np.ndarray,
    best_params: Dict[str, float],
    best_iteration: int,
    config: PipelineConfig,
) -> Any:
    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost is not installed. Install `catboost` to use primary_model=catboost.")

    X_train_val_cb, cat_feature_indices, _, _ = _prepare_catboost_frame(X_train_val)
    final_model = CatBoostRegressor(
        loss_function="MAE",
        eval_metric="MAE",
        iterations=max(int(best_iteration), 1),
        random_seed=config.random_seed,
        verbose=False,
        **best_params,
    )
    final_model.fit(X_train_val_cb, y_train_val, cat_features=cat_feature_indices)
    return final_model


def _save_shap_diagnostics(
    model: Any,
    X_test_transformed,
    feature_names: List[str],
    out_dir: str | Path,
) -> None:
    if shap is None:
        LOGGER.warning("SHAP is not installed; skipping SHAP diagnostics.")
        return

    target_dir = ensure_dir(out_dir)

    max_rows = min(500, X_test_transformed.shape[0])
    X_sample = X_test_transformed[:max_rows]
    if hasattr(X_sample, "toarray"):
        X_sample = X_sample.toarray()

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as exc:
        LOGGER.warning("SHAP diagnostics failed for current model type: %s", exc)
        return

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance_df.to_csv(target_dir / "shap_global_importance.csv", index=False)

    top_n = importance_df.head(25)
    plt.figure(figsize=(10, 7))
    plt.barh(top_n["feature"][::-1], top_n["mean_abs_shap"][::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("Global SHAP Importance (Top 25)")
    plt.tight_layout()
    plt.savefig(target_dir / "shap_global_importance_top25.png", dpi=140)
    plt.close()

    local_rows = []
    for idx in range(min(3, X_sample.shape[0])):
        row_values = shap_values[idx]
        top_idx = np.argsort(np.abs(row_values))[::-1][:10]
        for rank, feat_idx in enumerate(top_idx, start=1):
            local_rows.append(
                {
                    "sample_index": idx,
                    "rank": rank,
                    "feature": feature_names[int(feat_idx)],
                    "shap_value": float(row_values[int(feat_idx)]),
                }
            )

    pd.DataFrame(local_rows).to_csv(target_dir / "shap_local_explanations_top10.csv", index=False)


def _train_lightgbm_variant(split: SplitData, config: PipelineConfig) -> Dict[str, Any]:
    """Train one LightGBM variant (with a specific feature config) and score test set."""
    feature_builder = DraftFeatureBuilder(config=config)
    X_train = feature_builder.fit_transform(split.train)
    X_val = feature_builder.transform(split.validation)
    X_test = feature_builder.transform(split.test)

    y_train = build_target(split.train, config).to_numpy(dtype=float)
    y_val = build_target(split.validation, config).to_numpy(dtype=float)
    y_test = build_target(split.test, config).to_numpy(dtype=float)
    unit_to_seconds = 60.0 if config.target_unit == "minutes" else 1.0

    categorical_cols, numeric_cols = feature_builder.get_feature_columns()

    preprocessor_tune = build_preprocessor(categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    X_train_transformed = preprocessor_tune.fit_transform(X_train)
    X_val_transformed = preprocessor_tune.transform(X_val)

    best_params, _, best_val_mae = tune_lightgbm(
        X_train=X_train_transformed,
        y_train=y_train,
        X_val=X_val_transformed,
        y_val=y_val,
        config=config,
    )
    best_val_mae_seconds = best_val_mae * unit_to_seconds

    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    preprocessor_final = build_preprocessor(categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    X_train_val_transformed = preprocessor_final.fit_transform(X_train_val)
    X_test_transformed = preprocessor_final.transform(X_test)

    final_model = _train_final_lightgbm(
        X_train_val=X_train_val_transformed,
        y_train_val=y_train_val,
        best_params=best_params,
        config=config,
    )

    y_pred_model_unit = final_model.predict(X_test_transformed)
    y_pred_seconds = y_pred_model_unit * unit_to_seconds
    y_test_seconds = y_test * unit_to_seconds
    metrics = regression_metrics_seconds(y_test_seconds, y_pred_seconds)

    return {
        "feature_builder": feature_builder,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_test_seconds": y_test_seconds,
        "unit_to_seconds": unit_to_seconds,
        "X_train_val": X_train_val,
        "y_train_val": y_train_val,
        "X_test_transformed": X_test_transformed,
        "preprocessor_final": preprocessor_final,
        "model": final_model,
        "feature_names": preprocessor_final.get_feature_names_out().tolist(),
        "best_params": best_params,
        "best_val_mae_seconds": best_val_mae_seconds,
        "preds_model_unit": y_pred_model_unit,
        "preds_seconds": y_pred_seconds,
        "metrics": metrics,
        "model_name": "lightgbm",
    }


def _train_catboost_variant(split: SplitData, config: PipelineConfig) -> Dict[str, Any]:
    """Train one CatBoost variant (with a specific feature config) and score test set."""
    feature_builder = DraftFeatureBuilder(config=config)
    X_train = feature_builder.fit_transform(split.train)
    X_val = feature_builder.transform(split.validation)
    X_test = feature_builder.transform(split.test)

    y_train = build_target(split.train, config).to_numpy(dtype=float)
    y_val = build_target(split.validation, config).to_numpy(dtype=float)
    y_test = build_target(split.test, config).to_numpy(dtype=float)
    unit_to_seconds = 60.0 if config.target_unit == "minutes" else 1.0

    categorical_cols, numeric_cols = feature_builder.get_feature_columns()

    best_params, _, best_val_mae, best_iteration = tune_catboost(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        config=config,
    )
    best_val_mae_seconds = best_val_mae * unit_to_seconds

    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = np.concatenate([y_train, y_val], axis=0)
    X_test_cb, _, cat_cols, numeric_cols_cb = _prepare_catboost_frame(X_test)

    final_model = _train_final_catboost(
        X_train_val=X_train_val,
        y_train_val=y_train_val,
        best_params=best_params,
        best_iteration=best_iteration,
        config=config,
    )

    y_pred_model_unit = final_model.predict(X_test_cb)
    y_pred_seconds = y_pred_model_unit * unit_to_seconds
    y_test_seconds = y_test * unit_to_seconds
    metrics = regression_metrics_seconds(y_test_seconds, y_pred_seconds)

    return {
        "feature_builder": feature_builder,
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_test_seconds": y_test_seconds,
        "unit_to_seconds": unit_to_seconds,
        "X_train_val": X_train_val,
        "y_train_val": y_train_val,
        "X_test_transformed": X_test_cb,
        "preprocessor_final": None,
        "model": final_model,
        "feature_names": list(X_test_cb.columns),
        "best_params": best_params,
        "best_val_mae_seconds": best_val_mae_seconds,
        "best_iteration": int(best_iteration),
        "preds_model_unit": y_pred_model_unit,
        "preds_seconds": y_pred_seconds,
        "metrics": metrics,
        "model_name": "catboost",
        "catboost_categorical_cols": cat_cols,
        "catboost_numeric_cols": numeric_cols_cb,
    }


def _save_feature_column_artifacts(
    reports_dir: str | Path,
    categorical_cols: List[str],
    numeric_cols: List[str],
    transformed_feature_names: List[str],
) -> None:
    target_dir = ensure_dir(reports_dir)

    input_rows = (
        [{"feature": col, "feature_type": "categorical"} for col in categorical_cols]
        + [{"feature": col, "feature_type": "numeric"} for col in numeric_cols]
    )
    pd.DataFrame(input_rows).to_csv(target_dir / "model_input_feature_columns.csv", index=False)

    pd.DataFrame({"feature": transformed_feature_names}).to_csv(
        target_dir / "transformed_feature_columns.csv",
        index=False,
    )


def _save_catboost_feature_importance(
    model: Any,
    feature_names: List[str],
    reports_dir: str | Path,
) -> str | None:
    if not hasattr(model, "get_feature_importance"):
        return None

    importances = model.get_feature_importance()
    if importances is None:
        return None

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": pd.to_numeric(importances, errors="coerce"),
        }
    ).sort_values("importance", ascending=False)

    path = Path(reports_dir) / "catboost_feature_importance.csv"
    importance_df.to_csv(path, index=False)
    return str(path)


def _feature_group_ablation_configs(base_config: PipelineConfig) -> List[Tuple[str, PipelineConfig]]:
    variants = [
        (
            "A_baseline",
            replace(
                base_config,
                use_extended_rolling_team_priors=False,
                use_draft_summary_features=False,
                use_draft_interaction_features=False,
                use_draft_conditional_behaviour_features=False,
            ),
        ),
        (
            "B_plus_extra_rolling",
            replace(
                base_config,
                use_extended_rolling_team_priors=True,
                use_draft_summary_features=False,
                use_draft_interaction_features=False,
                use_draft_conditional_behaviour_features=False,
            ),
        ),
        (
            "C_plus_draft_summary",
            replace(
                base_config,
                use_extended_rolling_team_priors=True,
                use_draft_summary_features=True,
                use_draft_interaction_features=False,
                use_draft_conditional_behaviour_features=False,
            ),
        ),
        (
            "D_plus_interactions",
            replace(
                base_config,
                use_extended_rolling_team_priors=True,
                use_draft_summary_features=True,
                use_draft_interaction_features=True,
                use_draft_conditional_behaviour_features=False,
            ),
        ),
        (
            "E_plus_conditional_behaviour",
            replace(
                base_config,
                use_extended_rolling_team_priors=True,
                use_draft_summary_features=True,
                use_draft_interaction_features=True,
                use_draft_conditional_behaviour_features=True,
            ),
        ),
    ]
    return variants


def train_and_evaluate(config: PipelineConfig) -> Dict[str, object]:
    config.validate()
    seed_everything(config.random_seed)

    output_dir = ensure_dir(config.output_dir)
    reports_dir = ensure_dir(config.reports_dir)

    raw_df = load_raw_csv(config.input_csv)
    raw_df = filter_complete_games(raw_df)
    safe_columns = candidate_transformation_columns(raw_df)
    raw_df = raw_df[safe_columns].copy()

    schema_report = inspect_schema(raw_df)
    schema_text = format_schema_report(schema_report)
    (Path(reports_dir) / "schema_report.txt").write_text(schema_text, encoding="utf-8")

    match_df = flatten_to_match_level(raw_df, config=config, target_unit_guess=schema_report.gamelength_unit_guess)
    match_df = build_leakage_safe_rolling_priors(match_df, config=config)

    split = chronological_split(match_df, config)
    if config.primary_model == "catboost":
        main_run = _train_catboost_variant(split=split, config=config)
    else:
        main_run = _train_lightgbm_variant(split=split, config=config)

    primary_model_name: str = main_run["model_name"]
    feature_builder: DraftFeatureBuilder = main_run["feature_builder"]
    categorical_cols: List[str] = main_run["categorical_cols"]
    numeric_cols: List[str] = main_run["numeric_cols"]
    X_train: pd.DataFrame = main_run["X_train"]
    X_val: pd.DataFrame = main_run["X_val"]
    X_test: pd.DataFrame = main_run["X_test"]
    y_train: np.ndarray = main_run["y_train"]
    y_val: np.ndarray = main_run["y_val"]
    y_test_seconds: np.ndarray = main_run["y_test_seconds"]
    unit_to_seconds: float = main_run["unit_to_seconds"]
    X_train_val: pd.DataFrame = main_run["X_train_val"]
    y_train_val: np.ndarray = main_run["y_train_val"]
    preprocessor_final = main_run["preprocessor_final"]
    X_test_transformed = main_run["X_test_transformed"]
    final_primary_model = main_run["model"]
    best_params: Dict[str, float] = main_run["best_params"]
    best_val_mae_seconds: float = main_run["best_val_mae_seconds"]
    feature_names: List[str] = main_run["feature_names"]
    primary_preds_model_unit: np.ndarray = main_run["preds_model_unit"]

    _save_feature_column_artifacts(
        reports_dir=reports_dir,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        transformed_feature_names=feature_names,
    )

    # Baselines (trained on train+val where no tuning is required).
    global_mean_value = float(np.mean(y_train_val))

    league_baseline = GroupMeanBaseline(["league"]).fit(X_train_val, pd.Series(y_train_val))
    league_patch_baseline = GroupMeanBaseline(["league", "patch"]).fit(X_train_val, pd.Series(y_train_val))

    ridge_alpha_candidates = [1.0, 3.0, 10.0]
    best_alpha = ridge_alpha_candidates[0]
    best_alpha_mae = float("inf")
    for alpha in ridge_alpha_candidates:
        ridge_model = RidgeDraftBaseline(alpha=alpha).fit(X_train, pd.Series(y_train), categorical_cols, numeric_cols)
        ridge_val_pred = ridge_model.predict(X_val)
        val_mae = float(np.mean(np.abs(ridge_val_pred - y_val)))
        if val_mae < best_alpha_mae:
            best_alpha_mae = val_mae
            best_alpha = alpha

    ridge_model = RidgeDraftBaseline(alpha=best_alpha).fit(
        X_train_val,
        pd.Series(y_train_val),
        categorical_cols,
        numeric_cols,
    )

    preds_model_unit: Dict[str, np.ndarray] = {
        "global_mean": np.full(len(X_test), global_mean_value, dtype=float),
        "league_mean": league_baseline.predict(X_test),
        "league_patch_mean": league_patch_baseline.predict(X_test),
        "ridge_regression": ridge_model.predict(X_test),
        primary_model_name: primary_preds_model_unit,
    }
    preds_seconds = {name: values * unit_to_seconds for name, values in preds_model_unit.items()}

    metrics_by_model: Dict[str, Dict[str, float]] = {
        model_name: regression_metrics_seconds(y_test_seconds, y_pred_seconds)
        for model_name, y_pred_seconds in preds_seconds.items()
    }

    benchmark_df = benchmarking_table(metrics_by_model)
    benchmark_path = Path(reports_dir) / "benchmark_summary.csv"
    benchmark_df.to_csv(benchmark_path)
    LOGGER.info("Saved benchmark summary to %s", benchmark_path)

    champion_scaling_ablation_payload: Dict[str, Any] = {}
    champion_scaling_ablation_path: Path | None = None
    if config.run_champion_scaling_ablation:
        variant_train_fn = _train_catboost_variant if config.primary_model == "catboost" else _train_lightgbm_variant
        variant_metrics: Dict[str, Dict[str, float]] = {}
        variant_runs: Dict[bool, Dict[str, Any]] = {bool(config.use_champion_scaling_features): main_run}
        for use_scaling in [False, True]:
            label = "with_scaling" if use_scaling else "without_scaling"
            if use_scaling not in variant_runs:
                variant_config = replace(config, use_champion_scaling_features=use_scaling)
                variant_runs[use_scaling] = variant_train_fn(split=split, config=variant_config)
            variant_metrics[label] = variant_runs[use_scaling]["metrics"]

        ablation_df = benchmarking_table(variant_metrics)
        without_row = ablation_df.loc["without_scaling"]
        ablation_df["mae_minutes_delta_vs_without"] = ablation_df["mae_minutes"] - float(without_row["mae_minutes"])
        ablation_df["rmse_minutes_delta_vs_without"] = ablation_df["rmse_minutes"] - float(without_row["rmse_minutes"])
        ablation_df["median_absolute_error_minutes_delta_vs_without"] = (
            ablation_df["median_absolute_error_minutes"] - float(without_row["median_absolute_error_minutes"])
        )
        ablation_df["within_2_minutes_accuracy_delta_vs_without"] = (
            ablation_df["within_2_minutes_accuracy"] - float(without_row["within_2_minutes_accuracy"])
        )
        ablation_df["within_5_minutes_accuracy_delta_vs_without"] = (
            ablation_df["within_5_minutes_accuracy"] - float(without_row["within_5_minutes_accuracy"])
        )

        champion_scaling_ablation_path = Path(reports_dir) / "champion_scaling_ablation.csv"
        ablation_df.to_csv(champion_scaling_ablation_path)
        LOGGER.info("Saved champion scaling ablation summary to %s", champion_scaling_ablation_path)

        champion_scaling_ablation_payload = {
            "enabled": True,
            "variant_metrics": variant_metrics,
            "ablation_csv": str(champion_scaling_ablation_path),
        }
    else:
        champion_scaling_ablation_payload = {"enabled": False}

    feature_group_ablation_payload: Dict[str, Any] = {"enabled": False}
    feature_group_ablation_csv_path: Path | None = None
    feature_group_ablation_json_path: Path | None = None
    if config.run_feature_group_ablation:
        variant_train_fn = _train_catboost_variant if config.primary_model == "catboost" else _train_lightgbm_variant
        variant_metrics: Dict[str, Dict[str, float]] = {}
        variant_feature_counts: Dict[str, int] = {}

        for label, variant_cfg in _feature_group_ablation_configs(config):
            LOGGER.info(
                "Running feature-group ablation variant=%s (rolling=%s, summary=%s, interaction=%s, conditional=%s)",
                label,
                variant_cfg.use_extended_rolling_team_priors,
                variant_cfg.use_draft_summary_features,
                variant_cfg.use_draft_interaction_features,
                variant_cfg.use_draft_conditional_behaviour_features,
            )
            same_as_main = (
                variant_cfg.use_extended_rolling_team_priors == config.use_extended_rolling_team_priors
                and variant_cfg.use_draft_summary_features == config.use_draft_summary_features
                and variant_cfg.use_draft_interaction_features == config.use_draft_interaction_features
                and variant_cfg.use_draft_conditional_behaviour_features == config.use_draft_conditional_behaviour_features
                and variant_cfg.use_champion_scaling_features == config.use_champion_scaling_features
            )
            variant_run = main_run if same_as_main else variant_train_fn(split=split, config=variant_cfg)
            variant_metrics[label] = variant_run["metrics"]
            variant_feature_counts[label] = int(len(variant_run["feature_names"]))

        ablation_df = benchmarking_table(variant_metrics)
        ablation_df["feature_count"] = pd.Series(variant_feature_counts)
        baseline_row = ablation_df.loc["A_baseline"]
        ablation_df["mae_minutes_delta_vs_A"] = ablation_df["mae_minutes"] - float(baseline_row["mae_minutes"])
        ablation_df["rmse_minutes_delta_vs_A"] = ablation_df["rmse_minutes"] - float(baseline_row["rmse_minutes"])
        ablation_df["median_absolute_error_minutes_delta_vs_A"] = (
            ablation_df["median_absolute_error_minutes"] - float(baseline_row["median_absolute_error_minutes"])
        )
        ablation_df["within_2_minutes_accuracy_delta_vs_A"] = (
            ablation_df["within_2_minutes_accuracy"] - float(baseline_row["within_2_minutes_accuracy"])
        )
        ablation_df["within_5_minutes_accuracy_delta_vs_A"] = (
            ablation_df["within_5_minutes_accuracy"] - float(baseline_row["within_5_minutes_accuracy"])
        )

        feature_group_ablation_csv_path = Path(reports_dir) / "feature_group_ablation.csv"
        ablation_df.to_csv(feature_group_ablation_csv_path)
        feature_group_ablation_json_path = Path(reports_dir) / "feature_group_ablation.json"
        write_json(
            {
                "variants": list(ablation_df.index),
                "metrics_by_variant": variant_metrics,
                "feature_count_by_variant": variant_feature_counts,
            },
            feature_group_ablation_json_path,
        )
        LOGGER.info("Saved feature-group ablation to %s and %s", feature_group_ablation_csv_path, feature_group_ablation_json_path)

        feature_group_ablation_payload = {
            "enabled": True,
            "ablation_csv": str(feature_group_ablation_csv_path),
            "ablation_json": str(feature_group_ablation_json_path),
            "variant_metrics": variant_metrics,
            "feature_count_by_variant": variant_feature_counts,
        }

    test_eval_df = split.test[["gameid", "league", "patch", "date", "target_gamelength_seconds"]].copy()
    test_eval_df = test_eval_df.rename(columns={"target_gamelength_seconds": "y_true_seconds"})
    test_eval_df["y_pred_seconds"] = preds_seconds[primary_model_name]
    test_eval_df = add_duration_bucket(test_eval_df)
    LOGGER.info("Prepared test evaluation dataframe.")

    save_residual_plots(test_eval_df, reports_dir, primary_model_name)
    LOGGER.info("Saved residual plots.")

    league_breakdown = error_breakdown_table(test_eval_df, group_col="league")
    patch_breakdown = error_breakdown_table(test_eval_df, group_col="patch")
    bucket_breakdown = error_breakdown_table(test_eval_df, group_col="duration_bucket")

    league_breakdown.to_csv(Path(reports_dir) / "error_by_league.csv", index=False)
    patch_breakdown.to_csv(Path(reports_dir) / "error_by_patch.csv", index=False)
    bucket_breakdown.to_csv(Path(reports_dir) / "error_by_duration_bucket.csv", index=False)
    LOGGER.info("Saved error breakdown tables.")

    save_error_bar_plot(league_breakdown, "league", reports_dir, primary_model_name)
    save_error_bar_plot(patch_breakdown, "patch", reports_dir, primary_model_name)
    save_error_bar_plot(bucket_breakdown, "duration_bucket", reports_dir, primary_model_name)
    LOGGER.info("Saved error breakdown plots.")

    catboost_feature_importance_path: str | None = None
    if primary_model_name == "catboost":
        catboost_feature_importance_path = _save_catboost_feature_importance(
            model=final_primary_model,
            feature_names=feature_names,
            reports_dir=reports_dir,
        )
        if catboost_feature_importance_path:
            LOGGER.info("Saved CatBoost feature importance to %s", catboost_feature_importance_path)

    _save_shap_diagnostics(final_primary_model, X_test_transformed, feature_names, Path(reports_dir) / "shap")
    LOGGER.info("SHAP diagnostics step completed.")

    champion_scaling_lookup_path: str | None = None
    champion_scaling_lookup_artifact_path: str | None = None
    if config.use_champion_scaling_features:
        lookup_df = feature_builder.get_champion_scaling_lookup_table()
        if not lookup_df.empty:
            lookup_target = Path(reports_dir) / "champion_scaling_lookup.csv"
            lookup_df.to_csv(lookup_target, index=False)
            champion_scaling_lookup_path = str(lookup_target)
            LOGGER.info("Saved champion scaling lookup table to %s", lookup_target)
            if feature_builder.champion_scaling_lookup_ is not None:
                lookup_artifact_target = Path(output_dir) / "champion_scaling_lookup.joblib"
                feature_builder.champion_scaling_lookup_.save(lookup_artifact_target)
                champion_scaling_lookup_artifact_path = str(lookup_artifact_target)
                LOGGER.info("Saved champion scaling lookup artifact to %s", lookup_artifact_target)

    model_payload = {
        "model": final_primary_model,
        "preprocessor": preprocessor_final,
        "feature_builder": feature_builder,
        "primary_model": primary_model_name,
        "primary_model_best_params": best_params,
        "primary_model_validation_mae_seconds": best_val_mae_seconds,
        "ridge_alpha": best_alpha,
        "global_mean_baseline": global_mean_value * unit_to_seconds,
        "schema_report": schema_report.to_dict(),
        "config": config.to_dict(),
        "feature_names": feature_names,
        "input_feature_columns_csv": str(Path(reports_dir) / "model_input_feature_columns.csv"),
        "transformed_feature_columns_csv": str(Path(reports_dir) / "transformed_feature_columns.csv"),
    }
    if primary_model_name == "lightgbm":
        model_payload["best_lightgbm_params"] = best_params
        model_payload["best_lightgbm_validation_mae_seconds"] = best_val_mae_seconds
    if primary_model_name == "catboost":
        model_payload["best_catboost_params"] = best_params
        model_payload["best_catboost_validation_mae_seconds"] = best_val_mae_seconds
        if "best_iteration" in main_run:
            model_payload["best_catboost_iteration"] = int(main_run["best_iteration"])
        model_payload["catboost_categorical_cols"] = main_run.get("catboost_categorical_cols", [])
        model_payload["catboost_numeric_cols"] = main_run.get("catboost_numeric_cols", [])
    if champion_scaling_lookup_path:
        model_payload["champion_scaling_lookup_csv"] = champion_scaling_lookup_path
    if champion_scaling_lookup_artifact_path:
        model_payload["champion_scaling_lookup_artifact"] = champion_scaling_lookup_artifact_path
    if catboost_feature_importance_path:
        model_payload["catboost_feature_importance_csv"] = catboost_feature_importance_path

    joblib.dump(model_payload, Path(output_dir) / "model_artifacts.joblib")
    LOGGER.info("Saved model artifacts.")

    split_summary = {
        "train_rows": int(len(split.train)),
        "validation_rows": int(len(split.validation)),
        "test_rows": int(len(split.test)),
        "train_date_min": str(split.train["date"].min()),
        "train_date_max": str(split.train["date"].max()),
        "validation_date_min": str(split.validation["date"].min()),
        "validation_date_max": str(split.validation["date"].max()),
        "test_date_min": str(split.test["date"].min()),
        "test_date_max": str(split.test["date"].max()),
    }
    write_json(split_summary, Path(reports_dir) / "split_summary.json")
    LOGGER.info("Saved split summary.")

    write_json(
        {
            "primary_model": primary_model_name,
            "metrics_by_model": metrics_by_model,
            "primary_model_best_params": best_params,
            "primary_model_validation_mae_seconds": best_val_mae_seconds,
            "ridge_alpha": best_alpha,
            "split": split_summary,
            "champion_scaling_ablation": champion_scaling_ablation_payload,
            "feature_group_ablation": feature_group_ablation_payload,
        },
        Path(reports_dir) / "metrics_summary.json",
    )
    LOGGER.info("Saved metrics summary.")

    match_df.to_csv(Path(output_dir) / "match_level_table.csv", index=False)
    LOGGER.info("Saved match-level table CSV.")

    return {
        "primary_model": primary_model_name,
        "metrics_by_model": metrics_by_model,
        "benchmark_csv": str(benchmark_path),
        "champion_scaling_ablation_csv": str(champion_scaling_ablation_path) if champion_scaling_ablation_path else None,
        "feature_group_ablation_csv": str(feature_group_ablation_csv_path) if feature_group_ablation_csv_path else None,
        "feature_group_ablation_json": str(feature_group_ablation_json_path) if feature_group_ablation_json_path else None,
        "catboost_feature_importance_csv": catboost_feature_importance_path,
        "artifacts_path": str(Path(output_dir) / "model_artifacts.joblib"),
        "reports_dir": str(reports_dir),
    }
