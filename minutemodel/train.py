from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

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


def _save_shap_diagnostics(
    model: lgb.LGBMRegressor,
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

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

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

    # Baselines (trained on train+val where no tuning is required).
    X_train_val = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_train_val = np.concatenate([y_train, y_val], axis=0)

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

    best_params, _, best_val_mae = tune_lightgbm(
        X_train=X_train_transformed,
        y_train=y_train,
        X_val=X_val_transformed,
        y_val=y_val,
        config=config,
    )
    best_val_mae_seconds = best_val_mae * unit_to_seconds

    preprocessor_final = build_preprocessor(categorical_cols=categorical_cols, numeric_cols=numeric_cols)
    X_train_val_transformed = preprocessor_final.fit_transform(X_train_val)
    X_test_transformed = preprocessor_final.transform(X_test)

    final_lgbm_model = _train_final_lightgbm(
        X_train_val=X_train_val_transformed,
        y_train_val=y_train_val,
        best_params=best_params,
        config=config,
    )

    preds_model_unit: Dict[str, np.ndarray] = {
        "global_mean": np.full(len(X_test), global_mean_value, dtype=float),
        "league_mean": league_baseline.predict(X_test),
        "league_patch_mean": league_patch_baseline.predict(X_test),
        "ridge_regression": ridge_model.predict(X_test),
        "lightgbm": final_lgbm_model.predict(X_test_transformed),
    }
    preds_seconds = {name: values * unit_to_seconds for name, values in preds_model_unit.items()}
    y_test_seconds = y_test * unit_to_seconds

    metrics_by_model: Dict[str, Dict[str, float]] = {
        model_name: regression_metrics_seconds(y_test_seconds, y_pred_seconds)
        for model_name, y_pred_seconds in preds_seconds.items()
    }

    benchmark_df = benchmarking_table(metrics_by_model)
    benchmark_path = Path(reports_dir) / "benchmark_summary.csv"
    benchmark_df.to_csv(benchmark_path)
    LOGGER.info("Saved benchmark summary to %s", benchmark_path)

    test_eval_df = split.test[["gameid", "league", "patch", "date", "target_gamelength_seconds"]].copy()
    test_eval_df = test_eval_df.rename(columns={"target_gamelength_seconds": "y_true_seconds"})
    test_eval_df["y_pred_seconds"] = preds_seconds["lightgbm"]
    test_eval_df = add_duration_bucket(test_eval_df)
    LOGGER.info("Prepared test evaluation dataframe.")

    save_residual_plots(test_eval_df, reports_dir, "lightgbm")
    LOGGER.info("Saved residual plots.")

    league_breakdown = error_breakdown_table(test_eval_df, group_col="league")
    patch_breakdown = error_breakdown_table(test_eval_df, group_col="patch")
    bucket_breakdown = error_breakdown_table(test_eval_df, group_col="duration_bucket")

    league_breakdown.to_csv(Path(reports_dir) / "error_by_league.csv", index=False)
    patch_breakdown.to_csv(Path(reports_dir) / "error_by_patch.csv", index=False)
    bucket_breakdown.to_csv(Path(reports_dir) / "error_by_duration_bucket.csv", index=False)
    LOGGER.info("Saved error breakdown tables.")

    save_error_bar_plot(league_breakdown, "league", reports_dir, "lightgbm")
    save_error_bar_plot(patch_breakdown, "patch", reports_dir, "lightgbm")
    save_error_bar_plot(bucket_breakdown, "duration_bucket", reports_dir, "lightgbm")
    LOGGER.info("Saved error breakdown plots.")

    feature_names = preprocessor_final.get_feature_names_out().tolist()
    _save_shap_diagnostics(final_lgbm_model, X_test_transformed, feature_names, Path(reports_dir) / "shap")
    LOGGER.info("SHAP diagnostics step completed.")

    model_payload = {
        "model": final_lgbm_model,
        "preprocessor": preprocessor_final,
        "feature_builder": feature_builder,
        "best_lightgbm_params": best_params,
        "best_lightgbm_validation_mae_seconds": best_val_mae_seconds,
        "ridge_alpha": best_alpha,
        "global_mean_baseline": global_mean_value * unit_to_seconds,
        "schema_report": schema_report.to_dict(),
        "config": config.to_dict(),
        "feature_names": feature_names,
    }

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
            "metrics_by_model": metrics_by_model,
            "best_lightgbm_params": best_params,
            "best_lightgbm_validation_mae_seconds": best_val_mae_seconds,
            "ridge_alpha": best_alpha,
        },
        Path(reports_dir) / "metrics_summary.json",
    )
    LOGGER.info("Saved metrics summary.")

    match_df.to_csv(Path(output_dir) / "match_level_table.csv", index=False)
    LOGGER.info("Saved match-level table CSV.")

    return {
        "metrics_by_model": metrics_by_model,
        "benchmark_csv": str(benchmark_path),
        "artifacts_path": str(Path(output_dir) / "model_artifacts.joblib"),
        "reports_dir": str(reports_dir),
    }
