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
    pinball_loss,
    quantile_interval_metrics_seconds,
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


def _format_quantile_label(alpha: float) -> str:
    return f"p{int(round(float(alpha) * 100.0)):02d}"


def _sorted_quantile_levels(config: PipelineConfig) -> List[float]:
    return sorted(float(level) for level in config.quantile_levels)


def _enforce_non_crossing(prediction_map: Dict[float, np.ndarray]) -> Dict[float, np.ndarray]:
    if not prediction_map:
        return {}

    levels = sorted(prediction_map)
    stacked = np.column_stack([np.asarray(prediction_map[level], dtype=float) for level in levels])
    stacked = np.sort(stacked, axis=1)
    return {level: stacked[:, idx] for idx, level in enumerate(levels)}


def _quantile_calibration_rows(
    y_true_seconds: np.ndarray,
    prediction_seconds_by_quantile: Dict[float, np.ndarray],
    split_name: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for alpha in sorted(prediction_seconds_by_quantile):
        preds = np.asarray(prediction_seconds_by_quantile[alpha], dtype=float)
        empirical = float(np.mean(np.asarray(y_true_seconds, dtype=float) <= preds))
        rows.append(
            {
                "split": split_name,
                "quantile": float(alpha),
                "quantile_label": _format_quantile_label(alpha),
                "nominal_coverage": float(alpha),
                "empirical_cdf": empirical,
                "calibration_error": empirical - float(alpha),
            }
        )
    return rows


def _volatility_score(width_minutes: np.ndarray, reference_widths_minutes: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference_widths_minutes, dtype=float))
    if reference.size == 0:
        return np.zeros(len(width_minutes), dtype=float)

    clipped = np.asarray(width_minutes, dtype=float)
    ranks = np.searchsorted(reference, clipped, side="right")
    return ranks / float(reference.size)


def _apply_volatility_labels(
    width_minutes: np.ndarray,
    low_threshold_minutes: float,
    high_threshold_minutes: float,
) -> List[str]:
    labels: List[str] = []
    for width in np.asarray(width_minutes, dtype=float):
        if width <= low_threshold_minutes:
            labels.append("low volatility")
        elif width <= high_threshold_minutes:
            labels.append("medium volatility")
        else:
            labels.append("high volatility")
    return labels


def _build_quantile_prediction_frame(
    source_df: pd.DataFrame,
    actual_seconds: np.ndarray,
    prediction_seconds_by_quantile: Dict[float, np.ndarray],
    lower_alpha: float,
    median_alpha: float,
    upper_alpha: float,
    reference_widths_minutes: np.ndarray,
    low_threshold_minutes: float,
    high_threshold_minutes: float,
) -> pd.DataFrame:
    out = source_df[["gameid", "league", "patch", "date"]].copy().reset_index(drop=True)
    out["actual_duration_seconds"] = np.asarray(actual_seconds, dtype=float)
    out["actual_duration_minutes"] = out["actual_duration_seconds"] / 60.0

    for alpha in sorted(prediction_seconds_by_quantile):
        label = _format_quantile_label(alpha)
        preds = np.asarray(prediction_seconds_by_quantile[alpha], dtype=float)
        out[f"pred_{label}_seconds"] = preds
        out[f"pred_{label}_minutes"] = preds / 60.0

    lower_label = _format_quantile_label(lower_alpha)
    median_label = _format_quantile_label(median_alpha)
    upper_label = _format_quantile_label(upper_alpha)

    out["interval_width_seconds"] = out[f"pred_{upper_label}_seconds"] - out[f"pred_{lower_label}_seconds"]
    out["interval_width_minutes"] = out["interval_width_seconds"] / 60.0
    out["prediction_error_from_p50_seconds"] = out[f"pred_{median_label}_seconds"] - out["actual_duration_seconds"]
    out["prediction_error_from_p50_minutes"] = out["prediction_error_from_p50_seconds"] / 60.0

    widths_minutes = out["interval_width_minutes"].to_numpy(dtype=float)
    out["volatility_score"] = _volatility_score(widths_minutes, reference_widths_minutes)
    out["volatility_flag"] = _apply_volatility_labels(widths_minutes, low_threshold_minutes, high_threshold_minutes)
    return out


def _train_catboost_quantile_suite(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    X_train_val: pd.DataFrame,
    y_train_val: np.ndarray,
    config: PipelineConfig,
) -> Dict[str, Any]:
    if CatBoostRegressor is None:
        raise RuntimeError("CatBoost is not installed. Install `catboost` to use quantile regression.")

    quantile_levels = _sorted_quantile_levels(config)
    X_train_cb, cat_feature_indices_train, _, _ = _prepare_catboost_frame(X_train)
    X_val_cb, _, _, _ = _prepare_catboost_frame(X_val)
    X_test_cb, _, _, _ = _prepare_catboost_frame(X_test)
    X_train_val_cb, cat_feature_indices_train_val, _, _ = _prepare_catboost_frame(X_train_val)

    val_predictions: Dict[float, np.ndarray] = {}
    test_predictions: Dict[float, np.ndarray] = {}
    quantile_models: Dict[str, Any] = {}
    quantile_params: Dict[str, Dict[str, Any]] = {}
    quantile_iterations: Dict[str, int] = {}
    validation_pinball_losses: Dict[str, float] = {}

    for alpha in quantile_levels:
        quantile_label = _format_quantile_label(alpha)
        best_loss = float("inf")
        best_params: Dict[str, Any] = {}
        best_iteration = 0
        best_model: Any = None
        loss_function = f"Quantile:alpha={alpha}"

        for idx, params in enumerate(config.catboost_param_grid):
            candidate = CatBoostRegressor(
                loss_function=loss_function,
                eval_metric=loss_function,
                iterations=config.catboost_iterations,
                random_seed=config.random_seed,
                verbose=False,
                **params,
            )
            candidate.fit(
                X_train_cb,
                y_train,
                cat_features=cat_feature_indices_train,
                eval_set=(X_val_cb, y_val),
                use_best_model=True,
                early_stopping_rounds=config.catboost_early_stopping_rounds,
            )
            val_pred = np.asarray(candidate.predict(X_val_cb), dtype=float)
            val_loss = pinball_loss(y_val, val_pred, alpha)
            LOGGER.info(
                "CatBoost quantile candidate %d for %s validation pinball loss (target units): %.4f",
                idx + 1,
                quantile_label,
                val_loss,
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_params = dict(params)
                best_iteration = int(candidate.get_best_iteration())
                best_model = candidate

        if best_model is None:
            raise RuntimeError(f"CatBoost quantile tuning failed for quantile {quantile_label}.")

        if best_iteration <= 0:
            best_iteration = config.catboost_iterations

        final_model = CatBoostRegressor(
            loss_function=loss_function,
            eval_metric=loss_function,
            iterations=max(int(best_iteration), 1),
            random_seed=config.random_seed,
            verbose=False,
            **best_params,
        )
        final_model.fit(X_train_val_cb, y_train_val, cat_features=cat_feature_indices_train_val)

        val_predictions[alpha] = np.asarray(best_model.predict(X_val_cb), dtype=float)
        test_predictions[alpha] = np.asarray(final_model.predict(X_test_cb), dtype=float)
        quantile_models[str(alpha)] = final_model
        quantile_params[quantile_label] = best_params
        quantile_iterations[quantile_label] = int(best_iteration)
        validation_pinball_losses[quantile_label] = float(best_loss)

    if config.quantile_enforce_non_crossing:
        val_predictions = _enforce_non_crossing(val_predictions)
        test_predictions = _enforce_non_crossing(test_predictions)

    return {
        "quantile_levels": quantile_levels,
        "val_predictions_model_unit": val_predictions,
        "test_predictions_model_unit": test_predictions,
        "models": quantile_models,
        "best_params_by_quantile": quantile_params,
        "best_iteration_by_quantile": quantile_iterations,
        "validation_pinball_loss_by_quantile": validation_pinball_losses,
        "X_val_transformed": X_val_cb,
        "X_test_transformed": X_test_cb,
        "y_val": np.asarray(y_val, dtype=float),
        "y_test": np.asarray(y_test, dtype=float),
    }


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


def _same_feature_config(left: PipelineConfig, right: PipelineConfig) -> bool:
    return (
        left.use_extended_rolling_team_priors == right.use_extended_rolling_team_priors
        and left.use_draft_summary_features == right.use_draft_summary_features
        and left.use_draft_interaction_features == right.use_draft_interaction_features
        and left.use_draft_conditional_behaviour_features == right.use_draft_conditional_behaviour_features
        and left.use_turret_prior_features == right.use_turret_prior_features
        and left.use_extended_turret_prior_features == right.use_extended_turret_prior_features
        and left.use_champion_scaling_features == right.use_champion_scaling_features
        and left.use_sparse_champion_indicator_features == right.use_sparse_champion_indicator_features
        and left.use_pick_order_champion_features == right.use_pick_order_champion_features
        and left.use_series_game_number_feature == right.use_series_game_number_feature
        and left.use_role_specific_draft_features == right.use_role_specific_draft_features
        and left.use_bag_of_champions_fallback == right.use_bag_of_champions_fallback
    )


def _refinement_ablation_configs(base_config: PipelineConfig) -> List[Tuple[str, PipelineConfig]]:
    variants = [
        ("A_current_best", base_config),
        (
            "B_plus_game",
            replace(base_config, use_series_game_number_feature=True),
        ),
        (
            "C_minus_sparse_indicators",
            replace(base_config, use_sparse_champion_indicator_features=False),
        ),
        (
            "D_plus_game_minus_sparse_indicators",
            replace(
                base_config,
                use_series_game_number_feature=True,
                use_sparse_champion_indicator_features=False,
            ),
        ),
        (
            "E_plus_game_role_slots_only",
            replace(
                base_config,
                use_series_game_number_feature=True,
                use_sparse_champion_indicator_features=False,
                use_pick_order_champion_features=False,
            ),
        ),
    ]
    return variants


def _turret_feature_ablation_configs(base_config: PipelineConfig) -> List[Tuple[str, PipelineConfig]]:
    variants = [
        (
            "A_current_preferred",
            replace(
                base_config,
                use_turret_prior_features=False,
                use_extended_turret_prior_features=False,
            ),
        ),
        (
            "B_plus_core_turret_priors",
            replace(
                base_config,
                use_turret_prior_features=True,
                use_extended_turret_prior_features=False,
            ),
        ),
        (
            "C_plus_full_turret_priors",
            replace(
                base_config,
                use_turret_prior_features=True,
                use_extended_turret_prior_features=True,
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
    variant_train_fn = _train_catboost_variant if config.primary_model == "catboost" else _train_lightgbm_variant
    main_run = variant_train_fn(split=split, config=config)
    selected_run = main_run
    selected_config = config
    selected_variant_label = "A_current_best"

    refinement_ablation_payload: Dict[str, Any] = {"enabled": False}
    refinement_ablation_csv_path: Path | None = None
    refinement_ablation_json_path: Path | None = None
    if config.run_refinement_ablation:
        refinement_variants = _refinement_ablation_configs(config)
        refinement_runs: Dict[str, Dict[str, Any]] = {}
        refinement_variant_configs: Dict[str, PipelineConfig] = {}
        refinement_rows: List[Dict[str, Any]] = []

        for label, variant_cfg in refinement_variants:
            LOGGER.info(
                "Running refinement variant=%s (game=%s, sparse_indicators=%s, pick_order_slots=%s)",
                label,
                variant_cfg.use_series_game_number_feature,
                variant_cfg.use_sparse_champion_indicator_features,
                variant_cfg.use_pick_order_champion_features,
            )
            variant_run = main_run if _same_feature_config(variant_cfg, config) else variant_train_fn(split=split, config=variant_cfg)
            refinement_runs[label] = variant_run
            refinement_variant_configs[label] = variant_cfg

            row = {
                "variant": label,
                "use_series_game_number_feature": bool(variant_cfg.use_series_game_number_feature),
                "use_sparse_champion_indicator_features": bool(variant_cfg.use_sparse_champion_indicator_features),
                "use_pick_order_champion_features": bool(variant_cfg.use_pick_order_champion_features),
                "feature_count": int(len(variant_run["feature_names"])),
            }
            row.update(variant_run["metrics"])
            refinement_rows.append(row)

        refinement_df = pd.DataFrame(refinement_rows).sort_values("mae_minutes").reset_index(drop=True)
        best_row = refinement_df.iloc[0]
        selected_variant_label = str(best_row["variant"])
        selected_run = refinement_runs[selected_variant_label]
        selected_config = refinement_variant_configs[selected_variant_label]

        baseline_row = refinement_df.loc[refinement_df["variant"] == "A_current_best"].iloc[0]
        refinement_df["mae_minutes_delta_vs_A"] = refinement_df["mae_minutes"] - float(baseline_row["mae_minutes"])
        refinement_df["rmse_minutes_delta_vs_A"] = refinement_df["rmse_minutes"] - float(baseline_row["rmse_minutes"])
        refinement_df["median_absolute_error_minutes_delta_vs_A"] = (
            refinement_df["median_absolute_error_minutes"] - float(baseline_row["median_absolute_error_minutes"])
        )
        refinement_df["within_2_minutes_accuracy_delta_vs_A"] = (
            refinement_df["within_2_minutes_accuracy"] - float(baseline_row["within_2_minutes_accuracy"])
        )
        refinement_df["within_5_minutes_accuracy_delta_vs_A"] = (
            refinement_df["within_5_minutes_accuracy"] - float(baseline_row["within_5_minutes_accuracy"])
        )

        refinement_ablation_csv_path = Path(reports_dir) / "refinement_ablation.csv"
        refinement_df.to_csv(refinement_ablation_csv_path, index=False)
        refinement_ablation_json_path = Path(reports_dir) / "refinement_ablation.json"
        write_json(
            {
                "best_variant": selected_variant_label,
                "best_variant_config": {
                    "use_series_game_number_feature": bool(selected_config.use_series_game_number_feature),
                    "use_sparse_champion_indicator_features": bool(selected_config.use_sparse_champion_indicator_features),
                    "use_pick_order_champion_features": bool(selected_config.use_pick_order_champion_features),
                },
                "variants": refinement_rows,
            },
            refinement_ablation_json_path,
        )
        LOGGER.info("Saved refinement ablation to %s and %s", refinement_ablation_csv_path, refinement_ablation_json_path)

        refinement_ablation_payload = {
            "enabled": True,
            "best_variant": selected_variant_label,
            "ablation_csv": str(refinement_ablation_csv_path),
            "ablation_json": str(refinement_ablation_json_path),
        }

    turret_feature_ablation_payload: Dict[str, Any] = {"enabled": False}
    turret_feature_ablation_csv_path: Path | None = None
    turret_feature_ablation_json_path: Path | None = None
    turret_feature_summary_path: Path | None = None
    if config.run_turret_feature_ablation:
        turret_variants = _turret_feature_ablation_configs(selected_config)
        turret_runs: Dict[str, Dict[str, Any]] = {}
        turret_variant_configs: Dict[str, PipelineConfig] = {}
        turret_rows: List[Dict[str, Any]] = []

        for label, variant_cfg in turret_variants:
            LOGGER.info(
                "Running turret-feature variant=%s (turret=%s, extended_turret=%s)",
                label,
                variant_cfg.use_turret_prior_features,
                variant_cfg.use_extended_turret_prior_features,
            )
            variant_run = (
                selected_run
                if _same_feature_config(variant_cfg, selected_config)
                else variant_train_fn(split=split, config=variant_cfg)
            )
            turret_runs[label] = variant_run
            turret_variant_configs[label] = variant_cfg

            row = {
                "variant": label,
                "use_turret_prior_features": bool(variant_cfg.use_turret_prior_features),
                "use_extended_turret_prior_features": bool(variant_cfg.use_extended_turret_prior_features),
                "feature_count": int(len(variant_run["feature_names"])),
            }
            row.update(variant_run["metrics"])
            turret_rows.append(row)

        turret_df = pd.DataFrame(turret_rows).sort_values("mae_minutes").reset_index(drop=True)
        best_row = turret_df.iloc[0]
        selected_variant_label = str(best_row["variant"])
        selected_run = turret_runs[selected_variant_label]
        selected_config = turret_variant_configs[selected_variant_label]

        baseline_row = turret_df.loc[turret_df["variant"] == "A_current_preferred"].iloc[0]
        turret_df["mae_minutes_delta_vs_A"] = turret_df["mae_minutes"] - float(baseline_row["mae_minutes"])
        turret_df["rmse_minutes_delta_vs_A"] = turret_df["rmse_minutes"] - float(baseline_row["rmse_minutes"])
        turret_df["median_absolute_error_minutes_delta_vs_A"] = (
            turret_df["median_absolute_error_minutes"] - float(baseline_row["median_absolute_error_minutes"])
        )
        turret_df["within_2_minutes_accuracy_delta_vs_A"] = (
            turret_df["within_2_minutes_accuracy"] - float(baseline_row["within_2_minutes_accuracy"])
        )
        turret_df["within_5_minutes_accuracy_delta_vs_A"] = (
            turret_df["within_5_minutes_accuracy"] - float(baseline_row["within_5_minutes_accuracy"])
        )

        turret_feature_ablation_csv_path = Path(reports_dir) / "turret_feature_ablation.csv"
        turret_df.to_csv(turret_feature_ablation_csv_path, index=False)
        turret_feature_ablation_json_path = Path(reports_dir) / "turret_feature_ablation.json"
        write_json(
            {
                "best_variant": selected_variant_label,
                "best_variant_config": {
                    "use_turret_prior_features": bool(selected_config.use_turret_prior_features),
                    "use_extended_turret_prior_features": bool(selected_config.use_extended_turret_prior_features),
                },
                "source_columns_available": [
                    "firsttower",
                    "towers",
                    "opp_towers",
                    "firstmidtower",
                    "firsttothreetowers",
                ],
                "timed_turret_columns_available": [],
                "variants": turret_rows,
            },
            turret_feature_ablation_json_path,
        )

        summary_lines = [
            "# Turret Feature Summary",
            "",
            "Added leakage-safe historical turret priors using only prior matches.",
            "",
            "Available source columns:",
            "- `firsttower`",
            "- `towers`",
            "- `opp_towers`",
            "- `firstmidtower`",
            "- `firsttothreetowers`",
            "",
            "Unavailable timed turret sources:",
            "- no `turretdiffat15` / `towersat15` / `time_to_first_turret` style columns were found",
            "",
            f"Best variant: `{selected_variant_label}`",
            f"- use_turret_prior_features: `{selected_config.use_turret_prior_features}`",
            f"- use_extended_turret_prior_features: `{selected_config.use_extended_turret_prior_features}`",
        ]
        turret_feature_summary_path = Path(reports_dir) / "turret_feature_summary.md"
        turret_feature_summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

        LOGGER.info(
            "Saved turret feature ablation to %s and %s",
            turret_feature_ablation_csv_path,
            turret_feature_ablation_json_path,
        )
        turret_feature_ablation_payload = {
            "enabled": True,
            "best_variant": selected_variant_label,
            "ablation_csv": str(turret_feature_ablation_csv_path),
            "ablation_json": str(turret_feature_ablation_json_path),
            "summary_markdown": str(turret_feature_summary_path),
        }

    primary_model_name: str = selected_run["model_name"]
    feature_builder: DraftFeatureBuilder = selected_run["feature_builder"]
    categorical_cols: List[str] = selected_run["categorical_cols"]
    numeric_cols: List[str] = selected_run["numeric_cols"]
    X_train: pd.DataFrame = selected_run["X_train"]
    X_val: pd.DataFrame = selected_run["X_val"]
    X_test: pd.DataFrame = selected_run["X_test"]
    y_train: np.ndarray = selected_run["y_train"]
    y_val: np.ndarray = selected_run["y_val"]
    y_test: np.ndarray = selected_run["y_test"]
    y_test_seconds: np.ndarray = selected_run["y_test_seconds"]
    unit_to_seconds: float = selected_run["unit_to_seconds"]
    X_train_val: pd.DataFrame = selected_run["X_train_val"]
    y_train_val: np.ndarray = selected_run["y_train_val"]
    preprocessor_final = selected_run["preprocessor_final"]
    X_test_transformed = selected_run["X_test_transformed"]
    final_primary_model = selected_run["model"]
    best_params: Dict[str, float] = selected_run["best_params"]
    best_val_mae_seconds: float = selected_run["best_val_mae_seconds"]
    feature_names: List[str] = selected_run["feature_names"]
    primary_preds_model_unit: np.ndarray = selected_run["preds_model_unit"]

    _save_feature_column_artifacts(
        reports_dir=reports_dir,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        transformed_feature_names=feature_names,
    )

    quantile_payload: Dict[str, Any] = {"enabled": False}
    quantile_models: Dict[str, Any] = {}
    quantile_prediction_paths: Dict[str, str] = {}
    quantile_metrics_summary_path: Path | None = None
    quantile_comparison_path: Path | None = None
    quantile_diagnostics_path: Path | None = None
    if selected_config.enable_quantile_regression:
        quantile_run = _train_catboost_quantile_suite(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            X_train_val=X_train_val,
            y_train_val=y_train_val,
            config=selected_config,
        )

        quantile_levels = quantile_run["quantile_levels"]
        lower_alpha = float(quantile_levels[0])
        median_alpha = 0.5
        upper_alpha = float(quantile_levels[-1])

        val_prediction_seconds = {
            alpha: np.asarray(preds, dtype=float) * unit_to_seconds
            for alpha, preds in quantile_run["val_predictions_model_unit"].items()
        }
        test_prediction_seconds = {
            alpha: np.asarray(preds, dtype=float) * unit_to_seconds
            for alpha, preds in quantile_run["test_predictions_model_unit"].items()
        }
        y_val_seconds = np.asarray(y_val, dtype=float) * unit_to_seconds

        val_width_minutes = (val_prediction_seconds[upper_alpha] - val_prediction_seconds[lower_alpha]) / 60.0
        low_q, high_q = [float(level) for level in selected_config.volatility_threshold_quantiles]
        low_threshold_minutes, high_threshold_minutes = np.quantile(val_width_minutes, [low_q, high_q]).tolist()

        quantile_val_metrics = quantile_interval_metrics_seconds(
            y_true_sec=y_val_seconds,
            y_pred_p50_sec=val_prediction_seconds[median_alpha],
            y_pred_lower_sec=val_prediction_seconds[lower_alpha],
            y_pred_upper_sec=val_prediction_seconds[upper_alpha],
            lower_alpha=lower_alpha,
            upper_alpha=upper_alpha,
        )
        quantile_test_metrics = quantile_interval_metrics_seconds(
            y_true_sec=y_test_seconds,
            y_pred_p50_sec=test_prediction_seconds[median_alpha],
            y_pred_lower_sec=test_prediction_seconds[lower_alpha],
            y_pred_upper_sec=test_prediction_seconds[upper_alpha],
            lower_alpha=lower_alpha,
            upper_alpha=upper_alpha,
        )

        val_prediction_frame = _build_quantile_prediction_frame(
            source_df=split.validation,
            actual_seconds=y_val_seconds,
            prediction_seconds_by_quantile=val_prediction_seconds,
            lower_alpha=lower_alpha,
            median_alpha=median_alpha,
            upper_alpha=upper_alpha,
            reference_widths_minutes=val_width_minutes,
            low_threshold_minutes=float(low_threshold_minutes),
            high_threshold_minutes=float(high_threshold_minutes),
        )
        test_prediction_frame = _build_quantile_prediction_frame(
            source_df=split.test,
            actual_seconds=y_test_seconds,
            prediction_seconds_by_quantile=test_prediction_seconds,
            lower_alpha=lower_alpha,
            median_alpha=median_alpha,
            upper_alpha=upper_alpha,
            reference_widths_minutes=val_width_minutes,
            low_threshold_minutes=float(low_threshold_minutes),
            high_threshold_minutes=float(high_threshold_minutes),
        )

        val_predictions_path = Path(reports_dir) / "quantile_predictions_val.csv"
        test_predictions_path = Path(reports_dir) / "quantile_predictions_test.csv"
        val_prediction_frame.to_csv(val_predictions_path, index=False)
        test_prediction_frame.to_csv(test_predictions_path, index=False)

        calibration_rows = _quantile_calibration_rows(y_val_seconds, val_prediction_seconds, "validation")
        calibration_rows.extend(_quantile_calibration_rows(y_test_seconds, test_prediction_seconds, "test"))

        quantile_metrics_summary_path = Path(reports_dir) / "quantile_metrics_summary.json"
        write_json(
            {
                "quantile_levels": quantile_levels,
                "lower_quantile": lower_alpha,
                "median_quantile": median_alpha,
                "upper_quantile": upper_alpha,
                "validation_metrics": quantile_val_metrics,
                "test_metrics": quantile_test_metrics,
                "validation_pinball_loss_by_quantile": quantile_run["validation_pinball_loss_by_quantile"],
                "volatility_threshold_quantiles": selected_config.volatility_threshold_quantiles,
                "volatility_thresholds_minutes": {
                    "low": float(low_threshold_minutes),
                    "high": float(high_threshold_minutes),
                },
                "prediction_tables": {
                    "validation": str(val_predictions_path),
                    "test": str(test_predictions_path),
                },
            },
            quantile_metrics_summary_path,
        )

        point_metrics = selected_run["metrics"]
        quantile_comparison_path = Path(reports_dir) / "quantile_vs_point_comparison.json"
        write_json(
            {
                "point_model_name": primary_model_name,
                "point_model_test_metrics": point_metrics,
                "quantile_p50_test_metrics": {
                    "mae_minutes": quantile_test_metrics["mae_minutes"],
                    "rmse_minutes": quantile_test_metrics["rmse_minutes"],
                    "median_absolute_error_minutes": quantile_test_metrics["median_absolute_error_minutes"],
                    "within_2_minutes_accuracy": quantile_test_metrics["within_2_minutes_accuracy"],
                    "within_5_minutes_accuracy": quantile_test_metrics["within_5_minutes_accuracy"],
                    "mae_seconds": quantile_test_metrics["mae_seconds"],
                },
                "delta_p50_minus_point": {
                    "mae_minutes": quantile_test_metrics["mae_minutes"] - point_metrics["mae_minutes"],
                    "rmse_minutes": quantile_test_metrics["rmse_minutes"] - point_metrics["rmse_minutes"],
                    "median_absolute_error_minutes": (
                        quantile_test_metrics["median_absolute_error_minutes"]
                        - point_metrics["median_absolute_error_minutes"]
                    ),
                    "within_2_minutes_accuracy": (
                        quantile_test_metrics["within_2_minutes_accuracy"]
                        - point_metrics["within_2_minutes_accuracy"]
                    ),
                    "within_5_minutes_accuracy": (
                        quantile_test_metrics["within_5_minutes_accuracy"]
                        - point_metrics["within_5_minutes_accuracy"]
                    ),
                },
                "interval_summary": {
                    "average_interval_width_minutes": quantile_test_metrics["average_interval_width_minutes"],
                    "median_interval_width_minutes": quantile_test_metrics["median_interval_width_minutes"],
                    "interval_coverage": quantile_test_metrics["interval_coverage"],
                    "undercoverage_rate": quantile_test_metrics["undercoverage_rate"],
                    "overcoverage_rate": quantile_test_metrics["overcoverage_rate"],
                },
            },
            quantile_comparison_path,
        )

        quantile_diagnostics_path = Path(reports_dir) / "quantile_interval_diagnostics.json"
        write_json(
            {
                "quantile_levels": quantile_levels,
                "calibration_rows": calibration_rows,
                "validation_width_minutes_summary": {
                    "mean": float(np.mean(val_prediction_frame["interval_width_minutes"])),
                    "median": float(np.median(val_prediction_frame["interval_width_minutes"])),
                },
                "test_width_minutes_summary": {
                    "mean": float(np.mean(test_prediction_frame["interval_width_minutes"])),
                    "median": float(np.median(test_prediction_frame["interval_width_minutes"])),
                },
                "volatility_bucket_counts": {
                    "validation": val_prediction_frame["volatility_flag"].value_counts().to_dict(),
                    "test": test_prediction_frame["volatility_flag"].value_counts().to_dict(),
                },
            },
            quantile_diagnostics_path,
        )

        quantile_models = quantile_run["models"]
        quantile_prediction_paths = {
            "validation": str(val_predictions_path),
            "test": str(test_predictions_path),
        }
        quantile_payload = {
            "enabled": True,
            "quantile_levels": quantile_levels,
            "lower_quantile": lower_alpha,
            "median_quantile": median_alpha,
            "upper_quantile": upper_alpha,
            "metrics_summary_json": str(quantile_metrics_summary_path),
            "comparison_json": str(quantile_comparison_path),
            "diagnostics_json": str(quantile_diagnostics_path),
            "prediction_tables": quantile_prediction_paths,
            "volatility_thresholds_minutes": {
                "low": float(low_threshold_minutes),
                "high": float(high_threshold_minutes),
            },
        }
        LOGGER.info("Saved quantile regression artifacts to %s", reports_dir)

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
            same_as_main = _same_feature_config(variant_cfg, config)
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
        "config": selected_config.to_dict(),
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
        if "best_iteration" in selected_run:
            model_payload["best_catboost_iteration"] = int(selected_run["best_iteration"])
        model_payload["catboost_categorical_cols"] = selected_run.get("catboost_categorical_cols", [])
        model_payload["catboost_numeric_cols"] = selected_run.get("catboost_numeric_cols", [])
    if champion_scaling_lookup_path:
        model_payload["champion_scaling_lookup_csv"] = champion_scaling_lookup_path
    if champion_scaling_lookup_artifact_path:
        model_payload["champion_scaling_lookup_artifact"] = champion_scaling_lookup_artifact_path
    if catboost_feature_importance_path:
        model_payload["catboost_feature_importance_csv"] = catboost_feature_importance_path
    if quantile_payload["enabled"]:
        model_payload["quantile_regression"] = {
            "enabled": True,
            "quantile_levels": quantile_payload["quantile_levels"],
            "lower_quantile": quantile_payload["lower_quantile"],
            "median_quantile": quantile_payload["median_quantile"],
            "upper_quantile": quantile_payload["upper_quantile"],
            "prediction_tables": quantile_prediction_paths,
            "volatility_thresholds_minutes": quantile_payload["volatility_thresholds_minutes"],
        }
        model_payload["quantile_models"] = quantile_models

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
            "selected_variant": selected_variant_label,
            "primary_model": primary_model_name,
            "metrics_by_model": metrics_by_model,
            "primary_model_best_params": best_params,
            "primary_model_validation_mae_seconds": best_val_mae_seconds,
            "ridge_alpha": best_alpha,
            "split": split_summary,
            "refinement_ablation": refinement_ablation_payload,
            "turret_feature_ablation": turret_feature_ablation_payload,
            "champion_scaling_ablation": champion_scaling_ablation_payload,
            "feature_group_ablation": feature_group_ablation_payload,
            "quantile_regression": quantile_payload,
        },
        Path(reports_dir) / "metrics_summary.json",
    )
    LOGGER.info("Saved metrics summary.")

    match_df.to_csv(Path(output_dir) / "match_level_table.csv", index=False)
    LOGGER.info("Saved match-level table CSV.")

    return {
        "selected_variant": selected_variant_label,
        "primary_model": primary_model_name,
        "metrics_by_model": metrics_by_model,
        "benchmark_csv": str(benchmark_path),
        "refinement_ablation_csv": str(refinement_ablation_csv_path) if refinement_ablation_csv_path else None,
        "refinement_ablation_json": str(refinement_ablation_json_path) if refinement_ablation_json_path else None,
        "turret_feature_ablation_csv": str(turret_feature_ablation_csv_path) if turret_feature_ablation_csv_path else None,
        "turret_feature_ablation_json": str(turret_feature_ablation_json_path) if turret_feature_ablation_json_path else None,
        "champion_scaling_ablation_csv": str(champion_scaling_ablation_path) if champion_scaling_ablation_path else None,
        "feature_group_ablation_csv": str(feature_group_ablation_csv_path) if feature_group_ablation_csv_path else None,
        "feature_group_ablation_json": str(feature_group_ablation_json_path) if feature_group_ablation_json_path else None,
        "quantile_metrics_summary_json": str(quantile_metrics_summary_path) if quantile_metrics_summary_path else None,
        "quantile_vs_point_comparison_json": str(quantile_comparison_path) if quantile_comparison_path else None,
        "quantile_interval_diagnostics_json": str(quantile_diagnostics_path) if quantile_diagnostics_path else None,
        "catboost_feature_importance_csv": catboost_feature_importance_path,
        "artifacts_path": str(Path(output_dir) / "model_artifacts.joblib"),
        "reports_dir": str(reports_dir),
    }
