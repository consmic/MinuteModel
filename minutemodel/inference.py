from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

try:
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    shap = None


def load_artifacts(path: str | Path) -> Dict[str, Any]:
    artifacts = joblib.load(path)
    required = {"model", "feature_builder"}
    missing = sorted(required - set(artifacts.keys()))
    if missing:
        raise ValueError(f"Model artifacts missing keys: {missing}")

    artifacts.setdefault("preprocessor", None)
    artifacts.setdefault("feature_names", [])
    artifacts.setdefault("primary_model", "lightgbm")
    _patch_simple_imputer_compat(artifacts)
    return artifacts


def _patch_simple_imputer_compat(root: Any) -> None:
    """Bridge sklearn private attr renames across versions for unpickled pipelines."""
    seen: set[int] = set()

    def _walk(node: Any) -> None:
        node_id = id(node)
        if node_id in seen:
            return
        seen.add(node_id)

        if isinstance(node, SimpleImputer):
            if hasattr(node, "_fit_dtype") and not hasattr(node, "_fill_dtype"):
                setattr(node, "_fill_dtype", getattr(node, "_fit_dtype"))
            if hasattr(node, "_fill_dtype") and not hasattr(node, "_fit_dtype"):
                setattr(node, "_fit_dtype", getattr(node, "_fill_dtype"))

        if isinstance(node, dict):
            for value in node.values():
                _walk(value)
            return
        if isinstance(node, (list, tuple, set)):
            for value in node:
                _walk(value)
            return

        if hasattr(node, "__dict__"):
            for value in vars(node).values():
                _walk(value)

    _walk(root)


def _prepare_single_input(payload: Dict[str, Any]) -> pd.DataFrame:
    row = payload.copy()

    # Ensure list-form champions are preserved for bag-of-champions features.
    row.setdefault("blue_draft_champions", [])
    row.setdefault("red_draft_champions", [])

    if not isinstance(row["blue_draft_champions"], list):
        row["blue_draft_champions"] = [row["blue_draft_champions"]]
    if not isinstance(row["red_draft_champions"], list):
        row["red_draft_champions"] = [row["red_draft_champions"]]

    return pd.DataFrame([row])


def _explain_single_prediction(
    model,
    transformed_row,
    feature_names: List[str],
    top_k: int = 10,
) -> Dict[str, Any]:
    if shap is None:
        return {
            "warning": "SHAP is not installed in this environment; explanation unavailable.",
            "top_feature_contributions": [],
        }

    if hasattr(transformed_row, "toarray"):
        transformed_row = transformed_row.toarray()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_row)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    values = shap_values[0]
    top_idx = np.argsort(np.abs(values))[::-1][:top_k]
    top_features = [
        {
            "feature": feature_names[int(idx)],
            "shap_value": float(values[int(idx)]),
        }
        for idx in top_idx
    ]

    return {
        "expected_value": float(np.array(explainer.expected_value).reshape(-1)[0]),
        "top_feature_contributions": top_features,
    }


def predict_single_draft(
    artifact_path: str | Path,
    draft_payload: Dict[str, Any],
    include_explanation: bool = False,
) -> Dict[str, Any]:
    artifacts = load_artifacts(artifact_path)

    model = artifacts["model"]
    preprocessor = artifacts["preprocessor"]
    feature_builder = artifacts["feature_builder"]
    feature_names = artifacts.get("feature_names", [])
    primary_model = str(artifacts.get("primary_model", "lightgbm")).lower()
    catboost_categorical_cols = artifacts.get("catboost_categorical_cols", []) or []
    catboost_numeric_cols = artifacts.get("catboost_numeric_cols", []) or []
    quantile_bundle = artifacts.get("quantile_regression", {}) or {}
    quantile_models = artifacts.get("quantile_models", {}) or {}
    config = artifacts.get("config", {})
    target_unit = str(config.get("target_unit", "seconds")).lower()
    unit_to_seconds = 60.0 if target_unit == "minutes" else 1.0

    input_df = _prepare_single_input(draft_payload)
    feature_df = feature_builder.transform(input_df)
    if preprocessor is not None:
        transformed = preprocessor.transform(feature_df)
    else:
        transformed = feature_df.copy()
        if primary_model == "catboost":
            for col in catboost_categorical_cols:
                if col in transformed.columns:
                    transformed[col] = transformed[col].astype(str).fillna("__MISSING__")
            for col in catboost_numeric_cols:
                if col in transformed.columns:
                    transformed[col] = pd.to_numeric(transformed[col], errors="coerce")

    pred_model_unit = float(model.predict(transformed)[0])
    pred_seconds = pred_model_unit * unit_to_seconds

    response = {
        "predicted_duration_seconds": pred_seconds,
        "predicted_duration_minutes": pred_seconds / 60.0,
    }

    if quantile_bundle.get("enabled") and quantile_models:
        quantile_levels = sorted(float(level) for level in quantile_bundle.get("quantile_levels", []))
        quantile_preds_model_unit: Dict[float, float] = {}
        for alpha in quantile_levels:
            model_key = str(alpha)
            quantile_model = quantile_models.get(model_key, quantile_models.get(alpha))
            if quantile_model is None:
                continue
            quantile_preds_model_unit[alpha] = float(quantile_model.predict(transformed)[0])

        if quantile_preds_model_unit:
            ordered_levels = sorted(quantile_preds_model_unit)
            ordered_values = np.sort(np.array([quantile_preds_model_unit[level] for level in ordered_levels], dtype=float))
            quantile_preds_seconds = {
                level: float(ordered_values[idx] * unit_to_seconds)
                for idx, level in enumerate(ordered_levels)
            }

            lower_alpha = float(quantile_bundle.get("lower_quantile", ordered_levels[0]))
            median_alpha = float(quantile_bundle.get("median_quantile", 0.5))
            upper_alpha = float(quantile_bundle.get("upper_quantile", ordered_levels[-1]))
            lower_pred = quantile_preds_seconds.get(lower_alpha, quantile_preds_seconds[ordered_levels[0]])
            median_pred = quantile_preds_seconds.get(median_alpha, pred_seconds)
            upper_pred = quantile_preds_seconds.get(upper_alpha, quantile_preds_seconds[ordered_levels[-1]])
            interval_width_minutes = (upper_pred - lower_pred) / 60.0

            thresholds = quantile_bundle.get("volatility_thresholds_minutes", {}) or {}
            low_threshold = float(thresholds.get("low", 0.0))
            high_threshold = float(thresholds.get("high", low_threshold))
            if interval_width_minutes <= low_threshold:
                volatility_flag = "low volatility"
            elif interval_width_minutes <= high_threshold:
                volatility_flag = "medium volatility"
            else:
                volatility_flag = "high volatility"

            response["quantile_predictions"] = {
                f"predicted_p{int(round(level * 100)):02d}_seconds": value
                for level, value in quantile_preds_seconds.items()
            }
            response["quantile_predictions"].update(
                {
                    f"predicted_p{int(round(level * 100)):02d}_minutes": value / 60.0
                    for level, value in quantile_preds_seconds.items()
                }
            )
            response["quantile_predictions"]["interval_width_seconds"] = upper_pred - lower_pred
            response["quantile_predictions"]["interval_width_minutes"] = interval_width_minutes
            response["quantile_predictions"]["volatility_flag"] = volatility_flag
            response["quantile_predictions"]["default_interval"] = {
                "lower_quantile": lower_alpha,
                "median_quantile": median_alpha,
                "upper_quantile": upper_alpha,
            }

    if include_explanation:
        explanation_feature_names = feature_names if feature_names else list(feature_df.columns)
        response["explanation"] = _explain_single_prediction(
            model=model,
            transformed_row=transformed,
            feature_names=explanation_feature_names,
            top_k=10,
        )
        response["explanation"]["model_output_unit"] = target_unit

    return response
