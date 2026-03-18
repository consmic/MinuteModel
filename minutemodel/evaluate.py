from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error

LOGGER = logging.getLogger(__name__)


def regression_metrics_seconds(y_true_sec: np.ndarray, y_pred_sec: np.ndarray) -> Dict[str, float]:
    y_true_sec = np.asarray(y_true_sec, dtype=float)
    y_pred_sec = np.asarray(y_pred_sec, dtype=float)

    y_true_min = y_true_sec / 60.0
    y_pred_min = y_pred_sec / 60.0

    abs_err_min = np.abs(y_true_min - y_pred_min)

    metrics = {
        "mae_minutes": float(mean_absolute_error(y_true_min, y_pred_min)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(y_true_min, y_pred_min))),
        "median_absolute_error_minutes": float(median_absolute_error(y_true_min, y_pred_min)),
        "within_2_minutes_accuracy": float(np.mean(abs_err_min <= 2.0)),
        "within_5_minutes_accuracy": float(np.mean(abs_err_min <= 5.0)),
        "mae_seconds": float(mean_absolute_error(y_true_sec, y_pred_sec)),
    }
    return metrics


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residual = y_true - y_pred
    return float(np.mean(np.maximum(alpha * residual, (alpha - 1.0) * residual)))


def quantile_interval_metrics_seconds(
    y_true_sec: np.ndarray,
    y_pred_p50_sec: np.ndarray,
    y_pred_lower_sec: np.ndarray,
    y_pred_upper_sec: np.ndarray,
    lower_alpha: float,
    upper_alpha: float,
) -> Dict[str, float]:
    base_metrics = regression_metrics_seconds(y_true_sec, y_pred_p50_sec)

    lower = np.asarray(y_pred_lower_sec, dtype=float)
    upper = np.asarray(y_pred_upper_sec, dtype=float)
    y_true = np.asarray(y_true_sec, dtype=float)
    width_minutes = (upper - lower) / 60.0
    in_interval = (y_true >= lower) & (y_true <= upper)

    metrics = dict(base_metrics)
    metrics.update(
        {
            "lower_quantile": float(lower_alpha),
            "upper_quantile": float(upper_alpha),
            "average_interval_width_minutes": float(np.mean(width_minutes)),
            "median_interval_width_minutes": float(np.median(width_minutes)),
            "interval_coverage": float(np.mean(in_interval)),
            "undercoverage_rate": float(np.mean(y_true < lower)),
            "overcoverage_rate": float(np.mean(y_true > upper)),
        }
    )
    return metrics


def benchmarking_table(metrics_by_model: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    table = pd.DataFrame(metrics_by_model).T.sort_values("mae_minutes")
    table.index.name = "model"
    return table


def error_breakdown_table(
    eval_df: pd.DataFrame,
    group_col: str,
    y_true_col: str = "y_true_seconds",
    y_pred_col: str = "y_pred_seconds",
) -> pd.DataFrame:
    rows = []
    for group_value, group_df in eval_df.groupby(group_col):
        metrics = regression_metrics_seconds(group_df[y_true_col].to_numpy(), group_df[y_pred_col].to_numpy())
        rows.append(
            {
                group_col: group_value,
                "count": len(group_df),
                "mae_minutes": metrics["mae_minutes"],
                "rmse_minutes": metrics["rmse_minutes"],
            }
        )
    return pd.DataFrame(rows).sort_values("mae_minutes")


def add_duration_bucket(eval_df: pd.DataFrame, y_true_col: str = "y_true_seconds") -> pd.DataFrame:
    out = eval_df.copy()
    true_minutes = out[y_true_col] / 60.0
    bins = [0, 25, 30, 35, 40, 100]
    labels = ["<25", "25-30", "30-35", "35-40", "40+"]
    out["duration_bucket"] = pd.cut(true_minutes, bins=bins, labels=labels, include_lowest=True, right=False)
    return out


def save_residual_plots(eval_df: pd.DataFrame, out_dir: str | Path, model_name: str) -> None:
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    residuals_min = (eval_df["y_pred_seconds"] - eval_df["y_true_seconds"]) / 60.0
    y_pred_min = eval_df["y_pred_seconds"] / 60.0

    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred_min, residuals_min, alpha=0.35)
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Predicted duration (minutes)")
    plt.ylabel("Residual (pred - true) minutes")
    plt.title(f"Residual Scatter: {model_name}")
    plt.tight_layout()
    plt.savefig(target / f"{model_name}_residual_scatter.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.hist(residuals_min, bins=40, alpha=0.85)
    plt.xlabel("Residual (minutes)")
    plt.ylabel("Count")
    plt.title(f"Residual Distribution: {model_name}")
    plt.tight_layout()
    plt.savefig(target / f"{model_name}_residual_hist.png", dpi=140)
    plt.close()


def save_error_bar_plot(
    breakdown_df: pd.DataFrame,
    group_col: str,
    out_dir: str | Path,
    model_name: str,
) -> None:
    if breakdown_df.empty:
        LOGGER.warning("Skipping error bar plot for %s: empty dataframe.", group_col)
        return

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)

    plot_df = breakdown_df.sort_values("mae_minutes", ascending=False).head(20)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df[group_col].astype(str), plot_df["mae_minutes"])
    plt.xlabel("MAE (minutes)")
    plt.ylabel(group_col)
    plt.title(f"MAE by {group_col}: {model_name}")
    plt.tight_layout()
    plt.savefig(target / f"{model_name}_mae_by_{group_col}.png", dpi=140)
    plt.close()
