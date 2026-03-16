from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class BaselineResult:
    name: str
    predictions: np.ndarray


class GlobalMeanBaseline:
    def __init__(self) -> None:
        self.mean_: float = 0.0

    def fit(self, y: pd.Series) -> "GlobalMeanBaseline":
        self.mean_ = float(pd.to_numeric(y, errors="coerce").mean())
        return self

    def predict(self, n_samples: int) -> np.ndarray:
        return np.full(shape=n_samples, fill_value=self.mean_, dtype=float)


class GroupMeanBaseline:
    def __init__(self, group_cols: Iterable[str]) -> None:
        self.group_cols = list(group_cols)
        self.global_mean_: float = 0.0
        self.group_means_: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GroupMeanBaseline":
        df = X[self.group_cols].copy()
        df["target"] = pd.to_numeric(y, errors="coerce")

        self.global_mean_ = float(df["target"].mean())
        self.group_means_ = df.groupby(self.group_cols)["target"].mean()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.group_means_ is None:
            raise RuntimeError("GroupMeanBaseline not fitted.")

        merged = X[self.group_cols].merge(
            self.group_means_.rename("pred").reset_index(),
            on=self.group_cols,
            how="left",
        )
        preds = merged["pred"].fillna(self.global_mean_)
        return preds.to_numpy(dtype=float)


class RidgeDraftBaseline:
    def __init__(self, alpha: float = 3.0) -> None:
        self.alpha = alpha
        self.pipeline: Optional[Pipeline] = None

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_cols: List[str],
        numeric_cols: List[str],
    ) -> "RidgeDraftBaseline":
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", categorical_pipe, categorical_cols),
                ("numeric", numeric_pipe, numeric_cols),
            ],
            sparse_threshold=0.3,
        )

        self.pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("regressor", Ridge(alpha=self.alpha, random_state=42)),
            ]
        )

        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("RidgeDraftBaseline not fitted.")
        return self.pipeline.predict(X)


def train_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_cols: List[str],
    numeric_cols: List[str],
) -> Dict[str, BaselineResult]:
    baselines: Dict[str, BaselineResult] = {}

    global_model = GlobalMeanBaseline().fit(y_train)
    baselines["global_mean"] = BaselineResult(
        name="global_mean",
        predictions=global_model.predict(len(X_val)),
    )

    league_model = GroupMeanBaseline(["league"]).fit(X_train, y_train)
    baselines["league_mean"] = BaselineResult(
        name="league_mean",
        predictions=league_model.predict(X_val),
    )

    league_patch_model = GroupMeanBaseline(["league", "patch"]).fit(X_train, y_train)
    baselines["league_patch_mean"] = BaselineResult(
        name="league_patch_mean",
        predictions=league_patch_model.predict(X_val),
    )

    # Small alpha sweep on validation for a stable Ridge baseline.
    candidate_alphas = [1.0, 3.0, 10.0]
    best_alpha = candidate_alphas[0]
    best_mae = float("inf")
    best_model: Optional[RidgeDraftBaseline] = None

    for alpha in candidate_alphas:
        model = RidgeDraftBaseline(alpha=alpha).fit(X_train, y_train, categorical_cols, numeric_cols)
        preds = model.predict(X_val)
        mae = float(np.mean(np.abs(preds - y_val.to_numpy(dtype=float))))
        if mae < best_mae:
            best_mae = mae
            best_alpha = alpha
            best_model = model

    if best_model is None:
        raise RuntimeError("Failed to fit Ridge baseline.")

    baselines["ridge_regression"] = BaselineResult(
        name=f"ridge_regression_alpha_{best_alpha}",
        predictions=best_model.predict(X_val),
    )

    return baselines


def fit_group_baselines_for_inference(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Tuple[GlobalMeanBaseline, GroupMeanBaseline, GroupMeanBaseline]:
    global_model = GlobalMeanBaseline().fit(y_train)
    league_model = GroupMeanBaseline(["league"]).fit(X_train, y_train)
    league_patch_model = GroupMeanBaseline(["league", "patch"]).fit(X_train, y_train)
    return global_model, league_model, league_patch_model