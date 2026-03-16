from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from .champion_scaling import ChampionScalingLookup
from .config import PipelineConfig, is_forbidden_feature_column

LOGGER = logging.getLogger(__name__)


def _to_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _safe_group_key(team_id: Any, team_name: Any, side: str) -> str:
    if pd.notna(team_id):
        return f"id::{team_id}"
    if pd.notna(team_name):
        return f"name::{team_name}"
    return f"unknown::{side}"


def _shifted_team_rolling_mean(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
) -> pd.Series:
    return (
        df.groupby(group_col, group_keys=False)[value_col]
        .apply(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
        .reset_index(drop=True)
    )


def _shifted_group_expanding_mean(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    return (
        df.groupby(group_col, group_keys=False)[value_col]
        .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
        .reset_index(drop=True)
    )


def build_leakage_safe_rolling_priors(match_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Create rolling team priors using only historical matches."""
    rows = []
    for _, row in match_df.iterrows():
        rows.append(
            {
                "gameid": row["gameid"],
                "date": row["date"],
                "league": row.get("league"),
                "side": "Blue",
                "team_id": row.get("blue_team_id"),
                "team_name": row.get("blue_team_name"),
                "target_gamelength_seconds": row.get("target_gamelength_seconds"),
                "ckpm_source": row.get("blue_ckpm_source"),
            }
        )
        rows.append(
            {
                "gameid": row["gameid"],
                "date": row["date"],
                "league": row.get("league"),
                "side": "Red",
                "team_id": row.get("red_team_id"),
                "team_name": row.get("red_team_name"),
                "target_gamelength_seconds": row.get("target_gamelength_seconds"),
                "ckpm_source": row.get("red_ckpm_source"),
            }
        )

    team_history = pd.DataFrame(rows)
    team_history["date"] = pd.to_datetime(team_history["date"], errors="coerce", utc=False)
    team_history = team_history.sort_values(["date", "gameid", "side"]).reset_index(drop=True)

    team_history["team_key"] = team_history.apply(
        lambda r: _safe_group_key(r["team_id"], r["team_name"], r["side"]),
        axis=1,
    )

    team_history["target_gamelength_seconds"] = _to_float_series(team_history["target_gamelength_seconds"])
    team_history["ckpm_source"] = _to_float_series(team_history["ckpm_source"])

    window = int(config.rolling_window_size)
    duration_col = f"rolling_duration_{window}"
    ckpm_col = f"rolling_ckpm_{window}"

    team_history[duration_col] = _shifted_team_rolling_mean(
        team_history,
        group_col="team_key",
        value_col="target_gamelength_seconds",
        window=window,
    )

    team_history["league_duration_prior"] = _shifted_group_expanding_mean(
        team_history,
        group_col="league",
        value_col="target_gamelength_seconds",
    )
    team_history["global_duration_prior"] = team_history["target_gamelength_seconds"].shift(1).expanding(min_periods=1).mean()

    team_history[duration_col] = (
        team_history[duration_col]
        .fillna(team_history["league_duration_prior"])
        .fillna(team_history["global_duration_prior"])
        .fillna(team_history["target_gamelength_seconds"].mean())
    )

    if config.use_rolling_ckpm_prior and team_history["ckpm_source"].notna().any():
        team_history[ckpm_col] = _shifted_team_rolling_mean(
            team_history,
            group_col="team_key",
            value_col="ckpm_source",
            window=window,
        )
        team_history["league_ckpm_prior"] = _shifted_group_expanding_mean(
            team_history,
            group_col="league",
            value_col="ckpm_source",
        )
        team_history["global_ckpm_prior"] = team_history["ckpm_source"].shift(1).expanding(min_periods=1).mean()
        team_history[ckpm_col] = (
            team_history[ckpm_col]
            .fillna(team_history["league_ckpm_prior"])
            .fillna(team_history["global_ckpm_prior"])
            .fillna(team_history["ckpm_source"].mean())
        )
    else:
        team_history[ckpm_col] = np.nan

    blue = team_history[team_history["side"] == "Blue"][["gameid", duration_col, ckpm_col]].rename(
        columns={duration_col: "blue_rolling_duration_prior_seconds", ckpm_col: "blue_rolling_ckpm_prior"}
    )
    red = team_history[team_history["side"] == "Red"][["gameid", duration_col, ckpm_col]].rename(
        columns={duration_col: "red_rolling_duration_prior_seconds", ckpm_col: "red_rolling_ckpm_prior"}
    )

    merged = match_df.merge(blue, on="gameid", how="left").merge(red, on="gameid", how="left")
    merged["rolling_duration_prior_diff_seconds"] = (
        merged["blue_rolling_duration_prior_seconds"] - merged["red_rolling_duration_prior_seconds"]
    )
    merged["rolling_ckpm_prior_diff"] = merged["blue_rolling_ckpm_prior"] - merged["red_rolling_ckpm_prior"]

    return merged


def _sanitize_draft_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v) for v in value if pd.notna(v) and str(v).strip()]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        cleaned = text[1:-1].replace("'", "").replace('"', "")
        tokens = [tok.strip() for tok in cleaned.split(",") if tok.strip()]
        return tokens
    return [text]


def assert_no_forbidden_features(columns: List[str]) -> None:
    forbidden = [col for col in columns if is_forbidden_feature_column(col)]
    if forbidden:
        raise ValueError(
            "Forbidden in-game/post-game columns detected in model features: "
            f"{sorted(set(forbidden))}"
        )


@dataclass
class DraftFeatureBuilder:
    config: PipelineConfig
    blue_mlb: MultiLabelBinarizer = field(default_factory=MultiLabelBinarizer)
    red_mlb: MultiLabelBinarizer = field(default_factory=MultiLabelBinarizer)
    champion_scaling_lookup_: Optional[ChampionScalingLookup] = None
    fitted: bool = False

    categorical_columns_: List[str] = field(default_factory=list)
    numeric_columns_: List[str] = field(default_factory=list)
    blue_classes_set_: set[str] = field(default_factory=set)
    red_classes_set_: set[str] = field(default_factory=set)

    def fit(self, match_df: pd.DataFrame) -> "DraftFeatureBuilder":
        if self.config.use_champion_scaling_features:
            self.champion_scaling_lookup_ = ChampionScalingLookup(
                method=self.config.champion_scaling_method,
                smoothing=self.config.champion_scaling_smoothing,
                min_samples=self.config.champion_scaling_min_samples,
                recency_weighting=self.config.champion_scaling_recency_weighting,
                recency_half_life_days=self.config.champion_scaling_recency_half_life_days,
                patch_aware=self.config.champion_scaling_patch_aware,
            ).fit(match_df)
        else:
            self.champion_scaling_lookup_ = None

        if self.config.use_bag_of_champions_fallback:
            blue_lists = match_df["blue_draft_champions"].apply(_sanitize_draft_list)
            red_lists = match_df["red_draft_champions"].apply(_sanitize_draft_list)
            self.blue_mlb.fit(blue_lists)
            self.red_mlb.fit(red_lists)
            self.blue_classes_set_ = set(self.blue_mlb.classes_.tolist())
            self.red_classes_set_ = set(self.red_mlb.classes_.tolist())
        self.fitted = True
        return self

    def transform(self, match_df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("DraftFeatureBuilder must be fitted before calling transform().")

        features = pd.DataFrame(index=match_df.index)

        # Core known-at-draft metadata.
        base_categorical = [
            "league",
            "split",
            "patch",
            "blue_team_id",
            "red_team_id",
            "blue_team_name",
            "red_team_name",
        ]
        for col in base_categorical:
            features[col] = match_df.get(col, np.nan).astype("object")

        features["playoffs"] = match_df.get("playoffs", 0).astype("object")

        features["blue_first_pick"] = pd.to_numeric(match_df.get("blue_first_pick"), errors="coerce").fillna(0.0)
        features["red_first_pick"] = pd.to_numeric(match_df.get("red_first_pick"), errors="coerce").fillna(0.0)

        features["year"] = pd.to_numeric(match_df.get("year"), errors="coerce").fillna(0.0)

        numeric_prior_cols = [
            "blue_rolling_duration_prior_seconds",
            "red_rolling_duration_prior_seconds",
            "rolling_duration_prior_diff_seconds",
            "blue_rolling_ckpm_prior",
            "red_rolling_ckpm_prior",
            "rolling_ckpm_prior_diff",
        ]
        for col in numeric_prior_cols:
            features[col] = pd.to_numeric(match_df.get(col), errors="coerce")

        if self.config.use_role_specific_draft_features:
            role_cols = [
                "blue_top_champion",
                "blue_jng_champion",
                "blue_mid_champion",
                "blue_bot_champion",
                "blue_sup_champion",
                "red_top_champion",
                "red_jng_champion",
                "red_mid_champion",
                "red_bot_champion",
                "red_sup_champion",
            ]
            for col in role_cols:
                features[col] = match_df.get(col, np.nan).astype("object")

        pick_cols = [
            "blue_pick1",
            "blue_pick2",
            "blue_pick3",
            "blue_pick4",
            "blue_pick5",
            "red_pick1",
            "red_pick2",
            "red_pick3",
            "red_pick4",
            "red_pick5",
        ]
        ban_cols = [
            "blue_ban1",
            "blue_ban2",
            "blue_ban3",
            "blue_ban4",
            "blue_ban5",
            "red_ban1",
            "red_ban2",
            "red_ban3",
            "red_ban4",
            "red_ban5",
        ]
        for col in pick_cols + ban_cols:
            features[col] = match_df.get(col, np.nan).astype("object")

        if self.config.use_bag_of_champions_fallback:
            blue_lists_raw = match_df["blue_draft_champions"].apply(_sanitize_draft_list)
            red_lists_raw = match_df["red_draft_champions"].apply(_sanitize_draft_list)

            blue_unknown = blue_lists_raw.apply(lambda champs: float(any(ch not in self.blue_classes_set_ for ch in champs)))
            red_unknown = red_lists_raw.apply(lambda champs: float(any(ch not in self.red_classes_set_ for ch in champs)))

            blue_lists = blue_lists_raw.apply(lambda champs: [ch for ch in champs if ch in self.blue_classes_set_])
            red_lists = red_lists_raw.apply(lambda champs: [ch for ch in champs if ch in self.red_classes_set_])

            blue_matrix = self.blue_mlb.transform(blue_lists.tolist())
            red_matrix = self.red_mlb.transform(red_lists.tolist())

            blue_cols = [f"blue_has_{c}" for c in self.blue_mlb.classes_]
            red_cols = [f"red_has_{c}" for c in self.red_mlb.classes_]

            blue_df = pd.DataFrame(blue_matrix, index=match_df.index, columns=blue_cols)
            red_df = pd.DataFrame(red_matrix, index=match_df.index, columns=red_cols)
            features = pd.concat([features, blue_df, red_df], axis=1)
            features["blue_has_unknown_champion"] = blue_unknown.to_numpy(dtype=float)
            features["red_has_unknown_champion"] = red_unknown.to_numpy(dtype=float)

        if self.config.use_champion_scaling_features:
            if self.champion_scaling_lookup_ is None:
                raise RuntimeError(
                    "Champion scaling features were requested but ChampionScalingLookup is missing. "
                    "Call fit() on DraftFeatureBuilder before transform()."
                )
            scaling_df = self.champion_scaling_lookup_.transform(
                match_df,
                prefer_role_specific=self.config.use_role_specific_draft_features,
            )
            for col in scaling_df.columns:
                features[col] = pd.to_numeric(scaling_df[col], errors="coerce")

        # Columns by dtype for downstream preprocessing.
        categorical_columns = [
            col
            for col in features.columns
            if str(features[col].dtype) == "object"
        ]
        numeric_columns = [col for col in features.columns if col not in categorical_columns]

        self.categorical_columns_ = categorical_columns
        self.numeric_columns_ = numeric_columns

        assert_no_forbidden_features(list(features.columns))
        return features

    def fit_transform(self, match_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(match_df).transform(match_df)

    def get_feature_columns(self) -> Tuple[List[str], List[str]]:
        return self.categorical_columns_, self.numeric_columns_

    def get_champion_scaling_lookup_table(self) -> pd.DataFrame:
        if self.champion_scaling_lookup_ is None:
            return pd.DataFrame(
                columns=[
                    "champion",
                    "sample_size",
                    "champion_avg_seconds",
                    "scaling_coeff",
                    "smoothed_scaling_coeff",
                    "patch_count",
                ]
            )
        return self.champion_scaling_lookup_.to_lookup_table()


def build_target(match_df: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    if config.target_unit == "minutes":
        target = pd.to_numeric(match_df["target_gamelength_minutes"], errors="coerce")
    else:
        target = pd.to_numeric(match_df["target_gamelength_seconds"], errors="coerce")
    return target.astype(float)
