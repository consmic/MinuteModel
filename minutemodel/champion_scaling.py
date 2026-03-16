from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

ROLE_ORDER = ["top", "jng", "mid", "bot", "sup"]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _sanitize_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_clean_text(v) for v in value if _clean_text(v)]
    text = _clean_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].replace("'", "").replace('"', "")
        return [token.strip() for token in inner.split(",") if token.strip()]
    return [text]


@dataclass
class ChampionScalingLookup:
    """Fit champion scaling priors on train-only data and apply to future rows.

    Leakage safety: this class must be fit only on the training split, then reused
    to transform validation/test/inference rows without refitting.
    """

    method: str = "avg_duration_delta"
    smoothing: bool = True
    min_samples: int = 20
    recency_weighting: bool = False
    recency_half_life_days: int = 180
    patch_aware: bool = True

    fitted_: bool = False
    global_mean_seconds_: float = 0.0
    champion_coeff_map_: Dict[str, float] = field(default_factory=dict)
    champion_sample_map_: Dict[str, int] = field(default_factory=dict)
    patch_coeff_map_: Dict[Tuple[str, str], float] = field(default_factory=dict)
    patch_sample_map_: Dict[Tuple[str, str], int] = field(default_factory=dict)
    coeff_table_: pd.DataFrame = field(default_factory=pd.DataFrame)

    def fit(self, train_match_df: pd.DataFrame) -> "ChampionScalingLookup":
        records = self._build_training_records(train_match_df)
        if records.empty:
            self.fitted_ = True
            self.global_mean_seconds_ = float(pd.to_numeric(train_match_df.get("target_gamelength_seconds"), errors="coerce").mean())
            self.champion_coeff_map_ = {}
            self.champion_sample_map_ = {}
            self.patch_coeff_map_ = {}
            self.patch_sample_map_ = {}
            self.coeff_table_ = pd.DataFrame(columns=["champion", "sample_size", "scaling_coeff", "smoothed_scaling_coeff"])
            return self

        records["target_seconds"] = pd.to_numeric(records["target_seconds"], errors="coerce")
        records = records.dropna(subset=["target_seconds", "champion"]).copy()
        if records.empty:
            raise ValueError("Champion scaling fit failed: no valid champion-duration rows after cleaning.")

        records["weight"] = self._compute_weights(records)
        self.global_mean_seconds_ = self._weighted_mean(records["target_seconds"], records["weight"])

        champion_stats = self._champion_stats(records)
        self.champion_coeff_map_ = {
            row["champion"]: float(row["smoothed_scaling_coeff"] if self.smoothing else row["scaling_coeff"])
            for _, row in champion_stats.iterrows()
        }
        self.champion_sample_map_ = {
            row["champion"]: int(row["sample_size"])
            for _, row in champion_stats.iterrows()
        }

        if self.patch_aware:
            patch_stats = self._patch_stats(records, champion_stats)
            self.patch_coeff_map_ = {
                (row["champion"], row["patch"]): float(row["smoothed_scaling_coeff"] if self.smoothing else row["scaling_coeff"])
                for _, row in patch_stats.iterrows()
            }
            self.patch_sample_map_ = {
                (row["champion"], row["patch"]): int(row["sample_size"])
                for _, row in patch_stats.iterrows()
            }
            champion_stats = champion_stats.merge(
                patch_stats.groupby("champion")["patch"].nunique().rename("patch_count"),
                on="champion",
                how="left",
            )
            champion_stats["patch_count"] = champion_stats["patch_count"].fillna(0).astype(int)
        else:
            self.patch_coeff_map_ = {}
            self.patch_sample_map_ = {}
            champion_stats["patch_count"] = 0

        self.coeff_table_ = champion_stats.sort_values("sample_size", ascending=False).reset_index(drop=True)
        self.fitted_ = True
        return self

    def transform(self, match_df: pd.DataFrame, prefer_role_specific: bool = True) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("ChampionScalingLookup must be fitted before transform().")

        rows: List[Dict[str, float]] = []
        for _, row in match_df.iterrows():
            patch = _clean_text(row.get("patch"))
            blue_champs = self._extract_side_champions(row, side="blue", prefer_role_specific=prefer_role_specific)
            red_champs = self._extract_side_champions(row, side="red", prefer_role_specific=prefer_role_specific)

            blue_coeffs = [self._champion_coeff(champ, patch) for champ in blue_champs]
            red_coeffs = [self._champion_coeff(champ, patch) for champ in red_champs]

            blue_agg = self._aggregate_coeffs(blue_coeffs)
            red_agg = self._aggregate_coeffs(red_coeffs)

            rows.append(
                {
                    "blue_scaling_sum": blue_agg["sum"],
                    "red_scaling_sum": red_agg["sum"],
                    "blue_scaling_mean": blue_agg["mean"],
                    "red_scaling_mean": red_agg["mean"],
                    "blue_scaling_max": blue_agg["max"],
                    "red_scaling_max": red_agg["max"],
                    "blue_scaling_min": blue_agg["min"],
                    "red_scaling_min": red_agg["min"],
                    "blue_scaling_known_count": float(blue_agg["known_count"]),
                    "red_scaling_known_count": float(red_agg["known_count"]),
                    "scaling_diff": blue_agg["sum"] - red_agg["sum"],
                }
            )

        return pd.DataFrame(rows, index=match_df.index)

    def to_lookup_table(self) -> pd.DataFrame:
        if self.coeff_table_.empty:
            return pd.DataFrame(columns=["champion", "sample_size", "scaling_coeff", "smoothed_scaling_coeff", "patch_count"])
        return self.coeff_table_.copy()

    def save(self, path: str | Path) -> None:
        """Persist fitted lookup artifact for reuse in external inference services."""
        artifact = {
            "params": {
                "method": self.method,
                "smoothing": self.smoothing,
                "min_samples": self.min_samples,
                "recency_weighting": self.recency_weighting,
                "recency_half_life_days": self.recency_half_life_days,
                "patch_aware": self.patch_aware,
            },
            "state": {
                "fitted_": self.fitted_,
                "global_mean_seconds_": self.global_mean_seconds_,
                "champion_coeff_map_": self.champion_coeff_map_,
                "champion_sample_map_": self.champion_sample_map_,
                "patch_coeff_map_": self.patch_coeff_map_,
                "patch_sample_map_": self.patch_sample_map_,
                "coeff_table_": self.coeff_table_,
            },
        }
        joblib.dump(artifact, Path(path))

    @classmethod
    def load(cls, path: str | Path) -> "ChampionScalingLookup":
        artifact = joblib.load(Path(path))
        params = artifact.get("params", {})
        state = artifact.get("state", {})

        obj = cls(**params)
        obj.fitted_ = bool(state.get("fitted_", False))
        obj.global_mean_seconds_ = float(state.get("global_mean_seconds_", 0.0))
        obj.champion_coeff_map_ = dict(state.get("champion_coeff_map_", {}))
        obj.champion_sample_map_ = dict(state.get("champion_sample_map_", {}))
        obj.patch_coeff_map_ = dict(state.get("patch_coeff_map_", {}))
        obj.patch_sample_map_ = dict(state.get("patch_sample_map_", {}))
        coeff_table = state.get("coeff_table_", pd.DataFrame())
        obj.coeff_table_ = coeff_table if isinstance(coeff_table, pd.DataFrame) else pd.DataFrame(coeff_table)
        return obj

    def _build_training_records(self, match_df: pd.DataFrame) -> pd.DataFrame:
        records: List[Dict[str, Any]] = []
        for _, row in match_df.iterrows():
            target = pd.to_numeric(row.get("target_gamelength_seconds"), errors="coerce")
            if pd.isna(target):
                continue
            patch = _clean_text(row.get("patch"))
            date = pd.to_datetime(row.get("date"), errors="coerce", utc=False)

            for side in ["blue", "red"]:
                champs = self._extract_side_champions(row, side=side, prefer_role_specific=True)
                for champ in champs:
                    records.append(
                        {
                            "champion": champ,
                            "patch": patch,
                            "date": date,
                            "target_seconds": float(target),
                        }
                    )

        return pd.DataFrame(records)

    def _extract_side_champions(self, row: pd.Series, side: str, prefer_role_specific: bool = True) -> List[str]:
        side = side.lower()

        champions: List[str] = []
        if prefer_role_specific:
            role_values = [_clean_text(row.get(f"{side}_{role}_champion")) for role in ROLE_ORDER]
            champions = [champ for champ in role_values if champ]

        if len(champions) < 5:
            picks = [_clean_text(row.get(f"{side}_pick{idx}")) for idx in range(1, 6)]
            champions = champions + [pick for pick in picks if pick and pick not in champions]

        if len(champions) < 5:
            list_values = _sanitize_list(row.get(f"{side}_draft_champions"))
            champions = champions + [champ for champ in list_values if champ and champ not in champions]

        return champions[:5]

    def _compute_weights(self, records: pd.DataFrame) -> pd.Series:
        if not self.recency_weighting or "date" not in records.columns or records["date"].isna().all():
            return pd.Series(np.ones(len(records), dtype=float), index=records.index)

        max_date = records["date"].max()
        age_days = (max_date - records["date"]).dt.total_seconds() / 86400.0
        age_days = age_days.fillna(0.0).clip(lower=0.0)

        half_life = float(max(self.recency_half_life_days, 1))
        weights = np.power(0.5, age_days / half_life)
        return pd.Series(weights.astype(float), index=records.index)

    @staticmethod
    def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
        value_arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        weight_arr = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)

        valid = np.isfinite(value_arr) & np.isfinite(weight_arr) & (weight_arr > 0)
        if not valid.any():
            return float(np.nanmean(value_arr))
        return float(np.sum(value_arr[valid] * weight_arr[valid]) / np.sum(weight_arr[valid]))

    def _champion_stats(self, records: pd.DataFrame) -> pd.DataFrame:
        grouped = records.groupby("champion", dropna=False)
        rows: List[Dict[str, Any]] = []
        for champion, group in grouped:
            sample_size = int(len(group))
            champion_mean = self._weighted_mean(group["target_seconds"], group["weight"])
            scaling_coeff = champion_mean - self.global_mean_seconds_
            shrink = sample_size / float(sample_size + max(self.min_samples, 1))
            smoothed_scaling_coeff = scaling_coeff * shrink if self.smoothing else scaling_coeff
            rows.append(
                {
                    "champion": _clean_text(champion),
                    "sample_size": sample_size,
                    "champion_avg_seconds": champion_mean,
                    "scaling_coeff": scaling_coeff,
                    "smoothed_scaling_coeff": smoothed_scaling_coeff,
                }
            )
        out = pd.DataFrame(rows)
        out = out[out["champion"].astype(str).str.len() > 0].copy()
        return out

    def _patch_stats(self, records: pd.DataFrame, champion_stats: pd.DataFrame) -> pd.DataFrame:
        champion_delta_map = {
            row["champion"]: float(row["smoothed_scaling_coeff"] if self.smoothing else row["scaling_coeff"])
            for _, row in champion_stats.iterrows()
        }

        grouped = records.groupby(["champion", "patch"], dropna=False)
        rows: List[Dict[str, Any]] = []
        for (champion, patch), group in grouped:
            champion_clean = _clean_text(champion)
            patch_clean = _clean_text(patch)
            if not champion_clean:
                continue
            sample_size = int(len(group))
            patch_mean = self._weighted_mean(group["target_seconds"], group["weight"])
            scaling_coeff = patch_mean - self.global_mean_seconds_

            champion_prior = champion_delta_map.get(champion_clean, 0.0)
            shrink = sample_size / float(sample_size + max(self.min_samples, 1))
            smoothed_scaling_coeff = (scaling_coeff * shrink) + (champion_prior * (1.0 - shrink)) if self.smoothing else scaling_coeff
            rows.append(
                {
                    "champion": champion_clean,
                    "patch": patch_clean,
                    "sample_size": sample_size,
                    "scaling_coeff": scaling_coeff,
                    "smoothed_scaling_coeff": smoothed_scaling_coeff,
                }
            )

        return pd.DataFrame(rows)

    def _champion_coeff(self, champion: str, patch: str) -> float:
        champion_clean = _clean_text(champion)
        patch_clean = _clean_text(patch)
        if not champion_clean:
            return 0.0

        if self.patch_aware:
            key = (champion_clean, patch_clean)
            if key in self.patch_coeff_map_:
                return float(self.patch_coeff_map_[key])

        return float(self.champion_coeff_map_.get(champion_clean, 0.0))

    @staticmethod
    def _aggregate_coeffs(coeffs: List[float]) -> Dict[str, float]:
        if not coeffs:
            coeffs = [0.0]
        arr = np.array(coeffs, dtype=float)
        known_count = int(np.sum(np.isfinite(arr) & (arr != 0.0)))
        return {
            "sum": float(np.nansum(arr)),
            "mean": float(np.nanmean(arr)),
            "max": float(np.nanmax(arr)),
            "min": float(np.nanmin(arr)),
            "known_count": float(known_count),
        }
