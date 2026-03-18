from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


ALLOWED_METADATA_COLUMNS: List[str] = [
    "gameid",
    "league",
    "year",
    "split",
    "playoffs",
    "date",
    "game",
    "patch",
    "side",
    "teamname",
    "teamid",
    "firstPick",
]

ALLOWED_DRAFT_COLUMNS: List[str] = [
    "position",
    "champion",
    "ban1",
    "ban2",
    "ban3",
    "ban4",
    "ban5",
    "pick1",
    "pick2",
    "pick3",
    "pick4",
    "pick5",
]

TARGET_COLUMNS: List[str] = ["gamelength"]

# Post-game columns allowed as *historical source signals* for rolling priors only.
# These must never be used as direct current-match features.
HISTORICAL_PRIOR_SOURCE_COLUMNS: List[str] = [
    "result",
    "firstblood",
    "firstdragon",
    "dragons",
    "firstherald",
    "heralds",
    "firstbaron",
    "barons",
    "golddiffat15",
    "firsttower",
    "firstmidtower",
    "firsttothreetowers",
    "towers",
    "opp_towers",
]

IDENTIFIER_COLUMNS: List[str] = [
    "participantid",
    "datacompleteness",
    "url",
    "playername",
    "playerid",
]

# Explicit post-game / in-game fields that cannot enter V1 training features.
FORBIDDEN_COLUMNS: List[str] = [
    "result",
    "kills",
    "deaths",
    "assists",
    "teamkills",
    "teamdeaths",
    "firstblood",
    "firstbloodkill",
    "firstbloodassist",
    "firstbloodvictim",
    "kpm",
    "ckpm",
    "dragons",
    "opp_dragons",
    "elementaldrakes",
    "opp_elementaldrakes",
    "infernals",
    "mountains",
    "clouds",
    "oceans",
    "chemtechs",
    "hextechs",
    "dragonsoul",
    "opp_dragonsoul",
    "barons",
    "opp_barons",
    "heralds",
    "opp_heralds",
    "void_grubs",
    "opp_void_grubs",
    "atakhans",
    "opp_atakhans",
    "towers",
    "opp_towers",
    "inhibitors",
    "opp_inhibitors",
    "damagetochampions",
    "dpm",
    "damagetakenperminute",
    "visionscore",
    "vspm",
    "totalgold",
    "earnedgold",
    "earned gpm",
    "goldspent",
    "gspd",
    "minionkills",
    "monsterkills",
    "cspm",
    "xpdiff",
    "golddiff",
    "csdiff",
]

FORBIDDEN_COLUMN_PATTERNS: List[str] = [
    "@10",
    "@15",
    "@20",
    "@25",
    "at10",
    "at15",
    "at20",
    "at25",
    "golddiffat",
    "xpdiffat",
    "csdiffat",
    "killsat",
    "deathsat",
    "assistsat",
]

ALLOWED_COLUMNS: List[str] = sorted(
    set(ALLOWED_METADATA_COLUMNS + ALLOWED_DRAFT_COLUMNS + TARGET_COLUMNS + IDENTIFIER_COLUMNS + ["ckpm"])
)


@dataclass
class PipelineConfig:
    input_csv: str
    output_dir: str = "artifacts"
    reports_dir: str = "reports"

    primary_model: str = "catboost"
    use_role_specific_draft_features: bool = True
    use_bag_of_champions_fallback: bool = True
    use_sparse_champion_indicator_features: bool = True
    use_pick_order_champion_features: bool = True
    use_series_game_number_feature: bool = False
    target_unit: str = "seconds"
    rolling_window_size: int = 10
    use_champion_scaling_features: bool = False
    use_extended_rolling_team_priors: bool = True
    use_draft_summary_features: bool = True
    use_draft_interaction_features: bool = True
    use_draft_conditional_behaviour_features: bool = True
    use_turret_prior_features: bool = False
    use_extended_turret_prior_features: bool = False
    draft_conditional_min_samples: int = 5
    run_feature_group_ablation: bool = True
    run_refinement_ablation: bool = False
    run_turret_feature_ablation: bool = False
    champion_scaling_method: str = "avg_duration_delta"
    champion_scaling_smoothing: bool = True
    champion_scaling_min_samples: int = 20
    champion_scaling_recency_weighting: bool = False
    champion_scaling_recency_half_life_days: int = 180
    champion_scaling_patch_aware: bool = True
    run_champion_scaling_ablation: bool = True
    use_rolling_ckpm_prior: bool = True
    enable_quantile_regression: bool = False
    quantile_levels: List[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    quantile_enforce_non_crossing: bool = True
    volatility_threshold_quantiles: List[float] = field(default_factory=lambda: [0.33, 0.67])

    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    random_seed: int = 42

    catboost_iterations: int = 1600
    catboost_early_stopping_rounds: int = 120
    catboost_param_grid: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "depth": 5,
                "learning_rate": 0.03,
                "l2_leaf_reg": 9.0,
                "bagging_temperature": 0.5,
                "random_strength": 1.0,
            },
            {
                "depth": 6,
                "learning_rate": 0.05,
                "l2_leaf_reg": 3.0,
                "bagging_temperature": 0.0,
                "random_strength": 0.0,
            },
        ]
    )

    lgbm_param_grid: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "n_estimators": 400,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_child_samples": 20,
                "subsample": 0.9,
                "colsample_bytree": 0.9,
            },
            {
                "n_estimators": 700,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "min_child_samples": 30,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
            },
            {
                "n_estimators": 900,
                "learning_rate": 0.02,
                "num_leaves": 95,
                "min_child_samples": 40,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        ]
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        return cls(**payload)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "PipelineConfig":
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False)

    def validate(self) -> None:
        if self.primary_model not in {"catboost", "lightgbm"}:
            raise ValueError("primary_model must be 'catboost' or 'lightgbm'.")

        if self.target_unit not in {"seconds", "minutes"}:
            raise ValueError("target_unit must be 'seconds' or 'minutes'.")

        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-8:
            raise ValueError("train_fraction + validation_fraction + test_fraction must equal 1.0.")

        if self.catboost_iterations < 1:
            raise ValueError("catboost_iterations must be >= 1.")
        if self.catboost_early_stopping_rounds < 1:
            raise ValueError("catboost_early_stopping_rounds must be >= 1.")
        if self.primary_model == "catboost" and len(self.catboost_param_grid) == 0:
            raise ValueError("catboost_param_grid must contain at least one candidate when primary_model='catboost'.")
        if self.primary_model == "lightgbm" and len(self.lgbm_param_grid) == 0:
            raise ValueError("lgbm_param_grid must contain at least one candidate when primary_model='lightgbm'.")

        if self.champion_scaling_method != "avg_duration_delta":
            raise ValueError("champion_scaling_method currently supports only 'avg_duration_delta'.")
        if self.champion_scaling_min_samples < 1:
            raise ValueError("champion_scaling_min_samples must be >= 1.")
        if self.champion_scaling_recency_half_life_days < 1:
            raise ValueError("champion_scaling_recency_half_life_days must be >= 1.")
        if self.draft_conditional_min_samples < 1:
            raise ValueError("draft_conditional_min_samples must be >= 1.")
        if self.use_extended_turret_prior_features and not self.use_turret_prior_features:
            raise ValueError("use_extended_turret_prior_features requires use_turret_prior_features=True.")
        if self.enable_quantile_regression and self.primary_model != "catboost":
            raise ValueError("Quantile regression currently supports only primary_model='catboost'.")

        quantile_levels = [float(level) for level in self.quantile_levels]
        if not quantile_levels:
            raise ValueError("quantile_levels must contain at least one quantile.")
        if any(level <= 0.0 or level >= 1.0 for level in quantile_levels):
            raise ValueError("quantile_levels must be strictly between 0 and 1.")
        if len(set(quantile_levels)) != len(quantile_levels):
            raise ValueError("quantile_levels must not contain duplicates.")
        if 0.5 not in quantile_levels:
            raise ValueError("quantile_levels must include 0.5 so p50 can be evaluated against the point model.")

        if len(self.volatility_threshold_quantiles) != 2:
            raise ValueError("volatility_threshold_quantiles must contain exactly two values.")
        low_q, high_q = [float(level) for level in self.volatility_threshold_quantiles]
        if not (0.0 < low_q < high_q < 1.0):
            raise ValueError("volatility_threshold_quantiles must satisfy 0 < low < high < 1.")


def is_forbidden_feature_column(column_name: str) -> bool:
    name = str(column_name).lower().strip()
    if name in {c.lower() for c in FORBIDDEN_COLUMNS}:
        return True
    return any(pattern in name for pattern in FORBIDDEN_COLUMN_PATTERNS)


def allowed_raw_columns() -> List[str]:
    return sorted(set(ALLOWED_COLUMNS + HISTORICAL_PRIOR_SOURCE_COLUMNS))
