
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from .champion_archetypes import (
    SCORE_COLUMNS,
    aggregate_team_scores,
    classify_archetype,
    is_skirmish_comp,
    normalize_champion_name,
)
from .champion_scaling import ChampionScalingLookup
from .config import PipelineConfig, is_forbidden_feature_column

LOGGER = logging.getLogger(__name__)

ROLE_ORDER = ["top", "jng", "mid", "bot", "sup"]

BASE_PRIOR_COLUMNS: List[str] = [
    "blue_rolling_duration_prior_seconds",
    "red_rolling_duration_prior_seconds",
    "rolling_duration_prior_diff_seconds",
    "blue_rolling_ckpm_prior",
    "red_rolling_ckpm_prior",
    "rolling_ckpm_prior_diff",
]

EXTENDED_PRIOR_COLUMNS: List[str] = [
    "blue_rolling_win_rate_prior",
    "red_rolling_win_rate_prior",
    "rolling_win_rate_prior_diff",
    "blue_rolling_firstblood_rate_prior",
    "red_rolling_firstblood_rate_prior",
    "rolling_firstblood_rate_prior_diff",
    "blue_rolling_firstdragon_rate_prior",
    "red_rolling_firstdragon_rate_prior",
    "rolling_firstdragon_rate_prior_diff",
    "blue_rolling_dragons_by20_prior",
    "red_rolling_dragons_by20_prior",
    "rolling_dragons_by20_prior_diff",
    "blue_rolling_herald_rate_prior",
    "red_rolling_herald_rate_prior",
    "rolling_herald_rate_prior_diff",
    "blue_rolling_baron_rate_prior",
    "red_rolling_baron_rate_prior",
    "rolling_baron_rate_prior_diff",
    "blue_rolling_golddiff15_prior",
    "red_rolling_golddiff15_prior",
    "rolling_golddiff15_prior_diff",
    "blue_rolling_side_win_rate_prior",
    "red_rolling_side_win_rate_prior",
    "rolling_side_win_rate_prior_diff",
    "blue_rolling_side_duration_prior_seconds",
    "red_rolling_side_duration_prior_seconds",
    "rolling_side_duration_prior_diff_seconds",
]

CONDITIONAL_PRIOR_COLUMNS: List[str] = [
    "blue_conditional_duration_when_early_prior_seconds",
    "red_conditional_duration_when_early_prior_seconds",
    "conditional_duration_when_early_prior_diff_seconds",
    "blue_conditional_duration_when_scaling_prior_seconds",
    "red_conditional_duration_when_scaling_prior_seconds",
    "conditional_duration_when_scaling_prior_diff_seconds",
    "blue_conditional_firstdragon_when_early_prior",
    "red_conditional_firstdragon_when_early_prior",
    "conditional_firstdragon_when_early_prior_diff",
    "blue_conditional_herald_vs_scaling_prior",
    "red_conditional_herald_vs_scaling_prior",
    "conditional_herald_vs_scaling_prior_diff",
    "blue_conditional_ckpm_skirmish_prior",
    "red_conditional_ckpm_skirmish_prior",
    "conditional_ckpm_skirmish_prior_diff",
    "blue_conditional_duration_similar_matchup_prior_seconds",
    "red_conditional_duration_similar_matchup_prior_seconds",
    "conditional_duration_similar_matchup_prior_diff_seconds",
]

DRAFT_SUMMARY_OUTPUT_COLUMNS: List[str] = (
    [
        f"blue_{score}" for score in SCORE_COLUMNS
    ]
    + [f"red_{score}" for score in SCORE_COLUMNS]
    + [f"{score}_diff" for score in SCORE_COLUMNS]
    + [
        "blue_comp_archetype",
        "red_comp_archetype",
        "comp_archetype_matchup",
        "blue_is_skirmish_comp",
        "red_is_skirmish_comp",
    ]
)

DRAFT_INTERACTION_COLUMNS: List[str] = [
    "blue_early_vs_red_scaling",
    "red_early_vs_blue_scaling",
    "blue_engage_vs_red_peel",
    "red_engage_vs_blue_peel",
    "expected_match_pace_score",
    "expected_snowball_vs_stall_score",
]


def _to_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _safe_group_key(team_id: Any, team_name: Any, side: str) -> str:
    if pd.notna(team_id) and str(team_id).strip() not in {"", "nan", "None"}:
        return f"id::{team_id}"
    if pd.notna(team_name) and str(team_name).strip() not in {"", "nan", "None"}:
        return f"name::{team_name}"
    return f"unknown::{side}"


def _coerce_binary_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        clipped = (numeric > 0).astype(float)
        clipped = clipped.where(numeric.notna(), np.nan)
        return clipped

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1.0,
        "0": 0.0,
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "win": 1.0,
        "won": 1.0,
        "loss": 0.0,
        "lose": 0.0,
        "lost": 0.0,
    }
    return text.map(mapping).astype(float)


def _coerce_result_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        out = numeric.copy().astype(float)
        out = np.where(out > 0, 1.0, 0.0)
        return pd.Series(out, index=series.index, dtype=float)

    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1.0,
        "0": 0.0,
        "true": 1.0,
        "false": 0.0,
        "win": 1.0,
        "won": 1.0,
        "loss": 0.0,
        "lose": 0.0,
        "lost": 0.0,
    }
    return text.map(mapping).astype(float)


def _sanitize_draft_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [normalize_champion_name(v) for v in value if normalize_champion_name(v)]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        cleaned = text[1:-1].replace("'", "").replace('"', "")
        tokens = [normalize_champion_name(tok) for tok in cleaned.split(",") if normalize_champion_name(tok)]
        return tokens
    champion = normalize_champion_name(text)
    return [champion] if champion else []


def _extract_side_champions(row: pd.Series, side: str, prefer_role_specific: bool = True) -> List[str]:
    side = side.lower()
    champions: List[str] = []

    if prefer_role_specific:
        for role in ROLE_ORDER:
            champ = normalize_champion_name(row.get(f"{side}_{role}_champion"))
            if champ and champ not in champions:
                champions.append(champ)

    for idx in range(1, 6):
        pick = normalize_champion_name(row.get(f"{side}_pick{idx}"))
        if pick and pick not in champions:
            champions.append(pick)

    list_values = _sanitize_draft_list(row.get(f"{side}_draft_champions"))
    for champ in list_values:
        if champ not in champions:
            champions.append(champ)

    return champions[:5]


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


def _shifted_group_expanding_mean(df: pd.DataFrame, group_cols: str | Sequence[str], value_col: str) -> pd.Series:
    return (
        df.groupby(group_cols, group_keys=False)[value_col]
        .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
        .reset_index(drop=True)
    )


def _shifted_conditional_mean(
    df: pd.DataFrame,
    group_cols: str | Sequence[str],
    value_col: str,
    condition_col: str,
) -> pd.Series:
    def _calc(group: pd.DataFrame) -> pd.Series:
        values = _to_float_series(group[value_col])
        cond = _to_float_series(group[condition_col]).fillna(0.0)
        cond = (cond > 0).astype(float)

        prior_sum = (values * cond).shift(1).cumsum()
        prior_count = cond.shift(1).cumsum()
        return prior_sum / prior_count.replace(0.0, np.nan)

    return (
        df.groupby(group_cols, group_keys=False)[[value_col, condition_col]]
        .apply(_calc)
        .reset_index(drop=True)
    )


def _shifted_global_conditional_mean(df: pd.DataFrame, value_col: str, condition_col: str) -> pd.Series:
    values = _to_float_series(df[value_col])
    cond = _to_float_series(df[condition_col]).fillna(0.0)
    cond = (cond > 0).astype(float)

    prior_sum = (values * cond).shift(1).cumsum()
    prior_count = cond.shift(1).cumsum()
    return prior_sum / prior_count.replace(0.0, np.nan)


def _fill_prior_with_fallbacks(
    history: pd.DataFrame,
    raw_prior: pd.Series,
    value_col: str,
    fallback_global_col: str,
    fallback_league_col: str,
) -> pd.Series:
    filled = raw_prior.copy()
    filled = filled.fillna(history[fallback_league_col])
    filled = filled.fillna(history[fallback_global_col])
    overall = float(_to_float_series(history[value_col]).mean())
    if not np.isfinite(overall):
        overall = 0.0
    return filled.fillna(overall)


def _compute_prior(
    history: pd.DataFrame,
    value_col: str,
    out_col: str,
    window: int,
    group_col: str = "team_key",
) -> None:
    raw = _shifted_team_rolling_mean(history, group_col=group_col, value_col=value_col, window=window)
    history[out_col] = _fill_prior_with_fallbacks(
        history=history,
        raw_prior=raw,
        value_col=value_col,
        fallback_global_col=f"global_{value_col}_prior",
        fallback_league_col=f"league_{value_col}_prior",
    )


def _add_group_fallback_columns(history: pd.DataFrame, value_cols: Sequence[str]) -> None:
    for col in value_cols:
        history[f"league_{col}_prior"] = _shifted_group_expanding_mean(history, group_cols="league", value_col=col)
        history[f"global_{col}_prior"] = history[col].shift(1).expanding(min_periods=1).mean()


def _build_side_draft_context(match_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for _, row in match_df.iterrows():
        blue_champs = _extract_side_champions(row, side="blue", prefer_role_specific=True)
        red_champs = _extract_side_champions(row, side="red", prefer_role_specific=True)

        blue_scores = aggregate_team_scores(blue_champs)
        red_scores = aggregate_team_scores(red_champs)

        blue_arch = classify_archetype(blue_scores)
        red_arch = classify_archetype(red_scores)

        rows.append(
            {
                "gameid": row["gameid"],
                "blue_scores": blue_scores,
                "red_scores": red_scores,
                "blue_archetype": blue_arch,
                "red_archetype": red_arch,
                "blue_is_skirmish": float(is_skirmish_comp(blue_scores)),
                "red_is_skirmish": float(is_skirmish_comp(red_scores)),
            }
        )
    return pd.DataFrame(rows)


def _build_team_history(match_df: pd.DataFrame) -> pd.DataFrame:
    context_df = _build_side_draft_context(match_df)
    context_map = context_df.set_index("gameid")

    rows: List[Dict[str, Any]] = []
    for _, row in match_df.iterrows():
        gameid = row["gameid"]
        ctx = context_map.loc[gameid]

        rows.append(
            {
                "gameid": gameid,
                "date": row["date"],
                "league": row.get("league"),
                "patch": row.get("patch"),
                "side": "Blue",
                "team_id": row.get("blue_team_id"),
                "team_name": row.get("blue_team_name"),
                "target_gamelength_seconds": row.get("target_gamelength_seconds"),
                "ckpm_source": row.get("blue_ckpm_source"),
                "result_source": row.get("blue_result_source"),
                "firstblood_source": row.get("blue_firstblood_source"),
                "firstdragon_source": row.get("blue_firstdragon_source"),
                "dragons_source": row.get("blue_dragons_source"),
                "firstherald_source": row.get("blue_firstherald_source"),
                "heralds_source": row.get("blue_heralds_source"),
                "firstbaron_source": row.get("blue_firstbaron_source"),
                "barons_source": row.get("blue_barons_source"),
                "golddiffat15_source": row.get("blue_golddiffat15_source"),
                "team_archetype": ctx["blue_archetype"],
                "opp_archetype": ctx["red_archetype"],
                "is_early_comp": float(ctx["blue_archetype"] == "early"),
                "is_scaling_comp": float(ctx["blue_archetype"] == "scaling"),
                "is_skirmish_comp": float(ctx["blue_is_skirmish"]),
                "opp_is_scaling": float(ctx["red_archetype"] == "scaling"),
            }
        )
        rows.append(
            {
                "gameid": gameid,
                "date": row["date"],
                "league": row.get("league"),
                "patch": row.get("patch"),
                "side": "Red",
                "team_id": row.get("red_team_id"),
                "team_name": row.get("red_team_name"),
                "target_gamelength_seconds": row.get("target_gamelength_seconds"),
                "ckpm_source": row.get("red_ckpm_source"),
                "result_source": row.get("red_result_source"),
                "firstblood_source": row.get("red_firstblood_source"),
                "firstdragon_source": row.get("red_firstdragon_source"),
                "dragons_source": row.get("red_dragons_source"),
                "firstherald_source": row.get("red_firstherald_source"),
                "heralds_source": row.get("red_heralds_source"),
                "firstbaron_source": row.get("red_firstbaron_source"),
                "barons_source": row.get("red_barons_source"),
                "golddiffat15_source": row.get("red_golddiffat15_source"),
                "team_archetype": ctx["red_archetype"],
                "opp_archetype": ctx["blue_archetype"],
                "is_early_comp": float(ctx["red_archetype"] == "early"),
                "is_scaling_comp": float(ctx["red_archetype"] == "scaling"),
                "is_skirmish_comp": float(ctx["red_is_skirmish"]),
                "opp_is_scaling": float(ctx["blue_archetype"] == "scaling"),
            }
        )

    history = pd.DataFrame(rows)
    history["date"] = pd.to_datetime(history["date"], errors="coerce", utc=False)
    history = history.sort_values(["date", "gameid", "side"]).reset_index(drop=True)

    history["team_key"] = history.apply(
        lambda r: _safe_group_key(r.get("team_id"), r.get("team_name"), str(r.get("side"))),
        axis=1,
    )
    history["team_side_key"] = history["team_key"] + "::" + history["side"].astype(str)
    history["archetype_matchup"] = history["team_archetype"].astype(str) + "_vs_" + history["opp_archetype"].astype(str)

    history["target_gamelength_seconds"] = _to_float_series(history["target_gamelength_seconds"])
    history["ckpm_source"] = _to_float_series(history["ckpm_source"])
    history["result_source"] = _coerce_result_series(history["result_source"])
    history["firstblood_source"] = _coerce_binary_series(history["firstblood_source"])
    history["firstdragon_source"] = _coerce_binary_series(history["firstdragon_source"])
    history["dragons_source"] = _to_float_series(history["dragons_source"])
    history["firstherald_source"] = _coerce_binary_series(history["firstherald_source"])
    history["heralds_source"] = _to_float_series(history["heralds_source"])
    history["firstbaron_source"] = _coerce_binary_series(history["firstbaron_source"])
    history["barons_source"] = _to_float_series(history["barons_source"])
    history["golddiffat15_source"] = _to_float_series(history["golddiffat15_source"])

    # Proxy for "dragons by 20" where explicit @20 objective timings are unavailable.
    history["dragons_by20_proxy_source"] = history["dragons_source"]

    return history

def _merge_side_features(
    match_df: pd.DataFrame,
    history: pd.DataFrame,
    feature_cols: Sequence[str],
) -> pd.DataFrame:
    blue = history[history["side"] == "Blue"][["gameid", *feature_cols]].rename(
        columns={col: f"blue_{col}" for col in feature_cols}
    )
    red = history[history["side"] == "Red"][["gameid", *feature_cols]].rename(
        columns={col: f"red_{col}" for col in feature_cols}
    )

    merged = match_df.merge(blue, on="gameid", how="left").merge(red, on="gameid", how="left")
    return merged


def _add_diff_feature(merged: pd.DataFrame, base_col: str, diff_col: str) -> None:
    blue_col = f"blue_{base_col}"
    red_col = f"red_{base_col}"
    if blue_col in merged.columns and red_col in merged.columns:
        merged[diff_col] = _to_float_series(merged[blue_col]) - _to_float_series(merged[red_col])


def _build_conditional_priors(history: pd.DataFrame, config: PipelineConfig) -> None:
    min_samples = int(max(config.draft_conditional_min_samples, 1))

    history["cond_duration_early_team"] = _shifted_conditional_mean(
        history,
        group_cols="team_key",
        value_col="target_gamelength_seconds",
        condition_col="is_early_comp",
    )
    history["cond_duration_early_league"] = _shifted_conditional_mean(
        history,
        group_cols="league",
        value_col="target_gamelength_seconds",
        condition_col="is_early_comp",
    )
    history["cond_duration_early_global"] = _shifted_global_conditional_mean(
        history,
        value_col="target_gamelength_seconds",
        condition_col="is_early_comp",
    )

    history["conditional_duration_when_early_prior_seconds"] = (
        history["cond_duration_early_team"]
        .fillna(history["cond_duration_early_league"])
        .fillna(history["cond_duration_early_global"])
        .fillna(history["rolling_duration_prior_seconds"])
    )

    history["cond_duration_scaling_team"] = _shifted_conditional_mean(
        history,
        group_cols="team_key",
        value_col="target_gamelength_seconds",
        condition_col="is_scaling_comp",
    )
    history["cond_duration_scaling_league"] = _shifted_conditional_mean(
        history,
        group_cols="league",
        value_col="target_gamelength_seconds",
        condition_col="is_scaling_comp",
    )
    history["cond_duration_scaling_global"] = _shifted_global_conditional_mean(
        history,
        value_col="target_gamelength_seconds",
        condition_col="is_scaling_comp",
    )

    history["conditional_duration_when_scaling_prior_seconds"] = (
        history["cond_duration_scaling_team"]
        .fillna(history["cond_duration_scaling_league"])
        .fillna(history["cond_duration_scaling_global"])
        .fillna(history["rolling_duration_prior_seconds"])
    )

    history["cond_firstdragon_early_team"] = _shifted_conditional_mean(
        history,
        group_cols="team_key",
        value_col="firstdragon_source",
        condition_col="is_early_comp",
    )
    history["cond_firstdragon_early_league"] = _shifted_conditional_mean(
        history,
        group_cols="league",
        value_col="firstdragon_source",
        condition_col="is_early_comp",
    )
    history["cond_firstdragon_early_global"] = _shifted_global_conditional_mean(
        history,
        value_col="firstdragon_source",
        condition_col="is_early_comp",
    )
    history["conditional_firstdragon_when_early_prior"] = (
        history["cond_firstdragon_early_team"]
        .fillna(history["cond_firstdragon_early_league"])
        .fillna(history["cond_firstdragon_early_global"])
        .fillna(history["rolling_firstdragon_rate_prior"])
    )

    history["cond_herald_vs_scaling_team"] = _shifted_conditional_mean(
        history,
        group_cols="team_key",
        value_col="firstherald_source",
        condition_col="opp_is_scaling",
    )
    history["cond_herald_vs_scaling_league"] = _shifted_conditional_mean(
        history,
        group_cols="league",
        value_col="firstherald_source",
        condition_col="opp_is_scaling",
    )
    history["cond_herald_vs_scaling_global"] = _shifted_global_conditional_mean(
        history,
        value_col="firstherald_source",
        condition_col="opp_is_scaling",
    )
    history["conditional_herald_vs_scaling_prior"] = (
        history["cond_herald_vs_scaling_team"]
        .fillna(history["cond_herald_vs_scaling_league"])
        .fillna(history["cond_herald_vs_scaling_global"])
        .fillna(history["rolling_herald_rate_prior"])
    )

    history["cond_ckpm_skirmish_team"] = _shifted_conditional_mean(
        history,
        group_cols="team_key",
        value_col="ckpm_source",
        condition_col="is_skirmish_comp",
    )
    history["cond_ckpm_skirmish_league"] = _shifted_conditional_mean(
        history,
        group_cols="league",
        value_col="ckpm_source",
        condition_col="is_skirmish_comp",
    )
    history["cond_ckpm_skirmish_global"] = _shifted_global_conditional_mean(
        history,
        value_col="ckpm_source",
        condition_col="is_skirmish_comp",
    )
    history["conditional_ckpm_skirmish_prior"] = (
        history["cond_ckpm_skirmish_team"]
        .fillna(history["cond_ckpm_skirmish_league"])
        .fillna(history["cond_ckpm_skirmish_global"])
        .fillna(history["rolling_ckpm_prior"])
    )

    history["cond_matchup_team"] = _shifted_group_expanding_mean(
        history,
        group_cols=["team_key", "archetype_matchup"],
        value_col="target_gamelength_seconds",
    )
    history["cond_matchup_team_count"] = (
        history.groupby(["team_key", "archetype_matchup"], group_keys=False)
        .cumcount()
        .astype(float)
    )
    history["cond_matchup_league"] = _shifted_group_expanding_mean(
        history,
        group_cols=["league", "archetype_matchup"],
        value_col="target_gamelength_seconds",
    )
    history["cond_matchup_global"] = _shifted_group_expanding_mean(
        history,
        group_cols="archetype_matchup",
        value_col="target_gamelength_seconds",
    )

    sparse_mask = history["cond_matchup_team_count"] < float(min_samples)
    matchup_base = history["cond_matchup_team"].where(~sparse_mask, np.nan)
    history["conditional_duration_similar_matchup_prior_seconds"] = (
        matchup_base
        .fillna(history["cond_matchup_league"])
        .fillna(history["cond_matchup_global"])
        .fillna(history["rolling_duration_prior_seconds"])
    )


def build_leakage_safe_rolling_priors(match_df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Create leakage-safe historical priors using strictly earlier matches only.

    Important safety rule:
    - all priors are computed on chronologically sorted history using `shift(1)`
      before any rolling/expanding aggregation.
    - this prevents current/future match outcomes from entering current-row features.
    """
    if match_df.empty:
        return match_df.copy()

    work = match_df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce", utc=False)
    work = work.sort_values(["date", "gameid"]).reset_index(drop=True)

    history = _build_team_history(work)

    numeric_value_cols = [
        "target_gamelength_seconds",
        "ckpm_source",
        "result_source",
        "firstblood_source",
        "firstdragon_source",
        "dragons_by20_proxy_source",
        "firstherald_source",
        "firstbaron_source",
        "golddiffat15_source",
    ]
    _add_group_fallback_columns(history, value_cols=numeric_value_cols)

    window = int(max(config.rolling_window_size, 1))

    _compute_prior(
        history,
        value_col="target_gamelength_seconds",
        out_col="rolling_duration_prior_seconds",
        window=window,
        group_col="team_key",
    )

    if config.use_rolling_ckpm_prior and history["ckpm_source"].notna().any():
        _compute_prior(
            history,
            value_col="ckpm_source",
            out_col="rolling_ckpm_prior",
            window=window,
            group_col="team_key",
        )
    else:
        history["rolling_ckpm_prior"] = np.nan

    _compute_prior(
        history,
        value_col="result_source",
        out_col="rolling_win_rate_prior",
        window=window,
        group_col="team_key",
    )
    _compute_prior(
        history,
        value_col="firstblood_source",
        out_col="rolling_firstblood_rate_prior",
        window=window,
        group_col="team_key",
    )
    _compute_prior(
        history,
        value_col="firstdragon_source",
        out_col="rolling_firstdragon_rate_prior",
        window=window,
        group_col="team_key",
    )
    _compute_prior(
        history,
        value_col="dragons_by20_proxy_source",
        out_col="rolling_dragons_by20_prior",
        window=window,
        group_col="team_key",
    )
    _compute_prior(
        history,
        value_col="firstherald_source",
        out_col="rolling_herald_rate_prior",
        window=window,
        group_col="team_key",
    )
    _compute_prior(
        history,
        value_col="firstbaron_source",
        out_col="rolling_baron_rate_prior",
        window=window,
        group_col="team_key",
    )
    _compute_prior(
        history,
        value_col="golddiffat15_source",
        out_col="rolling_golddiff15_prior",
        window=window,
        group_col="team_key",
    )
    _compute_prior(
        history,
        value_col="result_source",
        out_col="rolling_side_win_rate_prior",
        window=window,
        group_col="team_side_key",
    )
    _compute_prior(
        history,
        value_col="target_gamelength_seconds",
        out_col="rolling_side_duration_prior_seconds",
        window=window,
        group_col="team_side_key",
    )

    _build_conditional_priors(history, config=config)

    side_feature_cols = [
        "rolling_duration_prior_seconds",
        "rolling_ckpm_prior",
        "rolling_win_rate_prior",
        "rolling_firstblood_rate_prior",
        "rolling_firstdragon_rate_prior",
        "rolling_dragons_by20_prior",
        "rolling_herald_rate_prior",
        "rolling_baron_rate_prior",
        "rolling_golddiff15_prior",
        "rolling_side_win_rate_prior",
        "rolling_side_duration_prior_seconds",
        "conditional_duration_when_early_prior_seconds",
        "conditional_duration_when_scaling_prior_seconds",
        "conditional_firstdragon_when_early_prior",
        "conditional_herald_vs_scaling_prior",
        "conditional_ckpm_skirmish_prior",
        "conditional_duration_similar_matchup_prior_seconds",
    ]

    merged = _merge_side_features(work, history, side_feature_cols)

    diff_map = {
        "rolling_duration_prior_seconds": "rolling_duration_prior_diff_seconds",
        "rolling_ckpm_prior": "rolling_ckpm_prior_diff",
        "rolling_win_rate_prior": "rolling_win_rate_prior_diff",
        "rolling_firstblood_rate_prior": "rolling_firstblood_rate_prior_diff",
        "rolling_firstdragon_rate_prior": "rolling_firstdragon_rate_prior_diff",
        "rolling_dragons_by20_prior": "rolling_dragons_by20_prior_diff",
        "rolling_herald_rate_prior": "rolling_herald_rate_prior_diff",
        "rolling_baron_rate_prior": "rolling_baron_rate_prior_diff",
        "rolling_golddiff15_prior": "rolling_golddiff15_prior_diff",
        "rolling_side_win_rate_prior": "rolling_side_win_rate_prior_diff",
        "rolling_side_duration_prior_seconds": "rolling_side_duration_prior_diff_seconds",
        "conditional_duration_when_early_prior_seconds": "conditional_duration_when_early_prior_diff_seconds",
        "conditional_duration_when_scaling_prior_seconds": "conditional_duration_when_scaling_prior_diff_seconds",
        "conditional_firstdragon_when_early_prior": "conditional_firstdragon_when_early_prior_diff",
        "conditional_herald_vs_scaling_prior": "conditional_herald_vs_scaling_prior_diff",
        "conditional_ckpm_skirmish_prior": "conditional_ckpm_skirmish_prior_diff",
        "conditional_duration_similar_matchup_prior_seconds": "conditional_duration_similar_matchup_prior_diff_seconds",
    }
    for base_col, diff_col in diff_map.items():
        _add_diff_feature(merged, base_col=base_col, diff_col=diff_col)

    return merged


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
    numeric_fill_values_: Dict[str, float] = field(default_factory=dict)

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
            blue_lists = match_df.get("blue_draft_champions", pd.Series([[]] * len(match_df))).apply(_sanitize_draft_list)
            red_lists = match_df.get("red_draft_champions", pd.Series([[]] * len(match_df))).apply(_sanitize_draft_list)
            self.blue_mlb.fit(blue_lists)
            self.red_mlb.fit(red_lists)
            self.blue_classes_set_ = set(self.blue_mlb.classes_.tolist())
            self.red_classes_set_ = set(self.red_mlb.classes_.tolist())

        prior_cols = BASE_PRIOR_COLUMNS.copy()
        if self.config.use_extended_rolling_team_priors:
            prior_cols += EXTENDED_PRIOR_COLUMNS
        if self.config.use_draft_conditional_behaviour_features:
            prior_cols += CONDITIONAL_PRIOR_COLUMNS

        self.numeric_fill_values_ = {}
        for col in prior_cols:
            values = _to_float_series(match_df.get(col, pd.Series([np.nan] * len(match_df))))
            median = float(values.median()) if values.notna().any() else 0.0
            if not np.isfinite(median):
                median = 0.0
            self.numeric_fill_values_[col] = median

        self.fitted = True
        return self

    def _numeric_feature(self, match_df: pd.DataFrame, col: str) -> pd.Series:
        default = float(self.numeric_fill_values_.get(col, 0.0))
        values = _to_float_series(match_df.get(col, pd.Series([np.nan] * len(match_df), index=match_df.index)))
        return values.fillna(default)

    def _build_draft_summary_features(self, match_df: pd.DataFrame) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for _, row in match_df.iterrows():
            blue_champs = _extract_side_champions(
                row,
                side="blue",
                prefer_role_specific=self.config.use_role_specific_draft_features,
            )
            red_champs = _extract_side_champions(
                row,
                side="red",
                prefer_role_specific=self.config.use_role_specific_draft_features,
            )

            blue_scores = aggregate_team_scores(blue_champs)
            red_scores = aggregate_team_scores(red_champs)
            blue_arch = classify_archetype(blue_scores)
            red_arch = classify_archetype(red_scores)

            row_out: Dict[str, Any] = {}
            for score in SCORE_COLUMNS:
                row_out[f"blue_{score}"] = float(blue_scores.get(score, 0.5))
                row_out[f"red_{score}"] = float(red_scores.get(score, 0.5))
                row_out[f"{score}_diff"] = float(row_out[f"blue_{score}"] - row_out[f"red_{score}"])

            row_out["blue_comp_archetype"] = blue_arch
            row_out["red_comp_archetype"] = red_arch
            row_out["comp_archetype_matchup"] = f"{blue_arch}_vs_{red_arch}"
            row_out["blue_is_skirmish_comp"] = float(is_skirmish_comp(blue_scores))
            row_out["red_is_skirmish_comp"] = float(is_skirmish_comp(red_scores))
            rows.append(row_out)

        return pd.DataFrame(rows, index=match_df.index)

    @staticmethod
    def _build_draft_interaction_features(summary_df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=summary_df.index)

        out["blue_early_vs_red_scaling"] = _to_float_series(summary_df["blue_early_game_score"]) - _to_float_series(
            summary_df["red_scaling_score"]
        )
        out["red_early_vs_blue_scaling"] = _to_float_series(summary_df["red_early_game_score"]) - _to_float_series(
            summary_df["blue_scaling_score"]
        )
        out["blue_engage_vs_red_peel"] = _to_float_series(summary_df["blue_engage_score"]) - _to_float_series(
            summary_df["red_peel_or_disengage_score"]
        )
        out["red_engage_vs_blue_peel"] = _to_float_series(summary_df["red_engage_score"]) - _to_float_series(
            summary_df["blue_peel_or_disengage_score"]
        )

        blue_tempo = (
            _to_float_series(summary_df["blue_early_game_score"])
            + _to_float_series(summary_df["blue_engage_score"])
            + _to_float_series(summary_df["blue_pick_catch_score"])
            + _to_float_series(summary_df["blue_objective_burn_score"])
        ) / 4.0
        red_tempo = (
            _to_float_series(summary_df["red_early_game_score"])
            + _to_float_series(summary_df["red_engage_score"])
            + _to_float_series(summary_df["red_pick_catch_score"])
            + _to_float_series(summary_df["red_objective_burn_score"])
        ) / 4.0
        overall_scaling = (
            _to_float_series(summary_df["blue_scaling_score"]) + _to_float_series(summary_df["red_scaling_score"])
        ) / 2.0
        overall_stall = (
            _to_float_series(summary_df["blue_waveclear_score"])
            + _to_float_series(summary_df["red_waveclear_score"])
            + _to_float_series(summary_df["blue_peel_or_disengage_score"])
            + _to_float_series(summary_df["red_peel_or_disengage_score"])
        ) / 4.0

        out["expected_match_pace_score"] = (blue_tempo + red_tempo) / 2.0 - 0.35 * overall_scaling
        out["expected_snowball_vs_stall_score"] = (blue_tempo + red_tempo) / 2.0 - overall_stall

        return out

    def transform(self, match_df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("DraftFeatureBuilder must be fitted before calling transform().")

        features = pd.DataFrame(index=match_df.index)

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

        prior_cols = BASE_PRIOR_COLUMNS.copy()
        if self.config.use_extended_rolling_team_priors:
            prior_cols += EXTENDED_PRIOR_COLUMNS
        if self.config.use_draft_conditional_behaviour_features:
            prior_cols += CONDITIONAL_PRIOR_COLUMNS

        for col in prior_cols:
            features[col] = self._numeric_feature(match_df, col)

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

        draft_summary_df: Optional[pd.DataFrame] = None
        if self.config.use_draft_summary_features or self.config.use_draft_interaction_features:
            draft_summary_df = self._build_draft_summary_features(match_df)

        if self.config.use_draft_summary_features and draft_summary_df is not None:
            summary_cols = [col for col in DRAFT_SUMMARY_OUTPUT_COLUMNS if col in draft_summary_df.columns]
            summary_payload = draft_summary_df[summary_cols].copy()
            for col in summary_cols:
                if str(summary_payload[col].dtype) == "object":
                    summary_payload[col] = summary_payload[col].astype("object")
                else:
                    summary_payload[col] = pd.to_numeric(summary_payload[col], errors="coerce")
            features = pd.concat([features, summary_payload], axis=1)

        if self.config.use_draft_interaction_features and draft_summary_df is not None:
            interaction_df = self._build_draft_interaction_features(draft_summary_df)
            interaction_payload = interaction_df[[col for col in DRAFT_INTERACTION_COLUMNS if col in interaction_df.columns]].copy()
            for col in interaction_payload.columns:
                interaction_payload[col] = pd.to_numeric(interaction_payload[col], errors="coerce")
            features = pd.concat([features, interaction_payload], axis=1)

        if self.config.use_bag_of_champions_fallback:
            blue_lists_raw = match_df.get("blue_draft_champions", pd.Series([[]] * len(match_df))).apply(_sanitize_draft_list)
            red_lists_raw = match_df.get("red_draft_champions", pd.Series([[]] * len(match_df))).apply(_sanitize_draft_list)

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

        categorical_columns = [col for col in features.columns if str(features[col].dtype) == "object"]
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
