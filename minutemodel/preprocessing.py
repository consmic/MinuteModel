from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import normalize_position, normalize_side, safe_first_non_null

LOGGER = logging.getLogger(__name__)

ROLE_ORDER = ["top", "jng", "mid", "bot", "sup"]
TEAM_META_COLUMNS = ["teamid", "teamname", "firstPick", "ckpm"]
TEAM_HISTORY_SOURCE_COLUMNS = [
    "result",
    "firstblood",
    "firstdragon",
    "dragons",
    "firstherald",
    "heralds",
    "firstbaron",
    "barons",
    "golddiffat15",
]
MATCH_META_COLUMNS = ["league", "year", "split", "playoffs", "date", "patch", "game", "url"]


def _first_stable_value(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    # Prefer mode if several repeated values exist across player/team rows.
    mode = non_null.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return non_null.iloc[0]


def _extract_slot_value(side_df: pd.DataFrame, column: str) -> Any:
    if column not in side_df.columns:
        return None
    value = _first_stable_value(side_df[column])
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _extract_role_champions(side_df: pd.DataFrame) -> Dict[str, Optional[str]]:
    role_champs: Dict[str, Optional[str]] = {role: None for role in ROLE_ORDER}
    if "champion" not in side_df.columns or "position_norm" not in side_df.columns:
        return role_champs

    player_rows = side_df[side_df["position_norm"].isin(ROLE_ORDER)]
    for role, role_df in player_rows.groupby("position_norm"):
        champ = _first_stable_value(role_df["champion"])
        role_champs[role] = None if pd.isna(champ) else str(champ).strip()

    return role_champs


def _extract_team_payload(side_df: pd.DataFrame, side_label: str) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "side": side_label,
    }

    for col in TEAM_META_COLUMNS:
        if col in side_df.columns:
            payload[col] = _first_stable_value(side_df[col])
        else:
            payload[col] = None

    for col in TEAM_HISTORY_SOURCE_COLUMNS:
        if col in side_df.columns:
            payload[col] = _first_stable_value(side_df[col])
        else:
            payload[col] = None

    for idx in range(1, 6):
        payload[f"ban{idx}"] = _extract_slot_value(side_df, f"ban{idx}")
        payload[f"pick{idx}"] = _extract_slot_value(side_df, f"pick{idx}")

    role_champs = _extract_role_champions(side_df)
    for role in ROLE_ORDER:
        payload[f"{role}_champion"] = role_champs[role]

    # Backfill missing pick slots from recovered role champions to keep draft slots consistent.
    role_ordered_champs = [role_champs[role] for role in ROLE_ORDER if role_champs[role]]
    existing_picks = [payload.get(f"pick{idx}") for idx in range(1, 6) if payload.get(f"pick{idx}")]
    fallback_champs = [champ for champ in role_ordered_champs if champ not in existing_picks]
    fallback_idx = 0
    for idx in range(1, 6):
        key = f"pick{idx}"
        if payload.get(key):
            continue
        if fallback_idx < len(fallback_champs):
            payload[key] = fallback_champs[fallback_idx]
            fallback_idx += 1

    # Build a side-specific list of drafted champions for bag-of-champions features.
    champion_pool: List[str] = []
    for role in ROLE_ORDER:
        champ = role_champs[role]
        if champ:
            champion_pool.append(champ)

    if len(champion_pool) < 5:
        for idx in range(1, 6):
            pick_value = payload.get(f"pick{idx}")
            if pick_value and pick_value not in champion_pool:
                champion_pool.append(pick_value)

    if len(champion_pool) < 5 and "champion" in side_df.columns:
        champs_from_rows = (
            side_df["champion"].dropna().astype(str).str.strip().replace("", np.nan).dropna().unique().tolist()
        )
        for champ in champs_from_rows:
            if champ not in champion_pool:
                champion_pool.append(champ)

    payload["draft_champions"] = champion_pool[:5]
    return payload


def _extract_match_metadata(game_df: pd.DataFrame) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for col in MATCH_META_COLUMNS:
        if col in game_df.columns:
            meta[col] = safe_first_non_null(game_df[col])
        else:
            meta[col] = None
    return meta


def _standardize_gamelength_seconds(value: Any, unit_guess: str) -> Optional[float]:
    if value is None or pd.isna(value):
        return None

    numeric = float(value)
    if numeric <= 0:
        return None

    if unit_guess == "minutes":
        return numeric * 60.0
    if unit_guess == "seconds":
        return numeric

    # Unknown unit: infer by plausible range for pro LoL match durations.
    if numeric > 600:
        return numeric
    if 10 <= numeric <= 80:
        return numeric * 60.0
    return numeric


def flatten_to_match_level(df: pd.DataFrame, config: PipelineConfig, target_unit_guess: str) -> pd.DataFrame:
    """Convert Oracle's Elixir multi-row match data into one row per gameid."""
    required = {"gameid", "side", "gamelength"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Cannot flatten without required columns: {missing}")

    work = df.copy()
    work["side_norm"] = work["side"].map(normalize_side)

    if "position" in work.columns:
        work["position_norm"] = work["position"].map(normalize_position)
    else:
        work["position_norm"] = None

    rows: List[Dict[str, Any]] = []
    dropped_games_missing_side = 0

    for gameid, game_df in work.groupby("gameid", sort=False):
        game_df = game_df.copy()

        sides = {side for side in game_df["side_norm"].dropna().unique().tolist() if side in {"Blue", "Red"}}
        if sides != {"Blue", "Red"}:
            dropped_games_missing_side += 1
            continue

        blue_df = game_df[game_df["side_norm"] == "Blue"]
        red_df = game_df[game_df["side_norm"] == "Red"]

        blue_payload = _extract_team_payload(blue_df, "Blue")
        red_payload = _extract_team_payload(red_df, "Red")

        match_meta = _extract_match_metadata(game_df)

        gamelength_raw = _first_stable_value(pd.to_numeric(game_df["gamelength"], errors="coerce"))
        gamelength_seconds = _standardize_gamelength_seconds(gamelength_raw, target_unit_guess)
        if gamelength_seconds is None:
            continue

        row: Dict[str, Any] = {
            "gameid": gameid,
            **match_meta,
            "blue_team_id": blue_payload.get("teamid"),
            "blue_team_name": blue_payload.get("teamname"),
            "blue_first_pick": blue_payload.get("firstPick"),
            "blue_ckpm_source": blue_payload.get("ckpm"),
            "blue_result_source": blue_payload.get("result"),
            "blue_firstblood_source": blue_payload.get("firstblood"),
            "blue_firstdragon_source": blue_payload.get("firstdragon"),
            "blue_dragons_source": blue_payload.get("dragons"),
            "blue_firstherald_source": blue_payload.get("firstherald"),
            "blue_heralds_source": blue_payload.get("heralds"),
            "blue_firstbaron_source": blue_payload.get("firstbaron"),
            "blue_barons_source": blue_payload.get("barons"),
            "blue_golddiffat15_source": blue_payload.get("golddiffat15"),
            "red_team_id": red_payload.get("teamid"),
            "red_team_name": red_payload.get("teamname"),
            "red_first_pick": red_payload.get("firstPick"),
            "red_ckpm_source": red_payload.get("ckpm"),
            "red_result_source": red_payload.get("result"),
            "red_firstblood_source": red_payload.get("firstblood"),
            "red_firstdragon_source": red_payload.get("firstdragon"),
            "red_dragons_source": red_payload.get("dragons"),
            "red_firstherald_source": red_payload.get("firstherald"),
            "red_heralds_source": red_payload.get("heralds"),
            "red_firstbaron_source": red_payload.get("firstbaron"),
            "red_barons_source": red_payload.get("barons"),
            "red_golddiffat15_source": red_payload.get("golddiffat15"),
            "target_gamelength_seconds": gamelength_seconds,
            "target_gamelength_minutes": gamelength_seconds / 60.0,
        }

        for role in ROLE_ORDER:
            row[f"blue_{role}_champion"] = blue_payload.get(f"{role}_champion")
            row[f"red_{role}_champion"] = red_payload.get(f"{role}_champion")

        for idx in range(1, 6):
            row[f"blue_ban{idx}"] = blue_payload.get(f"ban{idx}")
            row[f"blue_pick{idx}"] = blue_payload.get(f"pick{idx}")
            row[f"red_ban{idx}"] = red_payload.get(f"ban{idx}")
            row[f"red_pick{idx}"] = red_payload.get(f"pick{idx}")

        row["blue_draft_champions"] = blue_payload.get("draft_champions", [])
        row["red_draft_champions"] = red_payload.get("draft_champions", [])

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("Flattening produced an empty match-level table. Check input schema assumptions.")

    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=False)
    out = out.sort_values(["date", "gameid"]).drop_duplicates(subset=["gameid"], keep="last")

    if config.target_unit == "minutes":
        out["target_value"] = out["target_gamelength_minutes"]
    else:
        out["target_value"] = out["target_gamelength_seconds"]

    LOGGER.info("Match-level table rows: %d", len(out))
    if dropped_games_missing_side:
        LOGGER.warning("Dropped %d games without both blue/red sides.", dropped_games_missing_side)

    return out.reset_index(drop=True)
