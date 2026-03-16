from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from minutemodel.inference import load_artifacts

ROLE_ORDER = ["top", "jng", "mid", "bot", "sup"]
MAJOR_LEAGUES = {"LCK", "LPL", "LEC", "LCS", "MSI", "WORLDS"}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def parse_champion_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [clean_text(v) for v in value if clean_text(v)]
    text = clean_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].replace("'", "").replace('"', "")
        return [token.strip() for token in inner.split(",") if token.strip()]
    return [text]


def normalize_team_id(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    try:
        as_float = float(text)
        if as_float.is_integer():
            return str(int(as_float))
    except Exception:
        return text
    return text


def options_with_current(options: List[str], current: str) -> List[str]:
    clean_options = [opt for opt in sorted({clean_text(v) for v in options}) if opt]
    current_clean = clean_text(current)
    if current_clean and current_clean not in clean_options:
        clean_options = [current_clean] + clean_options
    return [""] + clean_options


@st.cache_resource
def load_artifacts_cached(path: str) -> Dict[str, Any]:
    return load_artifacts(path)


@st.cache_data
def load_match_table(path: str) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(table_path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    return df


@st.cache_data
def load_metrics_payload(metrics_path: str) -> Dict[str, Any]:
    path = Path(metrics_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_test_mae_minutes(metrics_path: str) -> Optional[float]:
    payload = load_metrics_payload(metrics_path)
    metrics = payload.get("metrics_by_model", {})

    primary = payload.get("primary_model")
    if primary and primary in metrics and "mae_minutes" in metrics[primary]:
        return float(metrics[primary]["mae_minutes"])

    for model_name in ["catboost", "lightgbm", "ridge_regression", "global_mean"]:
        if model_name in metrics and "mae_minutes" in metrics[model_name]:
            return float(metrics[model_name]["mae_minutes"])
    return None


def extract_ui_options(match_df: pd.DataFrame) -> Dict[str, Any]:
    if match_df.empty:
        return {
            "leagues": [],
            "patches": [],
            "splits": [],
            "team_names": [],
            "team_ids": [],
            "champions": [],
            "duration_median": 1900.0,
            "ckpm_median": 0.7,
        }

    def unique_values(cols: List[str]) -> List[str]:
        tokens: List[str] = []
        for col in cols:
            if col in match_df.columns:
                tokens.extend([clean_text(v) for v in match_df[col].dropna().tolist()])
        return sorted({tok for tok in tokens if tok})

    champions: List[str] = []
    champion_cols = [
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
    for col in champion_cols:
        if col in match_df.columns:
            champions.extend([clean_text(v) for v in match_df[col].dropna().tolist()])

    for list_col in ["blue_draft_champions", "red_draft_champions"]:
        if list_col in match_df.columns:
            for value in match_df[list_col].dropna().tolist():
                champions.extend(parse_champion_list(value))

    duration_series = pd.to_numeric(match_df.get("target_gamelength_seconds"), errors="coerce").dropna()
    ckpm_values = pd.concat(
        [
            pd.to_numeric(match_df.get("blue_rolling_ckpm_prior"), errors="coerce"),
            pd.to_numeric(match_df.get("red_rolling_ckpm_prior"), errors="coerce"),
        ],
        axis=0,
    ).dropna()

    return {
        "leagues": unique_values(["league"]),
        "patches": unique_values(["patch"]),
        "splits": unique_values(["split"]),
        "team_names": unique_values(["blue_team_name", "red_team_name"]),
        "team_ids": unique_values(["blue_team_id", "red_team_id"]),
        "champions": sorted({c for c in champions if c}),
        "duration_median": float(duration_series.median()) if not duration_series.empty else 1900.0,
        "ckpm_median": float(ckpm_values.median()) if not ckpm_values.empty else 0.7,
    }


def team_history_view(match_df: pd.DataFrame) -> pd.DataFrame:
    if match_df.empty:
        return pd.DataFrame()

    blue = pd.DataFrame(
        {
            "date": match_df.get("date"),
            "team_id": match_df.get("blue_team_id"),
            "team_name": match_df.get("blue_team_name"),
            "rolling_duration_prior_seconds": match_df.get("blue_rolling_duration_prior_seconds"),
            "rolling_ckpm_prior": match_df.get("blue_rolling_ckpm_prior"),
        }
    )
    red = pd.DataFrame(
        {
            "date": match_df.get("date"),
            "team_id": match_df.get("red_team_id"),
            "team_name": match_df.get("red_team_name"),
            "rolling_duration_prior_seconds": match_df.get("red_rolling_duration_prior_seconds"),
            "rolling_ckpm_prior": match_df.get("red_rolling_ckpm_prior"),
        }
    )
    out = pd.concat([blue, red], axis=0, ignore_index=True)
    out["team_id_norm"] = out["team_id"].map(normalize_team_id)
    out["team_name_norm"] = out["team_name"].map(lambda x: clean_text(x).lower())
    out = out.sort_values("date")
    return out


def build_calendar_board(match_df: pd.DataFrame, max_rows: int = 400) -> pd.DataFrame:
    if match_df.empty:
        return pd.DataFrame(
            columns=["gameid", "date", "game", "league", "region", "patch", "match", "featured"]
        )

    board = match_df.copy()
    board = board.sort_values("date", ascending=False).head(max_rows)

    board["game"] = "League of Legends"
    board["league"] = board.get("league", "").astype(str)
    board["region"] = board["league"]
    board["patch"] = board.get("patch", "").astype(str)
    board["match"] = board.get("blue_team_name", "").astype(str) + " vs " + board.get("red_team_name", "").astype(str)

    playoffs_col = pd.to_numeric(board.get("playoffs"), errors="coerce").fillna(0.0)
    major_col = board["league"].str.upper().isin(MAJOR_LEAGUES)
    board["featured"] = (playoffs_col > 0) | major_col

    keep_cols = [
        "gameid",
        "date",
        "game",
        "league",
        "region",
        "patch",
        "match",
        "featured",
        "blue_team_name",
        "red_team_name",
    ]
    return board[[c for c in keep_cols if c in board.columns]].copy()


def recent_template_rows(match_df: pd.DataFrame, max_rows: int = 300) -> pd.DataFrame:
    if match_df.empty:
        return pd.DataFrame()

    view = match_df.sort_values("date", ascending=False).head(max_rows).copy()
    view["template_label"] = (
        view["date"].dt.strftime("%Y-%m-%d")
        + " | "
        + view["league"].astype(str)
        + " | "
        + view["blue_team_name"].astype(str)
        + " vs "
        + view["red_team_name"].astype(str)
        + " | patch "
        + view["patch"].astype(str)
    )
    return view


def lookup_team_priors(
    team_history: pd.DataFrame,
    team_id: str,
    team_name: str,
) -> Tuple[Optional[float], Optional[float]]:
    if team_history.empty:
        return None, None

    subset = pd.DataFrame()
    normalized_id = normalize_team_id(team_id)
    normalized_name = clean_text(team_name).lower()

    if normalized_id:
        subset = team_history[team_history["team_id_norm"] == normalized_id]
    if subset.empty and normalized_name:
        subset = team_history[team_history["team_name_norm"] == normalized_name]
    if subset.empty:
        return None, None

    last_row = subset.iloc[-1]
    duration = pd.to_numeric(last_row["rolling_duration_prior_seconds"], errors="coerce")
    ckpm = pd.to_numeric(last_row["rolling_ckpm_prior"], errors="coerce")

    return (float(duration) if pd.notna(duration) else None, float(ckpm) if pd.notna(ckpm) else None)


def apply_template_to_defaults(template_row: pd.Series, defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = defaults.copy()
    out["league"] = clean_text(template_row.get("league")) or out["league"]
    out["patch"] = clean_text(template_row.get("patch")) or out["patch"]
    out["split"] = clean_text(template_row.get("split")) or out["split"]

    template_year = pd.to_numeric(template_row.get("year"), errors="coerce")
    if pd.notna(template_year):
        out["year"] = int(template_year)

    playoffs = pd.to_numeric(template_row.get("playoffs"), errors="coerce")
    if pd.notna(playoffs):
        out["playoffs"] = bool(int(playoffs))

    blue_first_pick = pd.to_numeric(template_row.get("blue_first_pick"), errors="coerce")
    if pd.notna(blue_first_pick):
        out["blue_first_pick"] = bool(int(blue_first_pick))

    out["blue_team_name"] = clean_text(template_row.get("blue_team_name")) or out["blue_team_name"]
    out["red_team_name"] = clean_text(template_row.get("red_team_name")) or out["red_team_name"]
    out["blue_team_id"] = clean_text(template_row.get("blue_team_id")) or out["blue_team_id"]
    out["red_team_id"] = clean_text(template_row.get("red_team_id")) or out["red_team_id"]

    for role in ROLE_ORDER:
        out[f"blue_role_{role}"] = clean_text(template_row.get(f"blue_{role}_champion")) or out[f"blue_role_{role}"]
        out[f"red_role_{role}"] = clean_text(template_row.get(f"red_{role}_champion")) or out[f"red_role_{role}"]

    for idx in range(1, 6):
        out[f"blue_pick_{idx}"] = clean_text(template_row.get(f"blue_pick{idx}"))
        out[f"red_pick_{idx}"] = clean_text(template_row.get(f"red_pick{idx}"))
        out[f"blue_ban_{idx}"] = clean_text(template_row.get(f"blue_ban{idx}"))
        out[f"red_ban_{idx}"] = clean_text(template_row.get(f"red_ban{idx}"))

    blue_dur = pd.to_numeric(template_row.get("blue_rolling_duration_prior_seconds"), errors="coerce")
    red_dur = pd.to_numeric(template_row.get("red_rolling_duration_prior_seconds"), errors="coerce")
    blue_ckpm = pd.to_numeric(template_row.get("blue_rolling_ckpm_prior"), errors="coerce")
    red_ckpm = pd.to_numeric(template_row.get("red_rolling_ckpm_prior"), errors="coerce")
    if pd.notna(blue_dur):
        out["blue_duration_prior"] = float(blue_dur)
    if pd.notna(red_dur):
        out["red_duration_prior"] = float(red_dur)
    if pd.notna(blue_ckpm):
        out["blue_ckpm_prior"] = float(blue_ckpm)
    if pd.notna(red_ckpm):
        out["red_ckpm_prior"] = float(red_ckpm)

    return out


def parse_role_quick_input(raw_text: str) -> Optional[List[str]]:
    tokens = [token.strip() for token in raw_text.split(",") if token.strip()]
    if len(tokens) != 5:
        return None
    return tokens


def default_form_state(options: Dict[str, Any], default_year: int) -> Dict[str, Any]:
    first = lambda values: values[0] if values else ""
    defaults: Dict[str, Any] = {
        "game": "League of Legends",
        "league": first(options["leagues"]),
        "patch": first(options["patches"]),
        "split": first(options["splits"]),
        "year": int(default_year),
        "playoffs": False,
        "blue_first_pick": True,
        "blue_team_name": first(options["team_names"]),
        "red_team_name": first(options["team_names"]),
        "blue_team_name_custom": "",
        "red_team_name_custom": "",
        "blue_team_id": first(options["team_ids"]),
        "red_team_id": first(options["team_ids"]),
        "blue_team_id_custom": "",
        "red_team_id_custom": "",
        "blue_duration_prior": float(options["duration_median"]),
        "red_duration_prior": float(options["duration_median"]),
        "blue_ckpm_prior": float(options["ckpm_median"]),
        "red_ckpm_prior": float(options["ckpm_median"]),
        "blue_roles_quick": "",
        "red_roles_quick": "",
    }
    for role in ROLE_ORDER:
        defaults[f"blue_role_{role}"] = ""
        defaults[f"red_role_{role}"] = ""
    for idx in range(1, 6):
        defaults[f"blue_pick_{idx}"] = ""
        defaults[f"red_pick_{idx}"] = ""
        defaults[f"blue_ban_{idx}"] = ""
        defaults[f"red_ban_{idx}"] = ""
    return defaults


def swap_form_sides() -> None:
    pairs = [
        ("blue_team_name", "red_team_name"),
        ("blue_team_name_custom", "red_team_name_custom"),
        ("blue_team_id", "red_team_id"),
        ("blue_team_id_custom", "red_team_id_custom"),
        ("blue_duration_prior", "red_duration_prior"),
        ("blue_ckpm_prior", "red_ckpm_prior"),
        ("blue_roles_quick", "red_roles_quick"),
    ]
    for role in ROLE_ORDER:
        pairs.append((f"blue_role_{role}", f"red_role_{role}"))
    for idx in range(1, 6):
        pairs.append((f"blue_pick_{idx}", f"red_pick_{idx}"))
        pairs.append((f"blue_ban_{idx}", f"red_ban_{idx}"))

    for left, right in pairs:
        st.session_state[left], st.session_state[right] = (
            st.session_state.get(right, ""),
            st.session_state.get(left, ""),
        )
    st.session_state["blue_first_pick"] = not bool(st.session_state.get("blue_first_pick", True))


def build_payload(form_data: Dict[str, Any]) -> Dict[str, Any]:
    blue_roles = form_data["blue_roles"]
    red_roles = form_data["red_roles"]

    def fallback_pick(raw_pick: str, role_name: str, role_map: Dict[str, str]) -> Optional[str]:
        value = clean_text(raw_pick)
        if value:
            return value
        role_value = clean_text(role_map.get(role_name))
        return role_value or None

    blue_picks = [
        fallback_pick(form_data["blue_picks"][idx], ROLE_ORDER[idx], blue_roles)
        for idx in range(5)
    ]
    red_picks = [
        fallback_pick(form_data["red_picks"][idx], ROLE_ORDER[idx], red_roles)
        for idx in range(5)
    ]

    blue_bans = [clean_text(v) or None for v in form_data["blue_bans"]]
    red_bans = [clean_text(v) or None for v in form_data["red_bans"]]

    blue_champions = [c for c in [clean_text(blue_roles[r]) for r in ROLE_ORDER] if c]
    red_champions = [c for c in [clean_text(red_roles[r]) for r in ROLE_ORDER] if c]

    if len(blue_champions) < 5:
        blue_champions = [c for c in blue_picks if clean_text(c)]
    if len(red_champions) < 5:
        red_champions = [c for c in red_picks if clean_text(c)]

    payload = {
        "league": clean_text(form_data["league"]),
        "split": clean_text(form_data["split"]),
        "patch": clean_text(form_data["patch"]),
        "year": int(form_data["year"]),
        "playoffs": int(form_data["playoffs"]),
        "blue_team_id": clean_text(form_data["blue_team_id"]) or None,
        "red_team_id": clean_text(form_data["red_team_id"]) or None,
        "blue_team_name": clean_text(form_data["blue_team_name"]) or None,
        "red_team_name": clean_text(form_data["red_team_name"]) or None,
        "blue_first_pick": int(form_data["blue_first_pick"]),
        "red_first_pick": int(1 - form_data["blue_first_pick"]),
        "blue_rolling_duration_prior_seconds": float(form_data["blue_duration_prior"]),
        "red_rolling_duration_prior_seconds": float(form_data["red_duration_prior"]),
        "rolling_duration_prior_diff_seconds": float(
            form_data["blue_duration_prior"] - form_data["red_duration_prior"]
        ),
        "blue_rolling_ckpm_prior": float(form_data["blue_ckpm_prior"]),
        "red_rolling_ckpm_prior": float(form_data["red_ckpm_prior"]),
        "rolling_ckpm_prior_diff": float(form_data["blue_ckpm_prior"] - form_data["red_ckpm_prior"]),
        "blue_draft_champions": blue_champions,
        "red_draft_champions": red_champions,
    }

    for role in ROLE_ORDER:
        payload[f"blue_{role}_champion"] = clean_text(blue_roles.get(role)) or None
        payload[f"red_{role}_champion"] = clean_text(red_roles.get(role)) or None

    for idx in range(5):
        payload[f"blue_pick{idx + 1}"] = blue_picks[idx]
        payload[f"red_pick{idx + 1}"] = red_picks[idx]
        payload[f"blue_ban{idx + 1}"] = blue_bans[idx]
        payload[f"red_ban{idx + 1}"] = red_bans[idx]

    return payload


def validate_draft_inputs(blue_roles: Dict[str, str], red_roles: Dict[str, str]) -> List[str]:
    errors: List[str] = []
    blue_list = [clean_text(blue_roles.get(role, "")) for role in ROLE_ORDER]
    red_list = [clean_text(red_roles.get(role, "")) for role in ROLE_ORDER]

    if any(not champ for champ in blue_list):
        errors.append("Blue side needs all 5 role champions.")
    if any(not champ for champ in red_list):
        errors.append("Red side needs all 5 role champions.")

    if len([c for c in blue_list if c]) == 5 and len(set(blue_list)) != 5:
        errors.append("Blue side has duplicate champions.")
    if len([c for c in red_list if c]) == 5 and len(set(red_list)) != 5:
        errors.append("Red side has duplicate champions.")

    overlap = set([c for c in blue_list if c]).intersection(set([c for c in red_list if c]))
    if overlap:
        errors.append("Champion overlap across teams: " + ", ".join(sorted(overlap)))

    return errors
