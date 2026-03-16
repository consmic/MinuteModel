from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from minutemodel.inference import load_artifacts, predict_single_draft

ROLE_ORDER = ["top", "jng", "mid", "bot", "sup"]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none"}:
        return ""
    return text


def _parse_champion_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [_clean_text(v) for v in value if _clean_text(v)]
    text = _clean_text(value)
    if not text:
        return []
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].replace("'", "").replace('"', "")
        return [token.strip() for token in inner.split(",") if token.strip()]
    return [text]


def _normalize_team_id(value: Any) -> str:
    text = _clean_text(value)
    if not text:
        return ""
    try:
        as_float = float(text)
        if as_float.is_integer():
            return str(int(as_float))
    except Exception:
        pass
    return text


def _options_with_current(options: List[str], current: str) -> List[str]:
    clean_options = [opt for opt in sorted({_clean_text(v) for v in options}) if opt]
    current_clean = _clean_text(current)
    if current_clean and current_clean not in clean_options:
        clean_options = [current_clean] + clean_options
    return [""] + clean_options


@st.cache_resource
def _load_artifacts_cached(path: str) -> Dict[str, Any]:
    return load_artifacts(path)


@st.cache_data
def _load_match_table(path: str) -> pd.DataFrame:
    table_path = Path(path)
    if not table_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(table_path, low_memory=False)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
    return df


@st.cache_data
def _load_test_mae_minutes(metrics_path: str) -> Optional[float]:
    path = Path(metrics_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("metrics_by_model", {}).get("lightgbm", {}).get("mae_minutes")


def _extract_ui_options(match_df: pd.DataFrame) -> Dict[str, Any]:
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
                tokens.extend([_clean_text(v) for v in match_df[col].dropna().tolist()])
        out = sorted({tok for tok in tokens if tok})
        return out

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
            champions.extend([_clean_text(v) for v in match_df[col].dropna().tolist()])

    for list_col in ["blue_draft_champions", "red_draft_champions"]:
        if list_col in match_df.columns:
            for value in match_df[list_col].dropna().tolist():
                champions.extend(_parse_champion_list(value))

    duration_series = pd.to_numeric(
        match_df.get("target_gamelength_seconds", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    ckpm_values = pd.concat(
        [
            pd.to_numeric(match_df.get("blue_rolling_ckpm_prior", pd.Series(dtype=float)), errors="coerce"),
            pd.to_numeric(match_df.get("red_rolling_ckpm_prior", pd.Series(dtype=float)), errors="coerce"),
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


def _team_history_view(match_df: pd.DataFrame) -> pd.DataFrame:
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
    out["team_id_norm"] = out["team_id"].map(_normalize_team_id)
    out["team_name_norm"] = out["team_name"].map(lambda x: _clean_text(x).lower())
    out = out.sort_values("date")
    return out


def _recent_template_rows(match_df: pd.DataFrame, max_rows: int = 300) -> pd.DataFrame:
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


def _lookup_team_priors(
    team_history: pd.DataFrame,
    team_id: str,
    team_name: str,
) -> Tuple[Optional[float], Optional[float]]:
    if team_history.empty:
        return None, None

    subset = pd.DataFrame()
    normalized_id = _normalize_team_id(team_id)
    normalized_name = _clean_text(team_name).lower()

    if normalized_id:
        subset = team_history[team_history["team_id_norm"] == normalized_id]
    if subset.empty and normalized_name:
        subset = team_history[team_history["team_name_norm"] == normalized_name]
    if subset.empty:
        return None, None

    last_row = subset.iloc[-1]
    duration = pd.to_numeric(last_row["rolling_duration_prior_seconds"], errors="coerce")
    ckpm = pd.to_numeric(last_row["rolling_ckpm_prior"], errors="coerce")

    duration_out = float(duration) if pd.notna(duration) else None
    ckpm_out = float(ckpm) if pd.notna(ckpm) else None
    return duration_out, ckpm_out


def _apply_template_to_defaults(
    template_row: pd.Series,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    out = defaults.copy()
    out["league"] = _clean_text(template_row.get("league")) or out["league"]
    out["patch"] = _clean_text(template_row.get("patch")) or out["patch"]
    out["split"] = _clean_text(template_row.get("split")) or out["split"]

    template_year = pd.to_numeric(template_row.get("year"), errors="coerce")
    if pd.notna(template_year):
        out["year"] = int(template_year)

    playoffs = pd.to_numeric(template_row.get("playoffs"), errors="coerce")
    if pd.notna(playoffs):
        out["playoffs"] = bool(int(playoffs))

    blue_first_pick = pd.to_numeric(template_row.get("blue_first_pick"), errors="coerce")
    if pd.notna(blue_first_pick):
        out["blue_first_pick"] = bool(int(blue_first_pick))

    out["blue_team_name"] = _clean_text(template_row.get("blue_team_name")) or out["blue_team_name"]
    out["red_team_name"] = _clean_text(template_row.get("red_team_name")) or out["red_team_name"]
    out["blue_team_id"] = _clean_text(template_row.get("blue_team_id")) or out["blue_team_id"]
    out["red_team_id"] = _clean_text(template_row.get("red_team_id")) or out["red_team_id"]

    for role in ROLE_ORDER:
        out[f"blue_role_{role}"] = _clean_text(template_row.get(f"blue_{role}_champion")) or out[f"blue_role_{role}"]
        out[f"red_role_{role}"] = _clean_text(template_row.get(f"red_{role}_champion")) or out[f"red_role_{role}"]

    for idx in range(1, 6):
        out[f"blue_pick_{idx}"] = _clean_text(template_row.get(f"blue_pick{idx}"))
        out[f"red_pick_{idx}"] = _clean_text(template_row.get(f"red_pick{idx}"))
        out[f"blue_ban_{idx}"] = _clean_text(template_row.get(f"blue_ban{idx}"))
        out[f"red_ban_{idx}"] = _clean_text(template_row.get(f"red_ban{idx}"))

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


def _parse_role_quick_input(raw_text: str) -> Optional[List[str]]:
    tokens = [token.strip() for token in raw_text.split(",") if token.strip()]
    if len(tokens) != 5:
        return None
    return tokens


def _default_form_state(options: Dict[str, Any], default_year: int) -> Dict[str, Any]:
    first = lambda values: values[0] if values else ""
    defaults: Dict[str, Any] = {
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


def _swap_form_sides() -> None:
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
        st.session_state[left], st.session_state[right] = st.session_state.get(right, ""), st.session_state.get(left, "")
    st.session_state["blue_first_pick"] = not bool(st.session_state.get("blue_first_pick", True))


def _build_payload(form_data: Dict[str, Any]) -> Dict[str, Any]:
    blue_roles = form_data["blue_roles"]
    red_roles = form_data["red_roles"]

    def fallback_pick(raw_pick: str, role_name: str, role_map: Dict[str, str]) -> Optional[str]:
        value = _clean_text(raw_pick)
        if value:
            return value
        role_value = _clean_text(role_map.get(role_name))
        return role_value or None

    blue_picks = [
        fallback_pick(form_data["blue_picks"][idx], ROLE_ORDER[idx], blue_roles)
        for idx in range(5)
    ]
    red_picks = [
        fallback_pick(form_data["red_picks"][idx], ROLE_ORDER[idx], red_roles)
        for idx in range(5)
    ]

    blue_bans = [_clean_text(v) or None for v in form_data["blue_bans"]]
    red_bans = [_clean_text(v) or None for v in form_data["red_bans"]]

    blue_champions = [c for c in [_clean_text(blue_roles[r]) for r in ROLE_ORDER] if c]
    red_champions = [c for c in [_clean_text(red_roles[r]) for r in ROLE_ORDER] if c]

    if len(blue_champions) < 5:
        blue_champions = [c for c in blue_picks if _clean_text(c)]
    if len(red_champions) < 5:
        red_champions = [c for c in red_picks if _clean_text(c)]

    payload = {
        "league": _clean_text(form_data["league"]),
        "split": _clean_text(form_data["split"]),
        "patch": _clean_text(form_data["patch"]),
        "year": int(form_data["year"]),
        "playoffs": int(form_data["playoffs"]),
        "blue_team_id": _clean_text(form_data["blue_team_id"]) or None,
        "red_team_id": _clean_text(form_data["red_team_id"]) or None,
        "blue_team_name": _clean_text(form_data["blue_team_name"]) or None,
        "red_team_name": _clean_text(form_data["red_team_name"]) or None,
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
        payload[f"blue_{role}_champion"] = _clean_text(blue_roles.get(role)) or None
        payload[f"red_{role}_champion"] = _clean_text(red_roles.get(role)) or None

    for idx in range(5):
        payload[f"blue_pick{idx + 1}"] = blue_picks[idx]
        payload[f"red_pick{idx + 1}"] = red_picks[idx]
        payload[f"blue_ban{idx + 1}"] = blue_bans[idx]
        payload[f"red_ban{idx + 1}"] = red_bans[idx]

    return payload


def _validate_draft_inputs(blue_roles: Dict[str, str], red_roles: Dict[str, str]) -> List[str]:
    errors: List[str] = []
    blue_list = [_clean_text(blue_roles.get(role, "")) for role in ROLE_ORDER]
    red_list = [_clean_text(red_roles.get(role, "")) for role in ROLE_ORDER]

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


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
        html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; color: #0f172a; }
        .stApp {
            background:
                radial-gradient(circle at 8% 12%, rgba(244, 167, 0, 0.20), transparent 40%),
                radial-gradient(circle at 96% 2%, rgba(30, 136, 229, 0.16), transparent 44%),
                linear-gradient(180deg, #f8fafc 0%, #eef4fb 100%);
        }
        .block-container { padding-top: 1.2rem; padding-bottom: 1.4rem; max-width: 1220px; }
        .hero {
            border: 1px solid rgba(15, 23, 42, 0.15);
            background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(255,255,255,0.82));
            border-radius: 20px;
            padding: 1.05rem 1.2rem;
            margin-bottom: 1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
        }
        .section-card {
            border-radius: 16px;
            border: 1px solid rgba(15, 23, 42, 0.13);
            background: rgba(255, 255, 255, 0.88);
            padding: 0.8rem 1rem;
            margin-bottom: 0.75rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="MinuteModel Predictor",
        page_icon=":hourglass_flowing_sand:",
        layout="wide",
    )
    _inject_styles()

    st.markdown(
        """
        <div class="hero">
        <div style="font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; color:#1d4ed8; font-weight:700;">Draft-Only Predictor</div>
        <h2 style="margin:0.1rem 0 0 0;">MinuteModel Match Duration Predictor</h2>
        <p style="margin:0.4rem 0 0 0;">
        Build a clean pre-game draft context and get a fast duration estimate for upcoming pro LoL matches.
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Model Files")
        artifact_path = st.text_input(
            "Artifact path",
            value="artifacts/model_artifacts.joblib",
        )
        match_table_path = st.text_input(
            "Match table path",
            value="artifacts/match_level_table.csv",
        )
        metrics_path = st.text_input(
            "Metrics summary path",
            value="reports/metrics_summary.json",
        )
        include_explanation = st.checkbox("Include SHAP explanation", value=True)

    artifact_file = Path(artifact_path)
    if not artifact_file.exists():
        st.error(f"Artifact file not found: {artifact_file}")
        st.stop()

    try:
        artifacts = _load_artifacts_cached(str(artifact_file))
    except Exception as exc:
        st.error(f"Failed to load artifacts: {exc}")
        st.stop()

    config = artifacts.get("config", {})
    match_df = _load_match_table(str(match_table_path))
    options = _extract_ui_options(match_df)
    team_history = _team_history_view(match_df)
    test_mae_minutes = _load_test_mae_minutes(metrics_path)

    default_year = int(pd.Timestamp.now(tz="UTC").year)
    if not match_df.empty and "date" in match_df.columns and match_df["date"].notna().any():
        default_year = int(match_df["date"].dropna().max().year)

    defaults = _default_form_state(options, default_year=default_year)
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    template_rows = _recent_template_rows(match_df)

    with st.sidebar:
        st.markdown("---")
        st.caption(
            "Model config: "
            f"role_specific={config.get('use_role_specific_draft_features', True)} | "
            f"bag_fallback={config.get('use_bag_of_champions_fallback', True)} | "
            f"target_unit={config.get('target_unit', 'seconds')}"
        )
        if test_mae_minutes is not None:
            st.caption(f"Historical test MAE: {test_mae_minutes:.2f} min")

        if not template_rows.empty:
            selected_template_label = st.selectbox(
                "Prefill from recent match",
                options=template_rows["template_label"].tolist(),
                index=0,
            )
            col_t1, col_t2 = st.columns(2)
            if col_t1.button("Load Template", use_container_width=True):
                row = template_rows[template_rows["template_label"] == selected_template_label].iloc[0]
                updated = _apply_template_to_defaults(row, defaults)
                for key, value in updated.items():
                    st.session_state[key] = value
                st.rerun()
            if col_t2.button("Swap Sides", use_container_width=True):
                _swap_form_sides()
                st.rerun()

        if st.button("Reset Form", use_container_width=True):
            for key, value in defaults.items():
                st.session_state[key] = value
            st.rerun()

    filled_roles = sum(
        bool(_clean_text(st.session_state.get(f"{side}_role_{role}", "")))
        for side in ["blue", "red"]
        for role in ROLE_ORDER
    )
    st.progress(filled_roles / 10.0, text=f"Draft completion: {filled_roles}/10 champions")

    with st.form("prediction_form"):
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 1) Match Metadata")
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)

        league = meta_col1.selectbox(
            "League",
            options=_options_with_current(options["leagues"], st.session_state.get("league", "")),
            key="league",
        )
        patch = meta_col2.selectbox(
            "Patch",
            options=_options_with_current(options["patches"], st.session_state.get("patch", "")),
            key="patch",
        )
        split = meta_col3.selectbox(
            "Split",
            options=_options_with_current(options["splits"], st.session_state.get("split", "")),
            key="split",
        )
        year = meta_col4.number_input(
            "Year",
            min_value=2020,
            max_value=2035,
            step=1,
            key="year",
        )
        playoffs = st.toggle("Playoffs match", key="playoffs")
        blue_first_pick = st.toggle("Blue side has first pick", key="blue_first_pick")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 2) Teams")
        team_col1, team_col2 = st.columns(2)
        with team_col1:
            st.markdown("#### Blue Side")
            blue_team_name = st.selectbox(
                "Blue team name",
                options=_options_with_current(options["team_names"], st.session_state.get("blue_team_name", "")),
                key="blue_team_name",
            )
            blue_team_name_custom = st.text_input("Blue team name override (optional)", key="blue_team_name_custom")
            blue_team_id = st.selectbox(
                "Blue team ID",
                options=_options_with_current(options["team_ids"], st.session_state.get("blue_team_id", "")),
                key="blue_team_id",
            )
            blue_team_id_custom = st.text_input("Blue team ID override (optional)", key="blue_team_id_custom")

        with team_col2:
            st.markdown("#### Red Side")
            red_team_name = st.selectbox(
                "Red team name",
                options=_options_with_current(options["team_names"], st.session_state.get("red_team_name", "")),
                key="red_team_name",
            )
            red_team_name_custom = st.text_input("Red team name override (optional)", key="red_team_name_custom")
            red_team_id = st.selectbox(
                "Red team ID",
                options=_options_with_current(options["team_ids"], st.session_state.get("red_team_id", "")),
                key="red_team_id",
            )
            red_team_id_custom = st.text_input("Red team ID override (optional)", key="red_team_id_custom")

        blue_team_name_final = _clean_text(blue_team_name_custom) or _clean_text(blue_team_name)
        red_team_name_final = _clean_text(red_team_name_custom) or _clean_text(red_team_name)
        blue_team_id_final = _clean_text(blue_team_id_custom) or _clean_text(blue_team_id)
        red_team_id_final = _clean_text(red_team_id_custom) or _clean_text(red_team_id)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 3) Historical Priors")
        auto_fill_priors = st.checkbox("Auto-fill priors from latest team history", value=True)

        if auto_fill_priors and not team_history.empty:
            blue_duration_auto, blue_ckpm_auto = _lookup_team_priors(
                team_history,
                team_id=blue_team_id_final,
                team_name=blue_team_name_final,
            )
            red_duration_auto, red_ckpm_auto = _lookup_team_priors(
                team_history,
                team_id=red_team_id_final,
                team_name=red_team_name_final,
            )
            if blue_duration_auto is not None:
                st.session_state["blue_duration_prior"] = float(blue_duration_auto)
            if red_duration_auto is not None:
                st.session_state["red_duration_prior"] = float(red_duration_auto)
            if blue_ckpm_auto is not None:
                st.session_state["blue_ckpm_prior"] = float(blue_ckpm_auto)
            if red_ckpm_auto is not None:
                st.session_state["red_ckpm_prior"] = float(red_ckpm_auto)

        prior_col1, prior_col2 = st.columns(2)
        with prior_col1:
            blue_duration_prior = st.number_input(
                "Blue rolling duration prior (seconds)",
                step=10.0,
                key="blue_duration_prior",
            )
            blue_ckpm_prior = st.number_input(
                "Blue rolling CKPM prior",
                step=0.01,
                key="blue_ckpm_prior",
            )
        with prior_col2:
            red_duration_prior = st.number_input(
                "Red rolling duration prior (seconds)",
                step=10.0,
                key="red_duration_prior",
            )
            red_ckpm_prior = st.number_input(
                "Red rolling CKPM prior",
                step=0.01,
                key="red_ckpm_prior",
            )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown("### 4) Draft Champions by Role")
        quick_col1, quick_col2 = st.columns(2)
        with quick_col1:
            blue_quick_roles = st.text_input(
                "Blue quick input (top,jng,mid,bot,sup)",
                key="blue_roles_quick",
            )
            parsed_blue = _parse_role_quick_input(blue_quick_roles)
            if parsed_blue:
                for idx, role in enumerate(ROLE_ORDER):
                    st.session_state[f"blue_role_{role}"] = parsed_blue[idx]
            elif _clean_text(blue_quick_roles):
                st.caption("Blue quick input needs exactly 5 comma-separated champions.")
        with quick_col2:
            red_quick_roles = st.text_input(
                "Red quick input (top,jng,mid,bot,sup)",
                key="red_roles_quick",
            )
            parsed_red = _parse_role_quick_input(red_quick_roles)
            if parsed_red:
                for idx, role in enumerate(ROLE_ORDER):
                    st.session_state[f"red_role_{role}"] = parsed_red[idx]
            elif _clean_text(red_quick_roles):
                st.caption("Red quick input needs exactly 5 comma-separated champions.")

        champ_suggestions = options["champions"][:]
        if not champ_suggestions:
            champ_suggestions = ["Aatrox", "Lee Sin", "Orianna", "Jinx", "Nautilus"]

        blue_roles: Dict[str, str] = {}
        red_roles: Dict[str, str] = {}
        role_col1, role_col2 = st.columns(2)
        with role_col1:
            st.markdown("#### Blue Roles")
            for role in ROLE_ORDER:
                blue_roles[role] = st.selectbox(
                    f"Blue {role.upper()}",
                    options=_options_with_current(champ_suggestions, st.session_state.get(f"blue_role_{role}", "")),
                    key=f"blue_role_{role}",
                )
        with role_col2:
            st.markdown("#### Red Roles")
            for role in ROLE_ORDER:
                red_roles[role] = st.selectbox(
                    f"Red {role.upper()}",
                    options=_options_with_current(champ_suggestions, st.session_state.get(f"red_role_{role}", "")),
                    key=f"red_role_{role}",
                )

        with st.expander("Optional: Pick/Ban Overrides"):
            st.caption("If left blank, picks default to role champions.")
            blue_pick_inputs: List[str] = []
            red_pick_inputs: List[str] = []
            blue_ban_inputs: List[str] = []
            red_ban_inputs: List[str] = []

            pick_col1, pick_col2 = st.columns(2)
            with pick_col1:
                st.markdown("#### Blue Picks / Bans")
                for idx in range(1, 6):
                    blue_pick_inputs.append(st.text_input(f"Blue pick{idx}", key=f"blue_pick_{idx}"))
                for idx in range(1, 6):
                    blue_ban_inputs.append(st.text_input(f"Blue ban{idx}", key=f"blue_ban_{idx}"))
            with pick_col2:
                st.markdown("#### Red Picks / Bans")
                for idx in range(1, 6):
                    red_pick_inputs.append(st.text_input(f"Red pick{idx}", key=f"red_pick_{idx}"))
                for idx in range(1, 6):
                    red_ban_inputs.append(st.text_input(f"Red ban{idx}", key=f"red_ban_{idx}"))

        st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button("Predict Match Duration")

    if not submitted:
        st.info("Fill in draft details and click Predict Match Duration.")
        st.stop()

    validation_errors = _validate_draft_inputs(blue_roles, red_roles)
    if not blue_team_name_final:
        validation_errors.append("Blue team name is required.")
    if not red_team_name_final:
        validation_errors.append("Red team name is required.")
    if blue_duration_prior <= 0 or red_duration_prior <= 0:
        validation_errors.append("Duration priors must be positive.")
    if blue_ckpm_prior <= 0 or red_ckpm_prior <= 0:
        validation_errors.append("CKPM priors must be positive.")

    if validation_errors:
        for err in validation_errors:
            st.error(err)
        st.stop()

    form_data = {
        "league": league,
        "patch": patch,
        "split": split,
        "year": year,
        "playoffs": playoffs,
        "blue_first_pick": blue_first_pick,
        "blue_team_name": blue_team_name_final,
        "red_team_name": red_team_name_final,
        "blue_team_id": blue_team_id_final,
        "red_team_id": red_team_id_final,
        "blue_duration_prior": blue_duration_prior,
        "red_duration_prior": red_duration_prior,
        "blue_ckpm_prior": blue_ckpm_prior,
        "red_ckpm_prior": red_ckpm_prior,
        "blue_roles": blue_roles,
        "red_roles": red_roles,
        "blue_picks": blue_pick_inputs if "blue_pick_inputs" in locals() else [""] * 5,
        "red_picks": red_pick_inputs if "red_pick_inputs" in locals() else [""] * 5,
        "blue_bans": blue_ban_inputs if "blue_ban_inputs" in locals() else [""] * 5,
        "red_bans": red_ban_inputs if "red_ban_inputs" in locals() else [""] * 5,
    }
    payload = _build_payload(form_data)

    try:
        prediction = predict_single_draft(
            artifact_path=artifact_file,
            draft_payload=payload,
            include_explanation=include_explanation,
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.stop()

    pred_minutes = float(prediction["predicted_duration_minutes"])
    pred_seconds = float(prediction["predicted_duration_seconds"])
    total_seconds = max(int(round(pred_seconds)), 0)
    predicted_clock = f"{total_seconds // 60}:{total_seconds % 60:02d}"
    mae_context = float(test_mae_minutes) if test_mae_minutes is not None else 4.2
    lower_bound = max(pred_minutes - mae_context, 0.0)
    upper_bound = pred_minutes + mae_context

    st.markdown("### Prediction")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Predicted Duration", f"{pred_minutes:.2f} min")
    metric_col2.metric("Predicted Seconds", f"{pred_seconds:.0f}s")
    metric_col3.metric("Estimated Clock", predicted_clock)
    st.caption(f"Typical historical error is about +/- {mae_context:.2f} minutes.")
    st.info(f"Practical range for this match: roughly {lower_bound:.1f} to {upper_bound:.1f} minutes.")

    with st.expander("Payload used for inference"):
        st.json(payload)

    explanation = prediction.get("explanation")
    if explanation:
        st.markdown("### Explanation")
        if explanation.get("warning"):
            st.warning(explanation["warning"])
        contribs = explanation.get("top_feature_contributions", [])
        if contribs:
            contrib_df = pd.DataFrame(contribs)
            st.dataframe(contrib_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
