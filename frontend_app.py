from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');
        html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
        .stApp {
            background:
                radial-gradient(circle at 10% 10%, rgba(245, 205, 84, 0.16), transparent 45%),
                radial-gradient(circle at 90% 0%, rgba(72, 176, 255, 0.14), transparent 50%),
                linear-gradient(180deg, #f5f7fa 0%, #edf2f7 100%);
        }
        .block-container { padding-top: 1.4rem; padding-bottom: 1.4rem; max-width: 1200px; }
        .hero {
            border: 1px solid rgba(30, 41, 59, 0.14);
            background: rgba(255, 255, 255, 0.8);
            border-radius: 18px;
            padding: 1rem 1.2rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            border-radius: 14px;
            border: 1px solid rgba(30, 41, 59, 0.16);
            background: #ffffff;
            padding: 0.9rem 1rem;
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
        <h2 style="margin:0;">MinuteModel Match Duration Predictor</h2>
        <p style="margin:0.4rem 0 0 0;">
        Draft-only prediction for upcoming pro LoL games, powered by your trained LightGBM artifact.
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
            "Match table path (optional)",
            value="artifacts/match_level_table.csv",
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
    st.caption(
        "Model config: "
        f"role_specific={config.get('use_role_specific_draft_features', True)}, "
        f"bag_fallback={config.get('use_bag_of_champions_fallback', True)}, "
        f"target_unit={config.get('target_unit', 'seconds')}"
    )

    match_df = _load_match_table(str(match_table_path))
    options = _extract_ui_options(match_df)
    team_history = _team_history_view(match_df)

    default_year = int(pd.Timestamp.utcnow().year)
    if not match_df.empty and "date" in match_df.columns and match_df["date"].notna().any():
        default_year = int(match_df["date"].dropna().max().year)

    with st.form("prediction_form"):
        st.markdown("### Match Metadata")
        meta_col1, meta_col2, meta_col3, meta_col4 = st.columns(4)

        league = meta_col1.selectbox(
            "League",
            options=options["leagues"] if options["leagues"] else [""],
            index=0,
        )
        patch = meta_col2.selectbox(
            "Patch",
            options=options["patches"] if options["patches"] else [""],
            index=0,
        )
        split = meta_col3.selectbox(
            "Split",
            options=options["splits"] if options["splits"] else [""],
            index=0,
        )
        year = meta_col4.number_input("Year", min_value=2020, max_value=2035, value=default_year, step=1)

        playoffs = st.toggle("Playoffs match", value=False)
        blue_first_pick = st.toggle("Blue side has first pick", value=True)

        st.markdown("### Teams")
        team_col1, team_col2 = st.columns(2)
        with team_col1:
            st.markdown("#### Blue Side")
            blue_team_name = st.selectbox(
                "Blue team name",
                options=options["team_names"] if options["team_names"] else [""],
                index=0,
            )
            blue_team_name_custom = st.text_input("Blue team name override (optional)", value="")
            blue_team_id = st.selectbox(
                "Blue team ID",
                options=options["team_ids"] if options["team_ids"] else [""],
                index=0,
            )
            blue_team_id_custom = st.text_input("Blue team ID override (optional)", value="")

        with team_col2:
            st.markdown("#### Red Side")
            red_team_name = st.selectbox(
                "Red team name",
                options=options["team_names"] if options["team_names"] else [""],
                index=0,
            )
            red_team_name_custom = st.text_input("Red team name override (optional)", value="")
            red_team_id = st.selectbox(
                "Red team ID",
                options=options["team_ids"] if options["team_ids"] else [""],
                index=0,
            )
            red_team_id_custom = st.text_input("Red team ID override (optional)", value="")

        blue_team_name_final = _clean_text(blue_team_name_custom) or _clean_text(blue_team_name)
        red_team_name_final = _clean_text(red_team_name_custom) or _clean_text(red_team_name)
        blue_team_id_final = _clean_text(blue_team_id_custom) or _clean_text(blue_team_id)
        red_team_id_final = _clean_text(red_team_id_custom) or _clean_text(red_team_id)

        st.markdown("### Historical Priors")
        auto_fill_priors = st.checkbox("Auto-fill priors from latest team history", value=True)
        blue_duration_prior_default = options["duration_median"]
        red_duration_prior_default = options["duration_median"]
        blue_ckpm_prior_default = options["ckpm_median"]
        red_ckpm_prior_default = options["ckpm_median"]

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
                blue_duration_prior_default = blue_duration_auto
            if red_duration_auto is not None:
                red_duration_prior_default = red_duration_auto
            if blue_ckpm_auto is not None:
                blue_ckpm_prior_default = blue_ckpm_auto
            if red_ckpm_auto is not None:
                red_ckpm_prior_default = red_ckpm_auto

        prior_col1, prior_col2 = st.columns(2)
        with prior_col1:
            blue_duration_prior = st.number_input(
                "Blue rolling duration prior (seconds)",
                value=float(blue_duration_prior_default),
                step=10.0,
            )
            blue_ckpm_prior = st.number_input(
                "Blue rolling CKPM prior",
                value=float(blue_ckpm_prior_default),
                step=0.01,
            )
        with prior_col2:
            red_duration_prior = st.number_input(
                "Red rolling duration prior (seconds)",
                value=float(red_duration_prior_default),
                step=10.0,
            )
            red_ckpm_prior = st.number_input(
                "Red rolling CKPM prior",
                value=float(red_ckpm_prior_default),
                step=0.01,
            )

        st.markdown("### Draft Champions by Role")
        champ_suggestions = options["champions"][:]
        if not champ_suggestions:
            champ_suggestions = ["", "Aatrox", "Lee Sin", "Orianna", "Jinx", "Nautilus"]

        blue_roles: Dict[str, str] = {}
        red_roles: Dict[str, str] = {}
        role_col1, role_col2 = st.columns(2)
        with role_col1:
            st.markdown("#### Blue Roles")
            for role in ROLE_ORDER:
                blue_roles[role] = st.selectbox(
                    f"Blue {role.upper()}",
                    options=champ_suggestions,
                    index=0,
                    key=f"blue_role_{role}",
                )
        with role_col2:
            st.markdown("#### Red Roles")
            for role in ROLE_ORDER:
                red_roles[role] = st.selectbox(
                    f"Red {role.upper()}",
                    options=champ_suggestions,
                    index=0,
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
                    blue_pick_inputs.append(st.text_input(f"Blue pick{idx}", value="", key=f"blue_pick_{idx}"))
                for idx in range(1, 6):
                    blue_ban_inputs.append(st.text_input(f"Blue ban{idx}", value="", key=f"blue_ban_{idx}"))
            with pick_col2:
                st.markdown("#### Red Picks / Bans")
                for idx in range(1, 6):
                    red_pick_inputs.append(st.text_input(f"Red pick{idx}", value="", key=f"red_pick_{idx}"))
                for idx in range(1, 6):
                    red_ban_inputs.append(st.text_input(f"Red ban{idx}", value="", key=f"red_ban_{idx}"))

        submitted = st.form_submit_button("Predict Match Duration")

    if not submitted:
        st.info("Fill in draft details and click Predict Match Duration.")
        st.stop()

    if not all(_clean_text(blue_roles[r]) for r in ROLE_ORDER):
        st.error("Blue side needs all 5 role champions.")
        st.stop()
    if not all(_clean_text(red_roles[r]) for r in ROLE_ORDER):
        st.error("Red side needs all 5 role champions.")
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
    predicted_clock = f"{int(pred_minutes)}:{int(round((pred_minutes % 1) * 60)):02d}"

    st.markdown("### Prediction")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Predicted Duration", f"{pred_minutes:.2f} min")
    metric_col2.metric("Predicted Seconds", f"{pred_seconds:.0f}s")
    metric_col3.metric("Approx Clock", predicted_clock)

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
