from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from frontend_ui import (
    FREE_PREDICTIONS_PER_MONTH,
    ROLE_ORDER,
    apply_app_styles,
    apply_template_to_defaults,
    append_journal_entry,
    build_calendar_board,
    build_payload,
    clean_text,
    compute_journal_metrics,
    default_form_state,
    ensure_form_state,
    ensure_usage_state,
    extract_ui_options,
    get_usage_snapshot,
    increment_usage,
    load_artifacts_cached,
    load_journal,
    load_match_table,
    load_metrics_payload,
    load_test_mae_minutes,
    lookup_team_priors,
    options_with_current,
    parse_role_quick_input,
    recent_template_rows,
    render_duration_prediction_card,
    render_empty_state,
    render_info_panel,
    render_page_intro,
    render_section_heading,
    reset_usage,
    save_journal,
    status_label,
    swap_form_sides,
    team_history_view,
    validate_draft_inputs,
)
from minutemodel.inference import predict_single_draft


NAV_ITEMS = [
    "Home",
    "Predictions",
    "Calendar",
    "Journal",
    "Model Performance",
    "Account / Usage",
]

SUPPORTED_GAMES = [
    "League of Legends",
    "Counter-Strike 2",
    "Dota 2",
    "VALORANT",
]

DEFAULT_ARTIFACT_PATH = "artifacts/model_artifacts.joblib"
DEFAULT_MATCH_TABLE_PATH = "artifacts/match_level_table.csv"
DEFAULT_METRICS_PATH = "reports/metrics_summary.json"
DEFAULT_JOURNAL_PATH = "reports/prediction_journal.csv"


@dataclass
class AppContext:
    artifact_path: Path
    match_table_path: Path
    metrics_path: Path
    journal_path: Path
    include_explanation: bool
    artifacts: Optional[Dict[str, Any]]
    artifact_error: Optional[str]
    config: Dict[str, Any]
    match_df: pd.DataFrame
    options: Dict[str, Any]
    team_history: pd.DataFrame
    template_rows: pd.DataFrame
    calendar_board: pd.DataFrame
    metrics_payload: Dict[str, Any]
    test_mae_minutes: Optional[float]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    cast = pd.to_numeric(value, errors="coerce")
    return float(cast) if pd.notna(cast) else float(default)


def _clock_from_seconds(seconds: float) -> str:
    total = max(int(round(seconds)), 0)
    return f"{total // 60}:{total % 60:02d}"


def _default_year_from_table(match_df: pd.DataFrame) -> int:
    now_year = int(pd.Timestamp.now(tz="UTC").year)
    if match_df.empty or "date" not in match_df.columns:
        return now_year
    valid = match_df["date"].dropna()
    if valid.empty:
        return now_year
    return int(valid.max().year)


def _resolve_primary_metric(metrics_payload: Dict[str, Any], metric_name: str) -> Optional[float]:
    metrics_by_model = metrics_payload.get("metrics_by_model", {})
    primary_model = str(metrics_payload.get("primary_model", "")).lower()

    if primary_model in metrics_by_model and metric_name in metrics_by_model[primary_model]:
        return _safe_float(metrics_by_model[primary_model][metric_name], default=np.nan)

    for fallback in ["catboost", "lightgbm", "ridge_regression", "global_mean"]:
        if fallback in metrics_by_model and metric_name in metrics_by_model[fallback]:
            return _safe_float(metrics_by_model[fallback][metric_name], default=np.nan)
    return None


def _style_status_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    work = df.copy()

    def _cell_style(value: Any) -> str:
        token = str(value).strip().lower()
        if token == "won":
            return "background-color: rgba(32, 211, 141, 0.18); color: #c8f9e6; font-weight: 700;"
        if token == "lost":
            return "background-color: rgba(255, 95, 126, 0.18); color: #ffd8e1; font-weight: 700;"
        if token == "push":
            return "background-color: rgba(244, 187, 74, 0.18); color: #ffe9b7; font-weight: 700;"
        return "background-color: rgba(158, 176, 204, 0.16); color: #d9e5fa; font-weight: 600;"

    style_cols = [col for col in ["status"] if col in work.columns]
    styler = work.style
    if style_cols:
        styler = styler.applymap(_cell_style, subset=style_cols)
    return styler


def _match_selector_options(template_rows: pd.DataFrame, league: str) -> list[str]:
    if template_rows.empty:
        return [""]
    subset = template_rows.copy()
    if clean_text(league):
        subset = subset[subset["league"].astype(str) == str(league)]
    if subset.empty:
        subset = template_rows.copy()
    options = subset["template_label"].astype(str).tolist()
    return [""] + options


def _load_context(
    artifact_path: str,
    match_table_path: str,
    metrics_path: str,
    journal_path: str,
    include_explanation: bool,
) -> AppContext:
    artifact_file = Path(artifact_path)
    artifacts: Optional[Dict[str, Any]] = None
    artifact_error: Optional[str] = None

    if not artifact_file.exists():
        artifact_error = f"Artifact file not found: {artifact_file}"
    else:
        try:
            artifacts = load_artifacts_cached(str(artifact_file))
        except Exception as exc:
            artifact_error = f"Failed to load model artifacts: {exc}"

    match_df = load_match_table(match_table_path)
    options = extract_ui_options(match_df)
    team_history = team_history_view(match_df)
    template_rows = recent_template_rows(match_df)
    calendar_board = build_calendar_board(match_df)
    metrics_payload = load_metrics_payload(metrics_path)
    test_mae_minutes = load_test_mae_minutes(metrics_path)

    return AppContext(
        artifact_path=artifact_file,
        match_table_path=Path(match_table_path),
        metrics_path=Path(metrics_path),
        journal_path=Path(journal_path),
        include_explanation=include_explanation,
        artifacts=artifacts,
        artifact_error=artifact_error,
        config=artifacts.get("config", {}) if artifacts else {},
        match_df=match_df,
        options=options,
        team_history=team_history,
        template_rows=template_rows,
        calendar_board=calendar_board,
        metrics_payload=metrics_payload,
        test_mae_minutes=test_mae_minutes,
    )


def _render_top_nav() -> str:
    if "nav_page" not in st.session_state:
        st.session_state["nav_page"] = "Home"

    with st.sidebar:
        st.markdown(
            """
            <div style="padding:0.2rem 0 0.6rem 0;">
              <div style="font-family:'Sora',sans-serif;font-weight:800;font-size:1.05rem;">MinuteModel</div>
              <div style="font-size:0.78rem;color:#9fb3cf;">Esports Prediction Desk</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected = st.radio("Navigation", NAV_ITEMS, key="nav_page")
    return selected


def _render_sidebar_settings() -> Dict[str, Any]:
    if "artifact_path" not in st.session_state:
        st.session_state["artifact_path"] = DEFAULT_ARTIFACT_PATH
    if "match_table_path" not in st.session_state:
        st.session_state["match_table_path"] = DEFAULT_MATCH_TABLE_PATH
    if "metrics_path" not in st.session_state:
        st.session_state["metrics_path"] = DEFAULT_METRICS_PATH
    if "journal_path" not in st.session_state:
        st.session_state["journal_path"] = DEFAULT_JOURNAL_PATH
    if "include_explanation" not in st.session_state:
        st.session_state["include_explanation"] = True

    usage = get_usage_snapshot()
    with st.sidebar:
        st.metric("Free Predictions Left", usage["remaining"])
        st.caption(f"{usage['used']} / {usage['allowance']} used in {usage['month']}")

        with st.expander("Runtime Settings", expanded=False):
            st.text_input("Model artifact", key="artifact_path")
            st.text_input("Match table", key="match_table_path")
            st.text_input("Metrics file", key="metrics_path")
            st.text_input("Journal file", key="journal_path")
            st.checkbox("Include SHAP explanation", key="include_explanation")

        st.caption("Tip: keep runtime settings collapsed for a cleaner workflow.")

    return {
        "artifact_path": str(st.session_state["artifact_path"]),
        "match_table_path": str(st.session_state["match_table_path"]),
        "metrics_path": str(st.session_state["metrics_path"]),
        "journal_path": str(st.session_state["journal_path"]),
        "include_explanation": bool(st.session_state["include_explanation"]),
    }


def _set_nav(page: str) -> None:
    st.session_state["nav_page"] = page
    st.rerun()


def _home_cta_buttons() -> None:
    c1, c2, c3 = st.columns(3)
    if c1.button("Generate Prediction", use_container_width=True):
        _set_nav("Predictions")
    if c2.button("View Calendar", use_container_width=True):
        _set_nav("Calendar")
    if c3.button("Open Journal", use_container_width=True):
        _set_nav("Journal")

def _render_home_page(ctx: AppContext) -> None:
    usage = get_usage_snapshot()
    primary_model = "Unavailable"
    if ctx.artifacts:
        primary_model = str(ctx.artifacts.get("primary_model", "model")).upper()

    mae = ctx.test_mae_minutes if ctx.test_mae_minutes is not None else _resolve_primary_metric(ctx.metrics_payload, "mae_minutes")
    rmse = _resolve_primary_metric(ctx.metrics_payload, "rmse_minutes")
    within5 = _resolve_primary_metric(ctx.metrics_payload, "within_5_minutes_accuracy")
    render_page_intro(
        title="Pro Esports Predictions, Draft to Decision",
        subtitle=(
            "Generate pre-game projections in seconds, monitor performance in a personal journal, "
            "and keep model confidence visible at every step."
        ),
        badges=[
            "LoL: Live",
            "CS2: Frontend Ready",
            "Dota 2: Planned",
            "VALORANT: Planned",
            f"Free predictions left: {usage['remaining']}",
        ],
        eyebrow="Esports Oracle Workflow",
    )
    _home_cta_buttons()
    render_section_heading("Model Snapshot", "Current benchmark context for the production model.")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Primary model", primary_model)
    m2.metric("MAE (minutes)", f"{mae:.2f}" if mae is not None else "N/A")
    m3.metric("RMSE (minutes)", f"{rmse:.2f}" if rmse is not None else "N/A")
    m4.metric("Within 5 min", f"{within5 * 100:.1f}%" if within5 is not None else "N/A")

    render_section_heading("How It Works", "New users should be prediction-ready in under a minute.")
    h1, h2, h3 = st.columns(3)
    with h1:
        render_info_panel("1) Pick Match Context", "Select game, league, and a match template or custom draft.")
    with h2:
        render_info_panel("2) Generate Projection", "Run the model and review forecast, confidence, and key factors.")
    with h3:
        render_info_panel("3) Track Outcomes", "Log picks in Journal to monitor win-rate, ROI, and profit trends.")

    render_section_heading("Supported Games", "Coverage roadmap for the platform.")
    games = pd.DataFrame(
        [
            {"Game": "League of Legends", "Status": "Active", "Available prediction": "Draft-only match duration"},
            {"Game": "Counter-Strike 2", "Status": "UI ready", "Available prediction": "Connect your CS2 backend"},
            {"Game": "Dota 2", "Status": "Planned", "Available prediction": "Coming soon"},
            {"Game": "VALORANT", "Status": "Planned", "Available prediction": "Coming soon"},
        ]
    )
    st.dataframe(games, use_container_width=True, hide_index=True)

    render_section_heading("Trust & Methodology", "Clear assumptions keep the product transparent.")
    render_info_panel(
        "What this model is",
        "A draft-only benchmark. It uses information available before game start and does not leak in-game outcomes.",
        tone="positive",
    )
    st.markdown(
        """
        - Validation is chronological, not random, to mirror real forward predictions.
        - Baselines are tracked so model value is measured honestly.
        - Explanations describe model behavior, not causal certainty.
        """
    )


def _auto_fill_priors(ctx: AppContext) -> None:
    blue_team_name = clean_text(st.session_state.get("blue_team_name_custom")) or clean_text(
        st.session_state.get("blue_team_name")
    )
    red_team_name = clean_text(st.session_state.get("red_team_name_custom")) or clean_text(
        st.session_state.get("red_team_name")
    )
    blue_team_id = clean_text(st.session_state.get("blue_team_id_custom")) or clean_text(
        st.session_state.get("blue_team_id")
    )
    red_team_id = clean_text(st.session_state.get("red_team_id_custom")) or clean_text(
        st.session_state.get("red_team_id")
    )

    blue_duration, blue_ckpm = lookup_team_priors(ctx.team_history, blue_team_id, blue_team_name)
    red_duration, red_ckpm = lookup_team_priors(ctx.team_history, red_team_id, red_team_name)

    if blue_duration is not None:
        st.session_state["blue_duration_prior"] = float(blue_duration)
    if red_duration is not None:
        st.session_state["red_duration_prior"] = float(red_duration)
    if blue_ckpm is not None:
        st.session_state["blue_ckpm_prior"] = float(blue_ckpm)
    if red_ckpm is not None:
        st.session_state["red_ckpm_prior"] = float(red_ckpm)


def _apply_template_selection(ctx: AppContext, defaults: Dict[str, Any], template_label: str) -> None:
    if ctx.template_rows.empty or not template_label:
        return
    selected = ctx.template_rows[ctx.template_rows["template_label"] == template_label]
    if selected.empty:
        return
    updated = apply_template_to_defaults(selected.iloc[0], defaults)
    for key, value in updated.items():
        st.session_state[key] = value


def _consume_calendar_prefill(ctx: AppContext, defaults: Dict[str, Any]) -> None:
    prefill_gameid = st.session_state.get("selected_template_gameid")
    if prefill_gameid is None or ctx.template_rows.empty:
        return
    selected = ctx.template_rows[ctx.template_rows["gameid"].astype(str) == str(prefill_gameid)]
    if selected.empty:
        st.session_state["selected_template_gameid"] = None
        return
    updated = apply_template_to_defaults(selected.iloc[0], defaults)
    for key, value in updated.items():
        st.session_state[key] = value
    st.session_state["selected_template_gameid"] = None
    st.success("Loaded selected calendar match into the prediction form.")

def _prediction_result_card(ctx: AppContext, result: Dict[str, Any]) -> None:
    quantile_payload = result.get("quantile_predictions", {}) or {}
    pred_minutes = _safe_float(
        quantile_payload.get("predicted_p50_minutes", result.get("predicted_p50_minutes", result.get("predicted_duration_minutes"))),
        0.0,
    )
    mae_minutes = _safe_float(result.get("mae_context_minutes"), 4.2)
    rmse_minutes = _safe_float(result.get("rmse_context_minutes"), max(mae_minutes * 1.3, 1.8))
    line_minutes = _safe_float(result.get("market_line_minutes"), 0.0)

    confidence_label = "Directional"
    confidence_detail = "Set a market line to derive directional probability."
    p_over = None
    p_under = None
    edge_pct = None
    predicted_side = "N/A"
    win_probability = None

    if line_minutes > 0:
        z = (pred_minutes - line_minutes) / max(rmse_minutes, 1e-6)
        p_over = float(_normal_cdf(z))
        p_under = float(1.0 - p_over)
        predicted_side = f"Over {line_minutes:.1f}m" if p_over >= p_under else f"Under {line_minutes:.1f}m"
        win_probability = max(p_over, p_under)
        edge_pct = float(abs(max(p_over, p_under) - 0.5) * 100.0)
        if edge_pct >= 15:
            confidence_label = "High"
        elif edge_pct >= 8:
            confidence_label = "Medium"
        else:
            confidence_label = "Low"
        confidence_detail = (
            f"Over {line_minutes:.1f}m: {p_over * 100:.1f}% | "
            f"Under {line_minutes:.1f}m: {p_under * 100:.1f}%"
        )

    quantile_thresholds = ((ctx.artifacts or {}).get("quantile_regression", {}) or {}).get("volatility_thresholds_minutes", {})

    render_section_heading(
        "Prediction Result",
        "p50 is the headline forecast. p10-p90 shows the likely pre-game duration range.",
    )
    render_duration_prediction_card(
        prediction=result,
        volatility_thresholds_minutes=quantile_thresholds if isinstance(quantile_thresholds, dict) else None,
    )

    if line_minutes > 0 and p_over is not None and p_under is not None:
        render_section_heading("Market Context", "Optional over/under framing relative to your chosen line.")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Market lean", predicted_side)
        k2.metric("Win probability", f"{win_probability * 100:.1f}%")
        k3.metric("Confidence / edge", f"{confidence_label} ({edge_pct:.1f}%)")
        k4.metric("Line", f"{line_minutes:.1f} min")
        st.info(
            f"{confidence_detail}. Estimated edge: {edge_pct:.1f}%."
        )
        st.caption(
            "Probability uses a normal error approximation around model RMSE and should be treated as directional."
        )

    explanation = result.get("explanation")
    if explanation:
        if explanation.get("warning"):
            st.warning(str(explanation["warning"]))
        top_features = explanation.get("top_feature_contributions", [])
        if top_features:
            top_df = pd.DataFrame(top_features)
            if "shap_value" in top_df.columns:
                top_df["impact"] = np.where(top_df["shap_value"] >= 0, "Longer", "Shorter")
                top_df["abs_shap"] = top_df["shap_value"].abs()
                top_df = top_df.sort_values("abs_shap", ascending=False)
                top_df["shap_value"] = top_df["shap_value"].round(4)
            render_section_heading("Key Supporting Factors")
            cols = [c for c in ["feature", "impact", "shap_value"] if c in top_df.columns]
            st.dataframe(top_df[cols], use_container_width=True, hide_index=True)

    with st.expander("Inference payload"):
        st.json(result.get("payload", {}))

    with st.expander("Save to Journal"):
        prediction_id = str(result.get("prediction_id"))
        already_saved = st.session_state.get("last_saved_prediction_id") == prediction_id
        j1, j2, j3 = st.columns(3)
        status = j1.selectbox(
            "Outcome status",
            ["Pending", "Won", "Lost", "Push"],
            index=0,
            key=f"journal_status_{prediction_id}",
        )
        odds = j2.number_input(
            "Decimal odds",
            min_value=1.01,
            value=1.90,
            step=0.01,
            key=f"journal_odds_{prediction_id}",
        )
        stake = j3.number_input(
            "Stake (units)",
            min_value=0.1,
            value=1.0,
            step=0.1,
            key=f"journal_stake_{prediction_id}",
        )
        notes = st.text_area(
            "Notes",
            key=f"journal_notes_{prediction_id}",
            placeholder="Context, line movement, or reasoning...",
        )
        if st.button("Add to Journal", use_container_width=True, disabled=already_saved, key=f"save_{prediction_id}"):
            entry = {
                "entry_id": prediction_id,
                "created_at_utc": result.get("prediction_timestamp_utc"),
                "game": result.get("game", "League of Legends"),
                "league": result.get("league", ""),
                "match_label": result.get("match_label", ""),
                "prediction_type": "Match Duration (minutes)",
                "predicted_value": pred_minutes,
                "confidence_label": confidence_label,
                "model_name": result.get("model_name", ""),
                "status": status,
                "odds_decimal": float(odds),
                "stake_units": float(stake),
                "profit_units": np.nan,
                "notes": notes,
            }
            append_journal_entry(ctx.journal_path, entry)
            st.session_state["last_saved_prediction_id"] = prediction_id
            st.success("Prediction saved to journal.")


def _render_predictions_page(ctx: AppContext) -> None:
    usage = get_usage_snapshot()
    render_page_intro(
        title="Predictions",
        subtitle="Configure pre-game inputs and generate a production inference in one flow.",
        badges=[
            "Primary action",
            f"Free predictions left: {usage['remaining']}",
            "Draft-only model",
        ],
        eyebrow="Prediction Console",
    )

    if ctx.artifact_error:
        render_info_panel("Model artifact missing", ctx.artifact_error, tone="danger")
        st.info("Update the model artifact path in Runtime Settings to enable predictions.")
        return

    defaults = default_form_state(ctx.options, default_year=_default_year_from_table(ctx.match_df))
    if "market_line_minutes" not in st.session_state:
        st.session_state["market_line_minutes"] = 0.0
    if "prediction_league_selector" not in st.session_state:
        st.session_state["prediction_league_selector"] = st.session_state.get("league", "")
    if "prediction_match_selector" not in st.session_state:
        st.session_state["prediction_match_selector"] = ""
    ensure_form_state(defaults)
    _consume_calendar_prefill(ctx, defaults)

    with st.container(border=True):
        render_section_heading("Match Selection", "Pick game, league, and optional match template before editing draft details.")
        s1, s2, s3 = st.columns(3)
        game_choice = s1.selectbox("Game", SUPPORTED_GAMES, key="game")
        league_selector = s2.selectbox(
            "Region / League",
            options_with_current(ctx.options["leagues"], st.session_state.get("prediction_league_selector", "")),
            key="prediction_league_selector",
        )
        template_options = _match_selector_options(ctx.template_rows, league_selector)
        selected_template_label = s3.selectbox(
            "Match selector",
            template_options,
            key="prediction_match_selector",
            help="Loads a recent match as a starting scaffold.",
        )

        if clean_text(league_selector):
            st.session_state["league"] = league_selector

        b1, b2, b3, b4 = st.columns(4)
        if b1.button("Load Match", use_container_width=True, disabled=not bool(selected_template_label)):
            _apply_template_selection(ctx, defaults, selected_template_label)
            st.rerun()
        if b2.button("Swap Blue/Red", use_container_width=True):
            swap_form_sides()
            st.rerun()
        if b3.button("Auto-fill Priors", use_container_width=True, disabled=ctx.team_history.empty):
            _auto_fill_priors(ctx)
            st.rerun()
        if b4.button("Reset Form", use_container_width=True):
            for key, value in defaults.items():
                st.session_state[key] = value
            st.session_state["market_line_minutes"] = 0.0
            st.rerun()

    if game_choice != "League of Legends":
        render_info_panel(
            "Backend routing notice",
            "This deployment is currently wired to the LoL draft-duration model. Switch to League of Legends to run inference.",
            tone="warning",
        )

    filled_roles = sum(
        bool(clean_text(st.session_state.get(f"{side}_role_{role}", "")))
        for side in ["blue", "red"]
        for role in ROLE_ORDER
    )
    st.progress(filled_roles / 10.0, text=f"Draft completion: {filled_roles}/10 champions")

    champ_options = ctx.options.get("champions", [])
    if not champ_options:
        champ_options = ["Aatrox", "Lee Sin", "Orianna", "Jinx", "Nautilus"]

    with st.form("prediction_form", clear_on_submit=False):
        render_section_heading("Prediction Form", "Complete draft, priors, and market context to generate output.")
        m1, m2, m3 = st.columns(3)
        patch = m1.selectbox(
            "Patch",
            options_with_current(ctx.options["patches"], st.session_state.get("patch", "")),
            key="patch",
        )
        split = m2.selectbox(
            "Split",
            options_with_current(ctx.options["splits"], st.session_state.get("split", "")),
            key="split",
        )
        year = m3.number_input("Year", min_value=2020, max_value=2035, step=1, key="year")
        m4, m5, m6 = st.columns(3)
        playoffs = m4.toggle("Playoffs", key="playoffs")
        blue_first_pick = m5.toggle("Blue has first pick", key="blue_first_pick")
        market_line_minutes = m6.number_input(
            "Market line (minutes)",
            min_value=0.0,
            step=0.25,
            key="market_line_minutes",
            help="Optional over/under line to estimate directional probability.",
        )

        render_section_heading("Teams", "Set team names/IDs and optional custom overrides.")
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("#### Blue Side")
            blue_team_name = st.selectbox(
                "Team name",
                options_with_current(ctx.options["team_names"], st.session_state.get("blue_team_name", "")),
                key="blue_team_name",
            )
            blue_team_name_custom = st.text_input("Team name override", key="blue_team_name_custom")
            blue_team_id = st.selectbox(
                "Team ID",
                options_with_current(ctx.options["team_ids"], st.session_state.get("blue_team_id", "")),
                key="blue_team_id",
            )
            blue_team_id_custom = st.text_input("Team ID override", key="blue_team_id_custom")
        with t2:
            st.markdown("#### Red Side")
            red_team_name = st.selectbox(
                "Team name",
                options_with_current(ctx.options["team_names"], st.session_state.get("red_team_name", "")),
                key="red_team_name",
            )
            red_team_name_custom = st.text_input("Team name override", key="red_team_name_custom")
            red_team_id = st.selectbox(
                "Team ID",
                options_with_current(ctx.options["team_ids"], st.session_state.get("red_team_id", "")),
                key="red_team_id",
            )
            red_team_id_custom = st.text_input("Team ID override", key="red_team_id_custom")

        render_section_heading("Historical Priors", "Rolling priors improve stability for unseen or sparse contexts.")
        p1, p2 = st.columns(2)
        blue_duration_prior = p1.number_input(
            "Blue rolling duration prior (seconds)",
            min_value=300.0,
            step=10.0,
            key="blue_duration_prior",
        )
        red_duration_prior = p2.number_input(
            "Red rolling duration prior (seconds)",
            min_value=300.0,
            step=10.0,
            key="red_duration_prior",
        )
        p3, p4 = st.columns(2)
        blue_ckpm_prior = p3.number_input(
            "Blue rolling CKPM prior",
            min_value=0.1,
            step=0.01,
            key="blue_ckpm_prior",
        )
        red_ckpm_prior = p4.number_input(
            "Red rolling CKPM prior",
            min_value=0.1,
            step=0.01,
            key="red_ckpm_prior",
        )

        render_section_heading("Draft Champions", "Role-specific champions are required for both sides.")
        q1, q2 = st.columns(2)
        blue_roles_quick = q1.text_input("Blue quick input (top,jng,mid,bot,sup)", key="blue_roles_quick")
        red_roles_quick = q2.text_input("Red quick input (top,jng,mid,bot,sup)", key="red_roles_quick")

        blue_roles: Dict[str, str] = {}
        red_roles: Dict[str, str] = {}
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("#### Blue Draft")
            for role in ROLE_ORDER:
                blue_roles[role] = st.selectbox(
                    f"{role.upper()}",
                    options_with_current(champ_options, st.session_state.get(f"blue_role_{role}", "")),
                    key=f"blue_role_{role}",
                )
        with r2:
            st.markdown("#### Red Draft")
            for role in ROLE_ORDER:
                red_roles[role] = st.selectbox(
                    f"{role.upper()}",
                    options_with_current(champ_options, st.session_state.get(f"red_role_{role}", "")),
                    key=f"red_role_{role}",
                )

        with st.expander("Optional Pick / Ban Overrides"):
            st.caption("If picks are blank, role champions are used.")
            tab_blue, tab_red = st.tabs(["Blue Overrides", "Red Overrides"])
            with tab_blue:
                blue_pick_inputs = [st.text_input(f"Blue pick{i}", key=f"blue_pick_{i}") for i in range(1, 6)]
                blue_ban_inputs = [st.text_input(f"Blue ban{i}", key=f"blue_ban_{i}") for i in range(1, 6)]
            with tab_red:
                red_pick_inputs = [st.text_input(f"Red pick{i}", key=f"red_pick_{i}") for i in range(1, 6)]
                red_ban_inputs = [st.text_input(f"Red ban{i}", key=f"red_ban_{i}") for i in range(1, 6)]

        submitted = st.form_submit_button("Generate Prediction", use_container_width=True, type="primary")

    if submitted:
        usage = get_usage_snapshot()
        if usage["remaining"] <= 0:
            st.warning(
                "Free prediction allowance reached for this month. Reset in Account/Usage or increase allowance."
            )
            return

        if game_choice != "League of Legends":
            st.warning(
                "This app instance is currently wired to the League of Legends draft-duration model. "
                "Select League of Legends to run prediction."
            )
            return

        blue_team_name_final = clean_text(blue_team_name_custom) or clean_text(blue_team_name)
        red_team_name_final = clean_text(red_team_name_custom) or clean_text(red_team_name)
        blue_team_id_final = clean_text(blue_team_id_custom) or clean_text(blue_team_id)
        red_team_id_final = clean_text(red_team_id_custom) or clean_text(red_team_id)

        parse_errors = []
        if clean_text(blue_roles_quick):
            parsed = parse_role_quick_input(blue_roles_quick)
            if parsed:
                for idx, role in enumerate(ROLE_ORDER):
                    blue_roles[role] = parsed[idx]
            else:
                parse_errors.append("Blue quick input must contain exactly 5 champions.")

        if clean_text(red_roles_quick):
            parsed = parse_role_quick_input(red_roles_quick)
            if parsed:
                for idx, role in enumerate(ROLE_ORDER):
                    red_roles[role] = parsed[idx]
            else:
                parse_errors.append("Red quick input must contain exactly 5 champions.")

        validation_errors = parse_errors + validate_draft_inputs(blue_roles, red_roles)
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
            return

        form_data = {
            "league": st.session_state.get("prediction_league_selector", st.session_state.get("league", "")),
            "patch": patch,
            "split": split,
            "year": int(year),
            "playoffs": bool(playoffs),
            "blue_first_pick": bool(blue_first_pick),
            "blue_team_name": blue_team_name_final,
            "red_team_name": red_team_name_final,
            "blue_team_id": blue_team_id_final,
            "red_team_id": red_team_id_final,
            "blue_duration_prior": float(blue_duration_prior),
            "red_duration_prior": float(red_duration_prior),
            "blue_ckpm_prior": float(blue_ckpm_prior),
            "red_ckpm_prior": float(red_ckpm_prior),
            "blue_roles": blue_roles,
            "red_roles": red_roles,
            "blue_picks": blue_pick_inputs,
            "red_picks": red_pick_inputs,
            "blue_bans": blue_ban_inputs,
            "red_bans": red_ban_inputs,
        }
        payload = build_payload(form_data)

        try:
            with st.spinner("Running model inference..."):
                prediction = predict_single_draft(
                    artifact_path=ctx.artifact_path,
                    draft_payload=payload,
                    include_explanation=ctx.include_explanation,
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        increment_usage()
        prediction_id = str(uuid.uuid4())
        primary_model = str((ctx.artifacts or {}).get("primary_model", "model"))
        mae_context = ctx.test_mae_minutes
        if mae_context is None:
            mae_context = _resolve_primary_metric(ctx.metrics_payload, "mae_minutes")
        if mae_context is None:
            mae_context = 4.2

        rmse_context = _resolve_primary_metric(ctx.metrics_payload, "rmse_minutes")
        if rmse_context is None:
            rmse_context = max(float(mae_context) * 1.3, 1.8)

        quantile_payload = prediction.get("quantile_predictions", {}) or {}
        result_record = {
            "prediction_id": prediction_id,
            "prediction_timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
            "game": game_choice,
            "league": st.session_state.get("prediction_league_selector", st.session_state.get("league", "")),
            "match_label": f"{blue_team_name_final} vs {red_team_name_final}",
            "model_name": primary_model,
            "predicted_duration_minutes": float(prediction["predicted_duration_minutes"]),
            "predicted_duration_seconds": float(prediction["predicted_duration_seconds"]),
            "mae_context_minutes": float(mae_context),
            "rmse_context_minutes": float(rmse_context),
            "market_line_minutes": float(market_line_minutes),
            "payload": payload,
            "explanation": prediction.get("explanation"),
            "quantile_predictions": quantile_payload,
            "predicted_p10_minutes": _safe_float(quantile_payload.get("predicted_p10_minutes"), np.nan),
            "predicted_p50_minutes": _safe_float(
                quantile_payload.get("predicted_p50_minutes", prediction.get("predicted_duration_minutes")),
                np.nan,
            ),
            "predicted_p90_minutes": _safe_float(quantile_payload.get("predicted_p90_minutes"), np.nan),
            "interval_width_minutes": _safe_float(quantile_payload.get("interval_width_minutes"), np.nan),
            "volatility_flag": quantile_payload.get("volatility_flag"),
        }
        st.session_state["last_prediction"] = result_record
        st.session_state["last_saved_prediction_id"] = None

    last_prediction = st.session_state.get("last_prediction")
    if last_prediction:
        _prediction_result_card(ctx, last_prediction)
    else:
        render_empty_state(
            "No prediction generated yet",
            "Configure match details and click Generate Prediction to populate this panel.",
        )


def _render_calendar_page(ctx: AppContext) -> None:
    render_page_intro(
        title="Calendar",
        subtitle="Scan upcoming/recent matches, filter quickly, and move selected fixtures into Predictions.",
        badges=["High scanability", "Featured match tagging"],
        eyebrow="Match Board",
    )

    board = ctx.calendar_board.copy()
    if board.empty:
        render_empty_state(
            "No match data available",
            "Update the match table path in Runtime Settings to populate the calendar.",
        )
        return

    board["date"] = pd.to_datetime(board["date"], errors="coerce")
    board = board.dropna(subset=["date"])

    render_section_heading("Filters", "Narrow by game, league, date range, and featured status.")
    c1, c2, c3 = st.columns(3)
    selected_game = c1.selectbox("Game", ["All", "League of Legends", "Counter-Strike 2", "Dota 2", "VALORANT"])
    league_options = ["All"] + sorted(board["league"].dropna().astype(str).unique().tolist())
    selected_league = c2.selectbox("League", league_options)
    featured_only = c3.toggle("Featured only", value=False)

    min_date = board["date"].min().date()
    max_date = board["date"].max().date()
    date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)

    filtered = board.copy()
    if selected_game != "All":
        filtered = filtered[filtered["game"] == selected_game]
    if selected_league != "All":
        filtered = filtered[filtered["league"] == selected_league]
    if featured_only:
        filtered = filtered[filtered["featured"] == True]

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date = pd.Timestamp(date_range[0])
        end_date = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered = filtered[(filtered["date"] >= start_date) & (filtered["date"] <= end_date)]

    if filtered.empty:
        render_empty_state("No matches found", "Adjust filters to see more fixtures.")
        return

    filtered = filtered.sort_values("date", ascending=False)
    featured_count = int(filtered["featured"].astype(bool).sum())
    render_section_heading("Calendar Overview")
    k1, k2, k3 = st.columns(3)
    k1.metric("Matches shown", int(len(filtered)))
    k2.metric("Featured matches", featured_count)
    k3.metric("Leagues shown", int(filtered["league"].nunique()))

    display = filtered.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d %H:%M")
    display["featured"] = np.where(display["featured"], "Featured", "Standard")
    display["interest"] = np.where(display["featured"] == "Featured", "High", "Normal")
    display = display.rename(columns={"match": "matchup"})
    st.dataframe(
        display[["date", "game", "league", "patch", "matchup", "featured", "interest"]],
        use_container_width=True,
        hide_index=True,
    )

    filtered["calendar_label"] = (
        filtered["date"].dt.strftime("%Y-%m-%d")
        + " | "
        + filtered["league"].astype(str)
        + " | "
        + filtered["match"].astype(str)
        + " | patch "
        + filtered["patch"].astype(str)
    )
    render_section_heading("Jump to Predictions", "Choose a match and load it directly into the prediction form.")
    selected_label = st.selectbox(
        "Open match in prediction flow",
        filtered["calendar_label"].tolist(),
    )
    if st.button("Use Selected Match in Predictions", use_container_width=True):
        selected = filtered[filtered["calendar_label"] == selected_label]
        if not selected.empty:
            st.session_state["selected_template_gameid"] = str(selected.iloc[0]["gameid"])
            _set_nav("Predictions")


def _render_journal_page(ctx: AppContext) -> None:
    render_page_intro(
        title="Journal",
        subtitle="Your lightweight performance dashboard for tracking picks, outcomes, and betting efficiency.",
        badges=["KPI tracking", "Editable outcomes", "Profit curve"],
        eyebrow="Performance Journal",
    )

    journal_df = load_journal(ctx.journal_path)
    if journal_df.empty:
        render_empty_state(
            "Journal is empty",
            "Generate a prediction and save it from the Predictions page to start tracking performance.",
        )
        return

    missing_id = journal_df["entry_id"].astype(str).str.strip().isin(["", "nan", "None"])
    if missing_id.any():
        journal_df.loc[missing_id, "entry_id"] = [str(uuid.uuid4()) for _ in range(int(missing_id.sum()))]
        save_journal(ctx.journal_path, journal_df)

    metrics = compute_journal_metrics(journal_df)
    render_section_heading("Journal KPIs")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total picks", metrics["total_picks"])
    m2.metric("Win rate", f"{metrics['win_rate'] * 100:.1f}%")
    m3.metric("ROI", f"{metrics['roi'] * 100:.1f}%")
    m4.metric("Profit/Loss", f"{metrics['profit_units']:.2f}u")
    m5.metric("Avg odds", f"{metrics['avg_odds']:.2f}")

    render_section_heading("Filters & Sorting", "Use filters to isolate specific games, leagues, or status groups.")
    f1, f2, f3 = st.columns(3)
    game_options = ["All"] + sorted(journal_df["game"].dropna().astype(str).unique().tolist())
    status_options = ["All"] + sorted(journal_df["status"].dropna().astype(str).unique().tolist())
    league_options = ["All"] + sorted(journal_df["league"].dropna().astype(str).unique().tolist())
    selected_game = f1.selectbox("Game filter", game_options)
    selected_status = f2.selectbox("Status filter", status_options)
    selected_league = f3.selectbox("League filter", league_options)

    filtered = journal_df.copy()
    if selected_game != "All":
        filtered = filtered[filtered["game"] == selected_game]
    if selected_status != "All":
        filtered = filtered[filtered["status"] == selected_status]
    if selected_league != "All":
        filtered = filtered[filtered["league"] == selected_league]

    if filtered.empty:
        render_empty_state("No entries match these filters", "Try broadening status/game/league filters.")
        return

    sort_col = st.selectbox(
        "Sort by",
        ["created_at_utc", "profit_units", "odds_decimal", "league", "status"],
    )
    ascending = st.toggle("Ascending sort", value=False)
    filtered = filtered.sort_values(sort_col, ascending=ascending, na_position="last")

    view_cols = [
        "created_at_utc",
        "game",
        "league",
        "match_label",
        "predicted_value",
        "status",
        "odds_decimal",
        "stake_units",
        "profit_units",
        "notes",
    ]
    table_view = filtered[view_cols].copy()
    table_view["status"] = table_view["status"].map(status_label)
    st.dataframe(_style_status_dataframe(table_view), use_container_width=True, hide_index=True)

    settled = filtered[filtered["status"].str.lower().isin(["won", "lost", "push"])].copy()
    if not settled.empty:
        settled["created_at_utc"] = pd.to_datetime(settled["created_at_utc"], errors="coerce")
        settled = settled.sort_values("created_at_utc")
        settled["profit_units"] = pd.to_numeric(settled["profit_units"], errors="coerce").fillna(0.0)
        settled["cumulative_profit"] = settled["profit_units"].cumsum()
        chart_df = settled[["created_at_utc", "cumulative_profit"]].set_index("created_at_utc")
        render_section_heading("Cumulative Profit")
        st.line_chart(chart_df)

    with st.expander("Edit Journal Entries"):
        editable_cols = ["entry_id", "status", "odds_decimal", "stake_units", "notes"]
        edit_df = filtered[editable_cols].copy()
        edited = st.data_editor(
            edit_df,
            use_container_width=True,
            hide_index=True,
            num_rows="fixed",
            key="journal_editor",
        )
        if st.button("Save Entry Updates", use_container_width=True):
            base = journal_df.set_index("entry_id")
            edited_indexed = edited.set_index("entry_id")
            for col in ["status", "odds_decimal", "stake_units", "notes"]:
                base.loc[edited_indexed.index, col] = edited_indexed[col]
            updated = base.reset_index()
            save_journal(ctx.journal_path, updated)
            st.success("Journal updates saved.")
            st.rerun()


def _load_optional_breakdown(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _render_model_performance_page(ctx: AppContext) -> None:
    render_page_intro(
        title="Model Performance",
        subtitle="Backtest metrics are presented transparently and should be interpreted as draft-only benchmarks.",
        badges=["Chronological validation", "LoL + CS2 sections", "Honest baseline comparisons"],
        eyebrow="Benchmark Center",
    )

    payload = ctx.metrics_payload
    metrics_by_model = payload.get("metrics_by_model", {})
    if not metrics_by_model:
        render_empty_state(
            "No metrics payload found",
            "Check the metrics path in Runtime Settings to populate performance dashboards.",
        )
        return

    primary_model = str(payload.get("primary_model", "unknown")).lower()
    primary_metrics = metrics_by_model.get(primary_model, {})

    render_section_heading("Primary Model Snapshot")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Primary model", primary_model.upper())
    k2.metric("MAE", f"{_safe_float(primary_metrics.get('mae_minutes'), np.nan):.2f} min")
    k3.metric("RMSE", f"{_safe_float(primary_metrics.get('rmse_minutes'), np.nan):.2f} min")
    k4.metric(
        "Within 5 min",
        f"{_safe_float(primary_metrics.get('within_5_minutes_accuracy'), np.nan) * 100:.1f}%",
    )

    tab_lol, tab_cs2 = st.tabs(["League of Legends", "Counter-Strike 2"])
    with tab_lol:
        render_section_heading("Model Comparison", "Lower MAE/RMSE and higher within-band rates are preferred.")
        rows = []
        for model_name, metrics in metrics_by_model.items():
            rows.append(
                {
                    "model": model_name,
                    "mae_minutes": _safe_float(metrics.get("mae_minutes"), np.nan),
                    "rmse_minutes": _safe_float(metrics.get("rmse_minutes"), np.nan),
                    "median_abs_error_minutes": _safe_float(metrics.get("median_absolute_error_minutes"), np.nan),
                    "within_2_min": _safe_float(metrics.get("within_2_minutes_accuracy"), np.nan),
                    "within_5_min": _safe_float(metrics.get("within_5_minutes_accuracy"), np.nan),
                }
            )
        summary_df = pd.DataFrame(rows).sort_values("mae_minutes", ascending=True)
        summary_df["within_2_min"] = (summary_df["within_2_min"] * 100.0).round(2)
        summary_df["within_5_min"] = (summary_df["within_5_min"] * 100.0).round(2)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        split_info = payload.get("split", {})
        if split_info:
            render_section_heading("Chronological Split")
            split_df = pd.DataFrame([split_info])
            st.dataframe(split_df, use_container_width=True, hide_index=True)

        base_dir = ctx.metrics_path.parent
        league_df = _load_optional_breakdown(base_dir / "error_by_league.csv")
        patch_df = _load_optional_breakdown(base_dir / "error_by_patch.csv")
        bucket_df = _load_optional_breakdown(base_dir / "error_by_duration_bucket.csv")

        if not league_df.empty:
            st.markdown("#### Error by League")
            st.dataframe(league_df, use_container_width=True, hide_index=True)
        if not patch_df.empty:
            st.markdown("#### Error by Patch")
            st.dataframe(patch_df, use_container_width=True, hide_index=True)
        if not bucket_df.empty:
            st.markdown("#### Error by Duration Bucket")
            st.dataframe(bucket_df, use_container_width=True, hide_index=True)

        st.info(
            "These metrics come from chronological holdout evaluation. Draft-only predictions are noisier "
            "than live-state models that use in-game telemetry."
        )

    with tab_cs2:
        render_info_panel(
            "CS2 panel ready",
            "Connect a CS2 metrics payload to display the same performance framework on this tab.",
            tone="warning",
        )
        st.caption("Current backend integration in this app instance points to the LoL model artifact.")


def _render_account_page(ctx: AppContext) -> None:
    render_page_intro(
        title="Account / Usage",
        subtitle="Manage free prediction allowance, monitor monthly usage, and export account data artifacts.",
        badges=["Usage controls", "Path visibility", "Journal export"],
        eyebrow="Account Center",
    )
    usage = get_usage_snapshot()

    render_section_heading("Usage Overview")
    m1, m2, m3 = st.columns(3)
    m1.metric("Monthly allowance", usage["allowance"])
    m2.metric("Used this month", usage["used"])
    m3.metric("Remaining", usage["remaining"])

    progress = 0.0
    if usage["allowance"] > 0:
        progress = min(float(usage["used"]) / float(usage["allowance"]), 1.0)
    st.progress(progress, text=f"Usage month: {usage['month']}")

    c1, c2 = st.columns(2)
    allowance_input = c1.number_input(
        "Set free monthly allowance",
        min_value=1,
        max_value=1000,
        value=int(st.session_state.get("usage_allowance", FREE_PREDICTIONS_PER_MONTH)),
        step=1,
    )
    if c1.button("Update allowance", use_container_width=True):
        st.session_state["usage_allowance"] = int(allowance_input)
        st.success("Usage allowance updated.")
        st.rerun()

    if c2.button("Reset usage counter", use_container_width=True):
        reset_usage()
        st.success("Usage counter reset for current month.")
        st.rerun()

    render_section_heading("Data Paths", "Current app pointers for artifacts and reports.")
    path_df = pd.DataFrame(
        [
            {"artifact_path": str(ctx.artifact_path)},
            {"match_table_path": str(ctx.match_table_path)},
            {"metrics_path": str(ctx.metrics_path)},
            {"journal_path": str(ctx.journal_path)},
        ]
    )
    st.dataframe(path_df, use_container_width=True, hide_index=True)

    journal_df = load_journal(ctx.journal_path)
    if not journal_df.empty:
        csv_bytes = journal_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Journal CSV",
            data=csv_bytes,
            file_name="prediction_journal.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    st.set_page_config(page_title="MinuteModel", page_icon=":chart_with_upwards_trend:", layout="wide")
    apply_app_styles()
    ensure_usage_state()

    _render_top_nav()
    settings = _render_sidebar_settings()
    ctx = _load_context(
        artifact_path=settings["artifact_path"],
        match_table_path=settings["match_table_path"],
        metrics_path=settings["metrics_path"],
        journal_path=settings["journal_path"],
        include_explanation=bool(settings["include_explanation"]),
    )

    page = st.session_state.get("nav_page", "Home")
    if page == "Home":
        _render_home_page(ctx)
    elif page == "Predictions":
        _render_predictions_page(ctx)
    elif page == "Calendar":
        _render_calendar_page(ctx)
    elif page == "Journal":
        _render_journal_page(ctx)
    elif page == "Model Performance":
        _render_model_performance_page(ctx)
    elif page == "Account / Usage":
        _render_account_page(ctx)
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()
