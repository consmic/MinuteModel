from __future__ import annotations

import html
import math
from typing import Any, Dict, Iterable, Optional

import streamlit as st


def render_page_intro(
    title: str,
    subtitle: str,
    badges: Optional[Iterable[str]] = None,
    eyebrow: str = "MinuteModel",
) -> None:
    title_safe = html.escape(str(title))
    subtitle_safe = html.escape(str(subtitle))
    eyebrow_safe = html.escape(str(eyebrow))
    badge_html = ""
    if badges:
        badge_html = "".join([f'<span class="oo-badge">{html.escape(str(item))}</span>' for item in badges])
    st.markdown(
        f"""
        <section class="oo-hero">
            <div class="oo-eyebrow">{eyebrow_safe}</div>
            <h1 class="oo-title">{title_safe}</h1>
            <p class="oo-subtitle">{subtitle_safe}</p>
            <div class="oo-badge-row">{badge_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(title: str, subtitle: str = "") -> None:
    title_safe = html.escape(str(title))
    subtitle_safe = html.escape(str(subtitle))
    st.markdown(
        f"""
        <div class="oo-section-head">{title_safe}</div>
        <div class="oo-section-sub">{subtitle_safe}</div>
        """,
        unsafe_allow_html=True,
    )


def render_info_panel(title: str, body: str, tone: str = "neutral") -> None:
    css_class = "oo-panel-neutral"
    if tone == "positive":
        css_class = "oo-panel-positive"
    elif tone == "warning":
        css_class = "oo-panel-warning"
    elif tone == "danger":
        css_class = "oo-panel-danger"

    title_safe = html.escape(str(title))
    body_safe = html.escape(str(body))
    st.markdown(
        f"""
        <div class="oo-panel {css_class}">
            <div class="oo-panel-title">{title_safe}</div>
            <div class="oo-panel-body">{body_safe}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, body: str) -> None:
    title_safe = html.escape(str(title))
    body_safe = html.escape(str(body))
    st.markdown(
        f"""
        <div class="oo-empty">
            <div class="oo-empty-title">{title_safe}</div>
            <div class="oo-empty-body">{body_safe}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _to_optional_float(value: Any) -> Optional[float]:
    try:
        cast = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(cast) or math.isinf(cast):
        return None
    return cast


def _first_number(*values: Any) -> Optional[float]:
    for value in values:
        cast = _to_optional_float(value)
        if cast is not None:
            return cast
    return None


def _volatility_css_class(flag: str) -> str:
    token = str(flag or "").strip().lower()
    if token.startswith("low"):
        return "oo-volatility-low"
    if token.startswith("medium"):
        return "oo-volatility-medium"
    if token.startswith("high"):
        return "oo-volatility-high"
    return "oo-volatility-neutral"


def _resolve_duration_prediction(
    prediction: Dict[str, Any],
    volatility_thresholds_minutes: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    quantile_payload = prediction.get("quantile_predictions", {}) or {}

    p10 = _first_number(
        quantile_payload.get("predicted_p10_minutes"),
        prediction.get("predicted_p10_minutes"),
        quantile_payload.get("pred_p10_minutes"),
        prediction.get("pred_p10_minutes"),
    )
    p50 = _first_number(
        quantile_payload.get("predicted_p50_minutes"),
        prediction.get("predicted_p50_minutes"),
        quantile_payload.get("pred_p50_minutes"),
        prediction.get("pred_p50_minutes"),
        prediction.get("predicted_duration_minutes"),
    )
    p90 = _first_number(
        quantile_payload.get("predicted_p90_minutes"),
        prediction.get("predicted_p90_minutes"),
        quantile_payload.get("pred_p90_minutes"),
        prediction.get("pred_p90_minutes"),
    )

    interval_width = _first_number(
        quantile_payload.get("interval_width_minutes"),
        prediction.get("interval_width_minutes"),
    )
    if interval_width is None and p10 is not None and p90 is not None:
        interval_width = max(p90 - p10, 0.0)

    volatility_flag = str(
        quantile_payload.get("volatility_flag")
        or prediction.get("volatility_flag")
        or ""
    ).strip()

    if not volatility_flag and interval_width is not None and volatility_thresholds_minutes:
        low_threshold = _to_optional_float(volatility_thresholds_minutes.get("low"))
        high_threshold = _to_optional_float(volatility_thresholds_minutes.get("high"))
        if low_threshold is not None and high_threshold is not None:
            if interval_width <= low_threshold:
                volatility_flag = "low volatility"
            elif interval_width <= high_threshold:
                volatility_flag = "medium volatility"
            else:
                volatility_flag = "high volatility"

    return {
        "p10_minutes": p10,
        "p50_minutes": p50,
        "p90_minutes": p90,
        "interval_width_minutes": interval_width,
        "volatility_flag": volatility_flag or None,
    }


def render_duration_prediction_card(
    prediction: Dict[str, Any],
    volatility_thresholds_minutes: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a reusable duration result card.

    UI meaning:
    - p50 = headline duration forecast
    - p10-p90 = likely pre-game range
    - volatility badge = uncertainty classification from interval width
    """

    resolved = _resolve_duration_prediction(
        prediction=prediction,
        volatility_thresholds_minutes=volatility_thresholds_minutes,
    )
    headline_minutes = resolved["p50_minutes"]
    p10 = resolved["p10_minutes"]
    p90 = resolved["p90_minutes"]
    interval_width = resolved["interval_width_minutes"]
    volatility_flag = resolved["volatility_flag"]

    headline_text = f"{headline_minutes:.1f} min" if headline_minutes is not None else "Unavailable"
    range_text = (
        f"{p10:.1f} - {p90:.1f} min"
        if p10 is not None and p90 is not None
        else "Quantile range unavailable"
    )
    width_text = f"Width {interval_width:.1f} min" if interval_width is not None else "Point forecast only"
    badge_text = volatility_flag.title() if volatility_flag else "Standard forecast"
    badge_class = _volatility_css_class(volatility_flag or "")

    with st.container(border=True):
        top_left, top_mid, top_right = st.columns([1.7, 1.15, 0.9])

        with top_left:
            st.markdown(
                f"""
                <div class="oo-result-block oo-result-headline-block">
                    <div class="oo-result-label">Predicted Duration</div>
                    <div class="oo-result-value">{html.escape(headline_text)}</div>
                    <div class="oo-result-subvalue">p50 is the model's headline duration forecast.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with top_mid:
            st.markdown(
                f"""
                <div class="oo-result-block">
                    <div class="oo-result-label">Likely Range</div>
                    <div class="oo-result-secondary">{html.escape(range_text)}</div>
                    <div class="oo-result-subvalue">{html.escape(width_text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with top_right:
            badge_body = (
                f'<span class="oo-volatility-pill {badge_class}">{html.escape(badge_text)}</span>'
                if volatility_flag
                else '<span class="oo-volatility-pill oo-volatility-neutral">Range pending</span>'
            )
            st.markdown(
                f"""
                <div class="oo-result-block oo-result-badge-block">
                    <div class="oo-result-label">Volatility</div>
                    <div class="oo-result-badge-wrap">{badge_body}</div>
                    <div class="oo-result-subvalue">Wider ranges imply more pre-game uncertainty.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if p10 is not None and headline_minutes is not None and p90 is not None and p90 > p10:
            marker_pct = max(0.0, min(100.0, ((headline_minutes - p10) / (p90 - p10)) * 100.0))
            st.markdown(
                f"""
                <div class="oo-range-shell">
                    <div class="oo-range-title">Likely Game-Length Window</div>
                    <div class="oo-range-bar">
                        <div class="oo-range-track"></div>
                        <div class="oo-range-fill"></div>
                        <div class="oo-range-marker" style="left: {marker_pct:.2f}%;"></div>
                    </div>
                    <div class="oo-range-labels">
                        <span>P10 {p10:.1f}m</span>
                        <span>P50 {headline_minutes:.1f}m</span>
                        <span>P90 {p90:.1f}m</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        meta_items = []
        if prediction.get("match_label"):
            meta_items.append(f"Match: {prediction['match_label']}")
        if prediction.get("league"):
            meta_items.append(f"League: {prediction['league']}")
        if prediction.get("model_name"):
            meta_items.append(f"Model: {prediction['model_name']}")
        if prediction.get("prediction_timestamp_utc"):
            meta_items.append(f"Updated: {prediction['prediction_timestamp_utc']}")

        if meta_items:
            chips = "".join([f'<span class="oo-meta-chip">{html.escape(item)}</span>' for item in meta_items])
            st.markdown(f'<div class="oo-meta-row">{chips}</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="oo-result-note">
                p50 is the most likely duration forecast. The likely range shows the model's uncertainty before the game starts.
                Higher volatility means a wider range of plausible game lengths.
            </div>
            """,
            unsafe_allow_html=True,
        )


def status_label(status: str) -> str:
    token = str(status or "").strip().lower()
    if token == "won":
        return "Won"
    if token == "lost":
        return "Lost"
    if token == "push":
        return "Push"
    return "Pending"
