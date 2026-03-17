from __future__ import annotations

import html
from typing import Iterable, Optional

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


def status_label(status: str) -> str:
    token = str(status or "").strip().lower()
    if token == "won":
        return "Won"
    if token == "lost":
        return "Lost"
    if token == "push":
        return "Push"
    return "Pending"
