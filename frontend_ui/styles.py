from __future__ import annotations

import streamlit as st


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        html, body, [class*="css"] {
            font-family: 'IBM Plex Sans', sans-serif;
            color: #0f172a;
        }

        h1, h2, h3, h4 {
            font-family: 'Sora', sans-serif;
            letter-spacing: -0.02em;
            color: #0b1d31;
        }

        .stApp {
            background:
                radial-gradient(circle at 10% 5%, rgba(18, 137, 167, 0.12), transparent 38%),
                radial-gradient(circle at 90% 0%, rgba(249, 115, 22, 0.10), transparent 42%),
                linear-gradient(180deg, #f6f9fc 0%, #eef4f8 100%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.0rem;
            padding-bottom: 1.25rem;
        }

        .hero-shell {
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 18px;
            padding: 1.2rem 1.35rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.98), rgba(255,255,255,0.88));
            box-shadow: 0 12px 34px rgba(15, 23, 42, 0.08);
            margin-bottom: 1rem;
        }

        .subtle-label {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #1d4ed8;
            font-weight: 700;
        }

        .app-card {
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.94);
            padding: 0.95rem 1.05rem;
            margin-bottom: 0.75rem;
        }

        .kpi-card {
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 12px;
            background: rgba(255,255,255,0.95);
            padding: 0.7rem 0.8rem;
            margin-bottom: 0.5rem;
        }

        .kpi-title {
            color: #516074;
            font-size: 0.8rem;
            margin-bottom: 0.1rem;
        }

        .kpi-value {
            color: #0f172a;
            font-weight: 700;
            font-size: 1.14rem;
        }

        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 0.2rem 0.55rem;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid rgba(15,23,42,0.14);
            background: rgba(255,255,255,0.85);
            margin-right: 0.35rem;
            margin-top: 0.25rem;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border-right: 1px solid rgba(15, 23, 42, 0.08);
        }

        div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            line-height: 1.45;
        }

        .stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(15, 23, 42, 0.15);
            font-weight: 600;
            min-height: 2.45rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.92);
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 12px;
            padding: 0.5rem 0.65rem;
        }

        div[data-testid="stMetricLabel"] {
            font-weight: 600;
        }

        div[data-testid="stForm"] {
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 14px;
            background: rgba(255, 255, 255, 0.90);
            padding: 0.95rem 1rem 0.65rem 1rem;
        }

        div[data-baseweb="select"] > div {
            border-radius: 10px !important;
        }

        textarea, input {
            border-radius: 10px !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            padding-top: 0.45rem;
            padding-bottom: 0.45rem;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(15, 23, 42, 0.10);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
