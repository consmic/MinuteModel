from __future__ import annotations

import streamlit as st


def apply_app_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700&display=swap');

        :root {
            --bg-base: #080b14;
            --bg-surface: #0e1422;
            --bg-surface-2: #111a2c;
            --bg-elev: #17233b;
            --line: #253554;
            --line-soft: #1c2b47;
            --text-main: #f4f8ff;
            --text-muted: #9eb0cc;
            --accent: #ff6f3d;
            --accent-2: #29c6ff;
            --good: #20d38d;
            --warn: #f4bb4a;
            --bad: #ff5f7e;
        }

        html, body, [class*="css"] {
            font-family: 'Manrope', sans-serif;
            color: var(--text-main);
        }

        h1, h2, h3, h4, h5 {
            font-family: 'Sora', sans-serif;
            letter-spacing: -0.02em;
            color: var(--text-main);
        }

        #MainMenu, footer {
            visibility: hidden;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(255, 111, 61, 0.18), transparent 36%),
                radial-gradient(circle at 92% 5%, rgba(41, 198, 255, 0.14), transparent 40%),
                linear-gradient(180deg, #070b13 0%, #0b1220 45%, #0a1324 100%);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.05rem;
            padding-bottom: 1.25rem;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0b1322 0%, #0f1a31 100%);
            border-right: 1px solid rgba(60, 84, 125, 0.45);
        }

        div[data-testid="stSidebar"] * {
            color: var(--text-main) !important;
        }

        [data-testid="stSidebarNavSeparator"] {
            border-color: rgba(80, 107, 150, 0.35) !important;
        }

        .oo-hero {
            border: 1px solid rgba(84, 112, 162, 0.45);
            border-radius: 18px;
            padding: 1.15rem 1.3rem;
            margin-bottom: 1rem;
            background:
                linear-gradient(130deg, rgba(23, 35, 61, 0.95), rgba(12, 20, 38, 0.95)),
                radial-gradient(circle at 20% 0%, rgba(255, 111, 61, 0.13), transparent 46%);
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.35);
        }

        .oo-eyebrow {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.10em;
            color: var(--accent-2);
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .oo-title {
            margin: 0;
            font-size: clamp(1.5rem, 2.2vw, 2.2rem);
        }

        .oo-subtitle {
            margin: 0.45rem 0 0.55rem 0;
            max-width: 900px;
            color: #b8c7de;
            font-size: 0.96rem;
        }

        .oo-badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.25rem;
        }

        .oo-badge {
            border-radius: 999px;
            border: 1px solid rgba(98, 130, 185, 0.5);
            color: #dfe9ff;
            background: rgba(24, 37, 63, 0.9);
            padding: 0.18rem 0.58rem;
            font-size: 0.74rem;
            font-weight: 600;
            white-space: nowrap;
        }

        .oo-section-head {
            font-family: 'Sora', sans-serif;
            font-size: 1.08rem;
            font-weight: 700;
            margin-top: 0.5rem;
            margin-bottom: 0.05rem;
            color: #f3f7ff;
        }

        .oo-section-sub {
            color: #9db2d0;
            font-size: 0.86rem;
            margin-bottom: 0.55rem;
        }

        .oo-panel {
            border-radius: 13px;
            border: 1px solid var(--line);
            background: linear-gradient(180deg, rgba(20, 31, 53, 0.98), rgba(15, 24, 41, 0.98));
            padding: 0.8rem 0.95rem;
            margin-bottom: 0.65rem;
        }

        .oo-panel-positive {
            border-color: rgba(32, 211, 141, 0.55);
        }

        .oo-panel-warning {
            border-color: rgba(244, 187, 74, 0.55);
        }

        .oo-panel-danger {
            border-color: rgba(255, 95, 126, 0.55);
        }

        .oo-panel-title {
            font-size: 0.82rem;
            color: #dce7fb;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }

        .oo-panel-body {
            color: #a6bbd9;
            font-size: 0.85rem;
        }

        .oo-empty {
            border-radius: 14px;
            border: 1px dashed rgba(90, 117, 160, 0.55);
            background: rgba(16, 25, 42, 0.72);
            padding: 1.15rem 1rem;
            text-align: center;
            margin-top: 0.45rem;
        }

        .oo-empty-title {
            font-weight: 700;
            color: #dce9ff;
            margin-bottom: 0.2rem;
        }

        .oo-empty-body {
            color: #9eb1ce;
            font-size: 0.88rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(78, 105, 149, 0.5);
            border-radius: 12px;
            padding: 0.48rem 0.65rem;
            background: linear-gradient(180deg, rgba(17, 27, 45, 0.94), rgba(14, 22, 38, 0.95));
        }

        div[data-testid="stMetricLabel"] {
            color: #9fb3cf !important;
            font-weight: 600;
        }

        div[data-testid="stMetricValue"] {
            color: #f4f8ff;
        }

        .stButton > button {
            border-radius: 11px;
            border: 1px solid rgba(111, 138, 183, 0.45);
            background: linear-gradient(180deg, #1b2943 0%, #162238 100%);
            color: #f3f7ff;
            font-weight: 700;
            min-height: 2.45rem;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.25);
        }

        .stButton > button:hover {
            border-color: rgba(255, 111, 61, 0.65);
            background: linear-gradient(180deg, #202f4b 0%, #1b2a42 100%);
            color: #ffffff;
        }

        div[data-testid="stForm"] {
            border: 1px solid rgba(75, 103, 147, 0.5);
            border-radius: 14px;
            background: linear-gradient(180deg, rgba(17, 27, 45, 0.92), rgba(12, 21, 36, 0.92));
            padding: 1rem 1rem 0.7rem 1rem;
            margin-bottom: 0.65rem;
        }

        div[data-testid="stMarkdownContainer"] p,
        label, .stCaption {
            color: #aac0dd;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] > div {
            border-radius: 10px !important;
            background: #101a2d !important;
            border-color: #314462 !important;
            color: #eff4ff !important;
        }

        .stToggle label, .stCheckbox label {
            color: #cad8ef !important;
        }

        [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        [data-baseweb="tab"] {
            border-radius: 10px 10px 0 0;
            background: rgba(17, 27, 45, 0.65);
        }

        [data-baseweb="tab"][aria-selected="true"] {
            background: rgba(27, 42, 72, 0.9);
            border-bottom: 2px solid var(--accent-2);
        }

        div[data-testid="stExpander"] {
            border: 1px solid rgba(68, 95, 139, 0.45);
            border-radius: 12px;
            background: rgba(12, 20, 34, 0.72);
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid rgba(70, 98, 140, 0.5);
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid rgba(78, 105, 149, 0.45);
            background: rgba(12, 20, 34, 0.75);
            color: #d9e7ff;
        }

        .stProgress > div > div > div > div {
            background-image: linear-gradient(90deg, var(--accent-2), var(--accent));
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
