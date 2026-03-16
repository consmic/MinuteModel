from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st

FREE_PREDICTIONS_PER_MONTH = 25

JOURNAL_COLUMNS = [
    "entry_id",
    "created_at_utc",
    "game",
    "league",
    "match_label",
    "prediction_type",
    "predicted_value",
    "confidence_label",
    "model_name",
    "status",
    "odds_decimal",
    "stake_units",
    "profit_units",
    "notes",
]


def ensure_form_state(defaults: Dict[str, Any]) -> None:
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def ensure_usage_state(free_allowance: int = FREE_PREDICTIONS_PER_MONTH) -> None:
    current_month = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")

    if "usage_month" not in st.session_state:
        st.session_state["usage_month"] = current_month
    if "usage_used" not in st.session_state:
        st.session_state["usage_used"] = 0
    if "usage_allowance" not in st.session_state:
        st.session_state["usage_allowance"] = int(free_allowance)

    if st.session_state["usage_month"] != current_month:
        st.session_state["usage_month"] = current_month
        st.session_state["usage_used"] = 0


def get_usage_snapshot() -> Dict[str, int | str]:
    allowance = int(st.session_state.get("usage_allowance", FREE_PREDICTIONS_PER_MONTH))
    used = int(st.session_state.get("usage_used", 0))
    remaining = max(allowance - used, 0)
    month = str(st.session_state.get("usage_month", pd.Timestamp.now(tz="UTC").strftime("%Y-%m")))
    return {
        "allowance": allowance,
        "used": used,
        "remaining": remaining,
        "month": month,
    }


def increment_usage() -> None:
    st.session_state["usage_used"] = int(st.session_state.get("usage_used", 0)) + 1


def reset_usage() -> None:
    st.session_state["usage_used"] = 0


def load_journal(path: str | Path) -> pd.DataFrame:
    journal_path = Path(path)
    if not journal_path.exists():
        return pd.DataFrame(columns=JOURNAL_COLUMNS)

    df = pd.read_csv(journal_path, low_memory=False)
    for col in JOURNAL_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df = df[JOURNAL_COLUMNS].copy()

    if "created_at_utc" in df.columns:
        df["created_at_utc"] = pd.to_datetime(df["created_at_utc"], errors="coerce", utc=False)

    for col in ["odds_decimal", "stake_units", "profit_units"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def save_journal(path: str | Path, journal_df: pd.DataFrame) -> None:
    journal_path = Path(path)
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    out = journal_df.copy()
    for col in JOURNAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[JOURNAL_COLUMNS]
    out.to_csv(journal_path, index=False)


def append_journal_entry(path: str | Path, entry: Dict[str, Any]) -> pd.DataFrame:
    journal = load_journal(path)
    entry_row = pd.DataFrame([entry])
    updated = pd.concat([journal, entry_row], axis=0, ignore_index=True)
    save_journal(path, updated)
    return updated


def compute_journal_metrics(journal_df: pd.DataFrame) -> Dict[str, float | int]:
    if journal_df.empty:
        return {
            "total_picks": 0,
            "settled_picks": 0,
            "win_rate": 0.0,
            "roi": 0.0,
            "profit_units": 0.0,
            "avg_odds": 0.0,
        }

    work = journal_df.copy()
    work["status"] = work["status"].fillna("Pending").astype(str)
    work["odds_decimal"] = pd.to_numeric(work.get("odds_decimal"), errors="coerce")
    work["stake_units"] = pd.to_numeric(work.get("stake_units"), errors="coerce")

    profit = []
    for _, row in work.iterrows():
        status = str(row.get("status", "Pending")).strip().lower()
        odds = float(row.get("odds_decimal")) if pd.notna(row.get("odds_decimal")) else np.nan
        stake = float(row.get("stake_units")) if pd.notna(row.get("stake_units")) else np.nan

        if not np.isfinite(stake) or stake <= 0:
            profit.append(np.nan)
            continue

        if status == "won" and np.isfinite(odds) and odds > 1.0:
            profit.append(stake * (odds - 1.0))
        elif status == "lost":
            profit.append(-stake)
        elif status == "push":
            profit.append(0.0)
        else:
            profit.append(np.nan)

    work["profit_units"] = profit

    settled = work[work["status"].str.lower().isin(["won", "lost", "push"])].copy()
    win_loss_only = work[work["status"].str.lower().isin(["won", "lost"])].copy()

    total_staked = pd.to_numeric(settled["stake_units"], errors="coerce").fillna(0.0).sum()
    total_profit = pd.to_numeric(settled["profit_units"], errors="coerce").fillna(0.0).sum()

    wins = int((win_loss_only["status"].str.lower() == "won").sum())
    losses = int((win_loss_only["status"].str.lower() == "lost").sum())
    denom = wins + losses
    win_rate = float(wins / denom) if denom > 0 else 0.0

    roi = float(total_profit / total_staked) if total_staked > 0 else 0.0
    avg_odds = float(pd.to_numeric(settled["odds_decimal"], errors="coerce").dropna().mean()) if not settled.empty else 0.0

    return {
        "total_picks": int(len(work)),
        "settled_picks": int(len(settled)),
        "win_rate": win_rate,
        "roi": roi,
        "profit_units": float(total_profit),
        "avg_odds": avg_odds,
    }
