from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from pandas.errors import ParserError

from .config import allowed_raw_columns

LOGGER = logging.getLogger(__name__)


MIN_REQUIRED_COLUMNS: List[str] = ["gameid", "gamelength", "date", "league", "patch", "side"]


class DataSchemaError(RuntimeError):
    """Raised when the expected Oracle's Elixir schema is missing required fields."""


def _normalize_columns(columns: Iterable[str]) -> List[str]:
    return [str(col).strip() for col in columns]


def _resolve_input_paths(path: str | Path) -> List[Path]:
    raw = str(path)
    p = Path(raw)

    if p.exists():
        if p.is_dir():
            candidates = sorted(p.glob("*.csv"))
        else:
            candidates = [p]
    else:
        candidates = [Path(match) for match in sorted(glob.glob(raw))]

    if not candidates:
        raise FileNotFoundError(f"No CSV files found for input path/pattern: {path}")

    return candidates


def _read_single_csv(csv_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except ParserError as exc:
        LOGGER.warning(
            "Strict CSV parsing failed for %s (%s). Retrying with tolerant parser (engine=python, on_bad_lines='skip').",
            csv_path,
            exc,
        )
        df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

    df.columns = _normalize_columns(df.columns)
    return df


def validate_required_columns(df: pd.DataFrame, required: Iterable[str] | None = None) -> None:
    required_cols = list(required or MIN_REQUIRED_COLUMNS)
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise DataSchemaError(f"Missing required columns: {missing}")


def load_raw_csv(path: str | Path) -> pd.DataFrame:
    csv_paths = _resolve_input_paths(path)
    LOGGER.info("Loading raw Oracle's Elixir CSV from %d file(s).", len(csv_paths))

    frames = [_read_single_csv(csv_path) for csv_path in csv_paths]
    df = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    validate_required_columns(df)
    df = coerce_types(df)

    LOGGER.info("Loaded %d rows x %d columns", len(df), len(df.columns))
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=False)

    for numeric_col in [
        "year",
        "game",
        "participantid",
        "gamelength",
        "firstPick",
        "ckpm",
        "result",
        "firstblood",
        "firstdragon",
        "dragons",
        "firstherald",
        "heralds",
        "firstbaron",
        "barons",
        "golddiffat15",
    ]:
        if numeric_col in out.columns:
            out[numeric_col] = pd.to_numeric(out[numeric_col], errors="coerce")

    if "datacompleteness" in out.columns:
        out["datacompleteness"] = out["datacompleteness"].astype(str).str.strip().str.lower()

    if "side" in out.columns:
        out["side"] = out["side"].astype(str).str.strip()

    return out


def filter_complete_games(df: pd.DataFrame) -> pd.DataFrame:
    """Prefer fully complete matches, but gracefully fall back to all rows when missing."""
    if "datacompleteness" not in df.columns:
        return df

    complete_mask = df["datacompleteness"].eq("complete")
    if complete_mask.any():
        LOGGER.info("Keeping only datacompleteness=complete rows: %d/%d", complete_mask.sum(), len(df))
        return df.loc[complete_mask].copy()

    LOGGER.warning("No rows with datacompleteness=complete found; using all rows.")
    return df


def candidate_transformation_columns(df: pd.DataFrame) -> List[str]:
    """Columns considered during raw -> match flattening."""
    candidates = sorted(set(allowed_raw_columns()).intersection(df.columns))
    LOGGER.debug("Using %d candidate transformation columns.", len(candidates))
    return candidates
