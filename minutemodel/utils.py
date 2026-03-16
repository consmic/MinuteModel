from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_json(payload: Dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_side(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"blue", "b", "100", "left"}:
        return "Blue"
    if token in {"red", "r", "200", "right"}:
        return "Red"
    return None


def normalize_position(value: Any) -> str | None:
    if value is None:
        return None
    token = str(value).strip().lower()
    mapping = {
        "top": "top",
        "jng": "jng",
        "jungle": "jng",
        "mid": "mid",
        "middle": "mid",
        "bot": "bot",
        "adc": "bot",
        "support": "sup",
        "sup": "sup",
        "team": "team",
    }
    return mapping.get(token)


def safe_first_non_null(series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]