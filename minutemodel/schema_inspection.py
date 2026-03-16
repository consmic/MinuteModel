from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .utils import normalize_position, normalize_side


@dataclass
class SchemaInspectionReport:
    n_rows: int
    n_columns: int
    n_games: int
    rows_per_game_quantiles: Dict[str, float]
    rows_by_side: Dict[str, int]
    rows_by_position: Dict[str, int]
    participantid_distribution: Dict[str, int]
    has_team_rows: bool
    has_player_rows: bool
    draft_duplication_rate_by_column: Dict[str, float]
    gamelength_unit_guess: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "n_games": self.n_games,
            "rows_per_game_quantiles": self.rows_per_game_quantiles,
            "rows_by_side": self.rows_by_side,
            "rows_by_position": self.rows_by_position,
            "participantid_distribution": self.participantid_distribution,
            "has_team_rows": self.has_team_rows,
            "has_player_rows": self.has_player_rows,
            "draft_duplication_rate_by_column": self.draft_duplication_rate_by_column,
            "gamelength_unit_guess": self.gamelength_unit_guess,
        }


DRAFT_COLUMNS: List[str] = [
    "champion",
    "ban1",
    "ban2",
    "ban3",
    "ban4",
    "ban5",
    "pick1",
    "pick2",
    "pick3",
    "pick4",
    "pick5",
]


def infer_gamelength_unit(gamelength: pd.Series) -> str:
    values = pd.to_numeric(gamelength, errors="coerce").dropna()
    if values.empty:
        return "unknown"

    median_value = float(values.median())
    if median_value > 600:
        return "seconds"
    if 10 <= median_value <= 80:
        return "minutes"
    return "unknown"


def _distribution_as_dict(series: pd.Series, top_n: int = 20) -> Dict[str, int]:
    counts = series.value_counts(dropna=False).head(top_n)
    return {str(idx): int(value) for idx, value in counts.items()}


def _draft_duplication_rate(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns or "gameid" not in df.columns or "side" not in df.columns:
        return 0.0

    grouped = (
        df[["gameid", "side", column]]
        .dropna(subset=[column])
        .assign(side=lambda x: x["side"].map(normalize_side))
        .dropna(subset=["side"])
        .groupby(["gameid", "side"])[column]
        .nunique(dropna=True)
    )
    if grouped.empty:
        return 0.0
    return float((grouped > 1).mean())


def inspect_schema(df: pd.DataFrame) -> SchemaInspectionReport:
    rows_per_game = df.groupby("gameid").size()
    quantiles = rows_per_game.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
    rows_per_game_quantiles = {
        "min": float(quantiles.loc[0.0]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "max": float(quantiles.loc[1.0]),
    }

    side_series = df["side"].map(normalize_side) if "side" in df.columns else pd.Series([], dtype="object")
    rows_by_side = _distribution_as_dict(side_series.fillna("unknown"))

    if "position" in df.columns:
        position_series = df["position"].map(normalize_position).fillna("unknown")
    else:
        position_series = pd.Series(["missing"] * len(df))
    rows_by_position = _distribution_as_dict(position_series)

    if "participantid" in df.columns:
        participantid_distribution = _distribution_as_dict(df["participantid"].astype("Int64"))
    else:
        participantid_distribution = {"missing": len(df)}

    has_team_rows = bool((position_series == "team").any())
    has_player_rows = bool(position_series.isin(["top", "jng", "mid", "bot", "sup"]).any())

    duplication = {col: _draft_duplication_rate(df, col) for col in DRAFT_COLUMNS}

    return SchemaInspectionReport(
        n_rows=int(len(df)),
        n_columns=int(len(df.columns)),
        n_games=int(df["gameid"].nunique(dropna=True)),
        rows_per_game_quantiles=rows_per_game_quantiles,
        rows_by_side=rows_by_side,
        rows_by_position=rows_by_position,
        participantid_distribution=participantid_distribution,
        has_team_rows=has_team_rows,
        has_player_rows=has_player_rows,
        draft_duplication_rate_by_column=duplication,
        gamelength_unit_guess=infer_gamelength_unit(df["gamelength"]),
    )


def format_schema_report(report: SchemaInspectionReport) -> str:
    lines = [
        "Schema Inspection Report",
        "========================",
        f"Rows: {report.n_rows}",
        f"Columns: {report.n_columns}",
        f"Unique games: {report.n_games}",
        f"Rows per game quantiles: {report.rows_per_game_quantiles}",
        f"Rows by side: {report.rows_by_side}",
        f"Rows by position: {report.rows_by_position}",
        f"Participant ID distribution (top): {report.participantid_distribution}",
        f"Has team rows: {report.has_team_rows}",
        f"Has player rows: {report.has_player_rows}",
        f"Draft duplication rates: {report.draft_duplication_rate_by_column}",
        f"Gamelength unit guess: {report.gamelength_unit_guess}",
    ]
    return "\n".join(lines)