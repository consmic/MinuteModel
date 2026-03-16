from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import PipelineConfig
from .data_loading import candidate_transformation_columns, filter_complete_games, load_raw_csv
from .inference import predict_single_draft
from .preprocessing import flatten_to_match_level
from .schema_inspection import format_schema_report, inspect_schema
from .train import train_and_evaluate
from .utils import setup_logging


def _load_config(config_path: str | None, input_csv: str | None = None) -> PipelineConfig:
    if config_path:
        cfg = PipelineConfig.from_yaml(config_path)
    else:
        if not input_csv:
            raise ValueError("Either config path or input_csv must be provided.")
        cfg = PipelineConfig(input_csv=input_csv)
    cfg.validate()
    return cfg


def cmd_inspect(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config, input_csv=args.input_csv)

    raw_df = load_raw_csv(cfg.input_csv)
    raw_df = filter_complete_games(raw_df)
    raw_df = raw_df[candidate_transformation_columns(raw_df)].copy()

    report = inspect_schema(raw_df)
    rendered = format_schema_report(report)
    print(rendered)

    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered, encoding="utf-8")


def cmd_flatten(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config, input_csv=args.input_csv)

    raw_df = load_raw_csv(cfg.input_csv)
    raw_df = filter_complete_games(raw_df)
    raw_df = raw_df[candidate_transformation_columns(raw_df)].copy()
    report = inspect_schema(raw_df)

    match_df = flatten_to_match_level(raw_df, config=cfg, target_unit_guess=report.gamelength_unit_guess)
    match_df.to_csv(args.output_csv, index=False)
    print(f"Wrote match-level table: {args.output_csv} ({len(match_df)} rows)")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _load_config(args.config)
    result = train_and_evaluate(cfg)
    print(json.dumps(result, indent=2))


def cmd_predict(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    pred = predict_single_draft(
        artifact_path=args.artifact_path,
        draft_payload=payload,
        include_explanation=args.explain,
    )

    rendered = json.dumps(pred, indent=2)
    print(rendered)

    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MinuteModel: draft-only LoL match duration regressor")
    parser.add_argument("--log-level", default="INFO", help="Python logging level")

    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect raw schema and row structure")
    inspect_parser.add_argument("--input-csv", default=None, help="Path to raw Oracle's Elixir CSV")
    inspect_parser.add_argument("--config", default=None, help="YAML config path")
    inspect_parser.add_argument("--output-report", default=None, help="Optional path to save text report")
    inspect_parser.set_defaults(func=cmd_inspect)

    flatten_parser = subparsers.add_parser("flatten", help="Build one-row-per-match table")
    flatten_parser.add_argument("--input-csv", default=None, help="Path to raw Oracle's Elixir CSV")
    flatten_parser.add_argument("--config", default=None, help="YAML config path")
    flatten_parser.add_argument("--output-csv", required=True, help="Output CSV path for flattened table")
    flatten_parser.set_defaults(func=cmd_flatten)

    train_parser = subparsers.add_parser("train", help="Train and evaluate full pipeline")
    train_parser.add_argument("--config", required=True, help="YAML config path")
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Run single-match inference")
    predict_parser.add_argument("--artifact-path", required=True, help="Path to model_artifacts.joblib")
    predict_parser.add_argument("--input-json", required=True, help="JSON payload with pre-game features")
    predict_parser.add_argument("--output-json", default=None, help="Optional output JSON path")
    predict_parser.add_argument("--explain", action="store_true", help="Include SHAP explanation payload")
    predict_parser.set_defaults(func=cmd_predict)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    level = getattr(logging, str(args.log_level).upper(), logging.INFO)
    setup_logging(level=level)
    args.func(args)


if __name__ == "__main__":
    main()
