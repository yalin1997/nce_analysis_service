import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from nce_analysis import pipeline
from nce_analysis.config import AnalysisConfig
from nce_analysis.io import InputFormatError, load_input
from nce_analysis.preprocessing.base import PreprocessingError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nce_analysis",
        description="Run the NCE cross-stage root-cause analysis pipeline on a wafer batch file.",
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Input wafer batch file (.csv or .parquet)",
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="YAML config file (optional; AnalysisConfig defaults if omitted)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to write result JSON (optional; prints to stdout if omitted)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        raw_df = load_input(args.input)
    except (InputFormatError, FileNotFoundError) as e:
        print(f"Error loading input file: {e}", file=sys.stderr)
        return 1

    try:
        config = AnalysisConfig.from_yaml(args.config) if args.config else AnalysisConfig()
    except (FileNotFoundError, ValidationError) as e:
        print(f"Error loading config file: {e}", file=sys.stderr)
        return 1

    try:
        result = pipeline.run(raw_df, config)
    except PreprocessingError as e:
        print(f"Error running analysis pipeline: {e}", file=sys.stderr)
        return 1

    output_json = result.model_dump_json(indent=2)
    if args.output:
        args.output.write_text(output_json, encoding="utf-8")
    else:
        print(output_json)
    return 0
