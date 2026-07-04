"""Run the sample wafer batch through nce_analysis.pipeline end to end.

Usage:
    python3.12 examples/run_example.py

For the file-based / CLI path (CSV or Parquet input, YAML config) instead
of this in-process API call, see examples/sample_data.csv,
examples/sample_data.parquet, examples/sample_config.yaml, and the
"Quick start" section of README.md.
"""

from nce_analysis import pipeline
from nce_analysis.config import AnalysisConfig
from sample_data import build_sample_dataframe


def main() -> None:
    raw_df = build_sample_dataframe()
    config = AnalysisConfig()  # defaults are enough for this sample batch

    result = pipeline.run(raw_df, config)

    print(f"Total wafers in batch: {len(raw_df)}")
    print()

    print(f"=== Summary (top {config.summary_top_n} suspects, one entry per tool/chamber/step) ===")
    if not result.Summary:
        print("(none)")
    for detail in result.Summary:
        print(
            f"- {detail.Suspect_Pre_ToolID}/{detail.Suspect_Pre_ChamberID} "
            f"(step {detail.Suspect_Pre_StepID}): {detail.Root_Cause_Type}, "
            f"confidence={detail.Confidence_Score:.1f}, "
            f"manual_review={detail.Requires_Manual_Review}, "
            f"coords={detail.Affected_Coordinates}"
        )
    print()

    print("=== Details (every candidate found, before summary dedup) ===")
    for detail in result.Details:
        print(
            f"- {detail.Suspect_Pre_ToolID}/{detail.Suspect_Pre_ChamberID}: "
            f"{detail.Root_Cause_Type}, confidence={detail.Confidence_Score:.1f}"
        )
    print()

    print("=== Insufficient-sample hotspots (too few wafers to analyze) ===")
    if not result.Insufficient_Sample_Hotspots:
        print("(none)")
    for hotspot in result.Insufficient_Sample_Hotspots:
        print(
            f"- ({hotspot.X_Posi}, {hotspot.Y_Posi}): "
            f"{hotspot.anomalous_wafer_count}/{hotspot.total_wafer_count} wafers anomalous"
        )
    print()

    print("Full result as JSON: result.model_dump_json(indent=2)")


if __name__ == "__main__":
    main()
