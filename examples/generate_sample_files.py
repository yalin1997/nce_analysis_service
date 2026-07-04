"""Regenerate the CLI-demo fixture files (sample_data.csv, sample_data.parquet,
sample_config.yaml) from sample_data.build_sample_dataframe().

Run this after changing examples/sample_data.py to refresh the fixtures;
the fixtures themselves are committed so the CLI quick-start works without
running this script first.

Usage:
    python3.12 examples/generate_sample_files.py
"""

from pathlib import Path

import pandas as pd
from sample_data import build_sample_dataframe

EXAMPLES_DIR = Path(__file__).parent


def main() -> None:
    raw_df = build_sample_dataframe()

    raw_df.to_parquet(EXAMPLES_DIR / "sample_data.parquet")

    long_df = raw_df.explode("Measurement_Points", ignore_index=True)
    points = pd.json_normalize(long_df["Measurement_Points"])
    long_df = pd.concat(
        [long_df.drop(columns=["Measurement_Points"]).reset_index(drop=True), points],
        axis=1,
    )
    long_df.to_csv(EXAMPLES_DIR / "sample_data.csv", index=False)

    (EXAMPLES_DIR / "sample_config.yaml").write_text(
        "# Overrides AnalysisConfig defaults; any field not listed here keeps\n"
        "# its default value (see nce_analysis/config.py).\n"
        "summary_top_n: 3\n"
    )

    print(f"Wrote {EXAMPLES_DIR / 'sample_data.csv'}")
    print(f"Wrote {EXAMPLES_DIR / 'sample_data.parquet'}")
    print(f"Wrote {EXAMPLES_DIR / 'sample_config.yaml'}")


if __name__ == "__main__":
    main()
