"""Export combined analysis tables to CSV (see plan: decisions checkpoints, behavioral, intervention summaries)."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import duckdb
import pandas as pd

from scripts import results_pipeline
from scripts.utils import constants

HICP_FILENAME = "France HICP_20251112134158.csv"


def _write_large(df, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def _write_summary(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=True, encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export combined results tables to CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "csv_export" / "combined",
        help="Directory for CSV files (created if missing).",
    )
    parser.add_argument(
        "--skip-static",
        action="store_true",
        help="Do not copy France HICP CSV into output dir.",
    )
    args = parser.parse_args(argv)

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    con_exp_1 = duckdb.connect(constants.EXP_1_DATABASE_FILE, read_only=False)
    con_exp_2 = duckdb.connect(constants.EXP_2_DATABASE_FILE, read_only=False)
    try:
        bundle = results_pipeline.build_export_bundle(con_exp_1, con_exp_2)
    finally:
        con_exp_1.close()
        con_exp_2.close()

    _write_large(bundle.demographics_combined, out / "demographics_combined.csv")
    _write_large(bundle.inflation_estimates_combined, out / "inflation_estimates_combined.csv")
    _write_large(
        bundle.decisions_pre_classify_behavioral_patterns,
        out / "decisions_pre_classify_behavioral_patterns.csv",
    )
    _write_large(bundle.decisions_all_enriched, out / "decisions_all_enriched.csv")
    _write_large(bundle.knowledge_combined, out / "knowledge_combined.csv")
    _write_large(bundle.econ_preferences_combined, out / "econ_preferences_combined.csv")
    _write_large(bundle.behavioral_combined, out / "behavioral_combined.csv")

    _write_summary(bundle.learning_effect_initial, out / "learning_effect_initial.csv")
    _write_summary(bundle.learning_effect_complete, out / "learning_effect_complete.csv")
    _write_summary(bundle.treatment_effect, out / "treatment_effect.csv")

    data_dir = results_pipeline.default_data_dir()
    hicp_src = data_dir / HICP_FILENAME
    if not args.skip_static:
        if hicp_src.is_file():
            shutil.copy2(hicp_src, out / HICP_FILENAME)
        else:
            print(f"Warning: optional static file not found, skipping copy: {hicp_src}")


if __name__ == "__main__":
    main()
