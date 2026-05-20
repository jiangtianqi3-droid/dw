from __future__ import annotations

import argparse
from pathlib import Path

from src.data.io import read_table, write_table
from src.utils.standard_revision_aggregation import (
    aggregate_standard_revision_priority,
    build_standard_revision_report,
    write_json_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate standard/clause-level revision priority.")
    parser.add_argument("--input", type=str, default="outputs/predictions_kg_enriched.csv", help="Input CSV/JSONL/XLSX.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/standard_revision_priority_summary.csv",
        help="Output CSV summary.",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default="outputs/standard_revision_priority_summary.json",
        help="Output JSON summary.",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default="outputs/standard_revision_priority_report.md",
        help="Output Markdown report.",
    )
    parser.add_argument("--top-k", type=int, default=50, help="Maximum rows to export.")
    parser.add_argument("--min-problem-count", type=int, default=1, help="Minimum linked problems per group.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataframe = read_table(input_path)
    result = aggregate_standard_revision_priority(
        dataframe=dataframe,
        top_k=args.top_k,
        min_problem_count=args.min_problem_count,
    )

    output_path = Path(args.output)
    json_output_path = Path(args.json_output)
    report_output_path = Path(args.report_output)

    write_table(result.dataframe, output_path)
    write_json_summary(result, json_output_path)
    report_output_path.parent.mkdir(parents=True, exist_ok=True)
    report_output_path.write_text(build_standard_revision_report(result), encoding="utf-8")

    print(f"Standard revision priority summary saved to: {output_path}")
    print(f"JSON summary saved to: {json_output_path}")
    print(f"Markdown report saved to: {report_output_path}")
    print(f"Rows exported: {len(result.dataframe)}")


if __name__ == "__main__":
    main()
