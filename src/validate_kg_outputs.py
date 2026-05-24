from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import read_table


REQUIRED_FIELDS = [
    "related_standard_name",
    "related_clause_text",
    "standard_match_confidence",
    "problem_standard_relation_type",
    "revision_need_type",
    "standard_revision_priority_initial",
    "graph_relation_type",
]

VALID_RELATION_TYPES = {
    "standard_execution",
    "standard_missing",
    "standard_lagging",
    "standard_ambiguous",
    "standard_conflict",
    "manual_review",
    "unmatched",
}
VALID_REVISION_TYPES = {"执行落实问题", "标准缺失", "标准表述歧义", "标准冲突", "适用性不足", "需人工判断"}
VALID_PRIORITIES = {"high", "medium", "low", "高", "中", "低"}
VALID_GRAPH_RELATIONS = {"PROBLEM_MATCHES_CLAUSE", "UNMATCHED", ""}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate KG enrichment output quality.")
    parser.add_argument("--input", type=str, default="outputs/review_sheet_with_kg.jsonl", help="CSV/JSONL/XLSX output to validate.")
    parser.add_argument("--warning-threshold", type=float, default=0.5, help="Missing clause warning threshold.")
    return parser.parse_args()


def _missing_rate(series: pd.Series) -> float:
    if len(series) == 0:
        return 0.0
    return float(series.fillna("").astype(str).str.strip().eq("").mean())


def validate_kg_output(path: str | Path, warning_threshold: float = 0.5) -> dict[str, object]:
    dataframe = read_table(path)
    missing_fields = [field for field in REQUIRED_FIELDS if field not in dataframe.columns]

    for field in missing_fields:
        dataframe[field] = ""

    confidence = pd.to_numeric(dataframe["standard_match_confidence"], errors="coerce")
    invalid_confidence = int((confidence.isna() | confidence.lt(0) | confidence.gt(1)).sum())
    invalid_relation_type = int((~dataframe["problem_standard_relation_type"].fillna("").astype(str).isin(VALID_RELATION_TYPES)).sum())
    invalid_revision_need_type = int((~dataframe["revision_need_type"].fillna("").astype(str).isin(VALID_REVISION_TYPES)).sum())
    invalid_priority = int((~dataframe["standard_revision_priority_initial"].fillna("").astype(str).isin(VALID_PRIORITIES)).sum())
    invalid_graph_relation_type = int((~dataframe["graph_relation_type"].fillna("").astype(str).isin(VALID_GRAPH_RELATIONS)).sum())

    standard_missing_rate = _missing_rate(dataframe["related_standard_name"])
    clause_missing_rate = _missing_rate(dataframe["related_clause_text"])
    status = "PASS"
    warnings: list[str] = []
    if missing_fields:
        warnings.append(f"missing fields: {', '.join(missing_fields)}")
        status = "WARNING"
    if clause_missing_rate > warning_threshold:
        warnings.append(f"related_clause_text missing rate {clause_missing_rate:.2%} exceeds {warning_threshold:.0%}")
        status = "WARNING"

    return {
        "total_records": len(dataframe),
        "missing_related_standard_name_rate": standard_missing_rate,
        "missing_related_clause_text_rate": clause_missing_rate,
        "invalid_confidence": invalid_confidence,
        "invalid_relation_type": invalid_relation_type,
        "invalid_revision_need_type": invalid_revision_need_type,
        "invalid_priority": invalid_priority,
        "invalid_graph_relation_type": invalid_graph_relation_type,
        "status": status,
        "warnings": warnings,
    }


def main() -> None:
    args = parse_args()
    result = validate_kg_output(args.input, warning_threshold=args.warning_threshold)
    print("[KG Output Validation]")
    print(f"Total records: {result['total_records']}")
    print(f"Missing related_standard_name: {result['missing_related_standard_name_rate']:.2%}")
    print(f"Missing related_clause_text: {result['missing_related_clause_text_rate']:.2%}")
    print(f"Invalid confidence: {result['invalid_confidence']}")
    print(f"Invalid relation_type: {result['invalid_relation_type']}")
    print(f"Invalid revision_need_type: {result['invalid_revision_need_type']}")
    print(f"Invalid priority: {result['invalid_priority']}")
    print(f"Invalid graph_relation_type: {result['invalid_graph_relation_type']}")
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")
    print(f"Status: {result['status']}")


if __name__ == "__main__":
    main()
