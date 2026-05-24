from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import read_table, write_table
from src.utils.graph_export import build_graph_node_edge_tables
from src.utils.kg_revision import enrich_dataframe_with_kg, load_kg_index
from src.utils.standard_revision_aggregation import aggregate_standard_revision_priority


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small KG revision end-to-end smoke test.")
    parser.add_argument("--input", type=str, default="data/examples/sample_predictions.jsonl")
    parser.add_argument("--kg", type=str, default="data/kg/sample_kg_graph.json")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--min-score", type=float, default=0.12)
    return parser.parse_args()


def _print_distribution(title: str, series: pd.Series) -> None:
    print(f"{title}:")
    for value, count in series.fillna("").astype(str).value_counts().items():
        print(f"  - {value or 'empty'}: {count}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_table(args.input)
    kg_index = load_kg_index(args.kg)
    enriched = enrich_dataframe_with_kg(predictions, kg_index=kg_index, min_score=args.min_score)

    review_jsonl = output_dir / "review_sheet_with_kg.jsonl"
    review_csv = output_dir / "review_sheet_with_kg.csv"
    priority_csv = output_dir / "standard_revision_priority.csv"
    nodes_csv = output_dir / "graph_nodes.csv"
    edges_csv = output_dir / "graph_edges.csv"

    write_table(enriched, review_jsonl)
    write_table(enriched, review_csv)

    aggregation = aggregate_standard_revision_priority(enriched, top_k=50, min_problem_count=1)
    write_table(aggregation.dataframe, priority_csv)

    nodes, edges = build_graph_node_edge_tables(enriched)
    write_table(nodes, nodes_csv)
    write_table(edges, edges_csv)

    total = len(enriched)
    matched = int(enriched["related_clause_id"].fillna("").astype(str).str.strip().ne("").sum())
    match_rate = matched / total if total else 0.0

    print("[KG Revision Smoke Test]")
    print(f"Total problems: {total}")
    print(f"Matched clauses: {matched}")
    print(f"Match rate: {match_rate:.2%}")
    _print_distribution("Relation types", enriched["problem_standard_relation_type"])
    _print_distribution("Revision need types", enriched["revision_need_type"])
    _print_distribution("Revision priority", enriched["standard_revision_priority_initial"])
    print(f"Graph nodes: {len(nodes)}")
    print(f"Graph edges: {len(edges)}")
    print(f"Review JSONL: {review_jsonl}")
    print(f"Review CSV: {review_csv}")
    print(f"Standard priority CSV: {priority_csv}")
    print(f"Graph nodes CSV: {nodes_csv}")
    print(f"Graph edges CSV: {edges_csv}")


if __name__ == "__main__":
    main()
