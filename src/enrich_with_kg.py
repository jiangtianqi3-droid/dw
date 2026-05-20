from __future__ import annotations

import argparse

from src.data.io import read_table, write_table
from src.utils.config import resolve_path
from src.utils.kg_revision import (
    build_markdown_report,
    build_standard_priority_report,
    enrich_dataframe_with_kg,
    load_kg_index,
    write_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Link model predictions to the topic-2 knowledge graph.")
    parser.add_argument("--config", type=str, default="configs/real_problem_level_v1.yaml", help="Reserved for config compatibility.")
    parser.add_argument("--input-file", type=str, required=True, help="Prediction CSV/JSONL/XLSX.")
    parser.add_argument("--kg-graph", type=str, required=True, help="Topic-2 kg_graph.json path.")
    parser.add_argument("--output-file", type=str, required=True, help="Output issue-level CSV/JSONL/XLSX.")
    parser.add_argument("--standard-report", type=str, required=True, help="Output standard priority CSV/JSONL/XLSX.")
    parser.add_argument("--markdown-report", type=str, required=True, help="Output markdown validation report.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of KG matches retained in kg_top_matches.")
    parser.add_argument("--min-score", type=float, default=0.18, help="Minimum score for accepting a KG point match.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input_file)
    kg_graph_path = resolve_path(args.kg_graph)
    output_path = resolve_path(args.output_file)
    standard_report_path = resolve_path(args.standard_report)
    markdown_report_path = resolve_path(args.markdown_report)

    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")
    if kg_graph_path is None or not kg_graph_path.exists():
        raise FileNotFoundError(f"KG graph file not found: {args.kg_graph}")
    if output_path is None or standard_report_path is None or markdown_report_path is None:
        raise ValueError("Output paths must not be empty.")

    dataframe = read_table(input_path)
    kg_index = load_kg_index(kg_graph_path)
    enriched = enrich_dataframe_with_kg(dataframe, kg_index=kg_index, top_k=args.top_k, min_score=args.min_score)
    standard_report = build_standard_priority_report(enriched)
    markdown_report = build_markdown_report(enriched, standard_report, kg_graph_path)

    write_table(enriched, output_path)
    write_table(standard_report, standard_report_path)
    write_text(markdown_report_path, markdown_report)

    matched_count = int((enriched["kg_match_status"] == "matched").sum())
    print(f"KG-linked predictions saved to: {output_path}")
    print(f"Standard priority report saved to: {standard_report_path}")
    print(f"Markdown report saved to: {markdown_report_path}")
    print(f"Rows: {len(enriched)}; matched: {matched_count}")


if __name__ == "__main__":
    main()

