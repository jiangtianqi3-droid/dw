from __future__ import annotations

import argparse

from src.data.io import read_table, write_table
from src.utils.config import load_config, resolve_path
from src.utils.decision_support import build_enriched_predictions, get_category_model_settings
from src.utils.graph_export import build_graph_export_dataframe, resolve_graph_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export graph-friendly issue records with explicit nodes and edges.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--input-file", type=str, required=True, help="Input CSV/JSONL/XLSX file.")
    parser.add_argument("--output-file", type=str, default=None, help="Output JSONL file.")
    parser.add_argument(
        "--label-role",
        type=str,
        default="auto",
        choices=["auto", "category", "level"],
        help="Whether the current task label should be exported as issue_category or issue_level.",
    )
    parser.add_argument("--category-config", type=str, default=None, help="Optional category-task config.")
    parser.add_argument("--category-model-path", type=str, default=None, help="Optional category model path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_path = resolve_path(args.input_file)

    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    output_path = resolve_path(str(resolve_graph_output_path(config, input_path, args.output_file)))
    dataframe = read_table(input_path)
    category_config, category_model_path = get_category_model_settings(config)
    dataframe = build_enriched_predictions(
        dataframe=dataframe,
        config=config,
        category_config_path=args.category_config or category_config,
        category_model_path=args.category_model_path or category_model_path,
    )
    exported = build_graph_export_dataframe(dataframe, config=config, label_role=args.label_role)
    write_table(exported, output_path)

    print(f"Graph export saved to: {output_path}")
    print(f"Rows exported: {len(exported)}")
    print("Structured graph fields exported: graph_nodes, graph_edges")


if __name__ == "__main__":
    main()
