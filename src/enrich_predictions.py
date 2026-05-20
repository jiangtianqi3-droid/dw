from __future__ import annotations

import argparse

from src.data.io import read_table, write_table
from src.utils.config import load_config, resolve_path
from src.utils.decision_support import (
    build_enriched_predictions,
    get_category_model_settings,
    resolve_enriched_output_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich existing prediction results with decision fields.")
    parser.add_argument("--config", type=str, default="configs/real_problem_level_v1.yaml", help="Path to YAML config.")
    parser.add_argument("--input-file", type=str, required=True, help="Prediction CSV/JSONL/XLSX.")
    parser.add_argument("--output-file", type=str, default=None, help="Output CSV path.")
    parser.add_argument("--category-config", type=str, default=None, help="Optional category-task config.")
    parser.add_argument("--category-model-path", type=str, default=None, help="Optional category model path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    input_path = resolve_path(args.input_file)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    category_config, category_model = get_category_model_settings(config)
    dataframe = read_table(input_path)
    enriched = build_enriched_predictions(
        dataframe=dataframe,
        config=config,
        category_config_path=args.category_config or category_config,
        category_model_path=args.category_model_path or category_model,
    )
    output_path = resolve_enriched_output_path(config, args.output_file)
    write_table(enriched, output_path)
    print(f"Enriched predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
