from __future__ import annotations

import argparse

from src.data.io import read_table, write_table
from src.utils.config import load_config, resolve_path
from src.utils.decision_support import get_category_model_settings
from src.utils.feature_matrix import build_feature_matrix, resolve_feature_matrix_output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a composite feature matrix with CLS vectors.")
    parser.add_argument("--config", type=str, default="configs/real_problem_level_v1.yaml", help="Level-task config.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the saved level model.")
    parser.add_argument("--input-file", type=str, required=True, help="Input CSV/JSONL/XLSX file.")
    parser.add_argument("--output-file", type=str, default=None, help="Output CSV path.")
    parser.add_argument("--category-config", type=str, default=None, help="Optional category-task config.")
    parser.add_argument("--category-model-path", type=str, default=None, help="Optional category model path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    category_config, category_model = get_category_model_settings(config)
    input_path = resolve_path(args.input_file)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    feature_matrix = build_feature_matrix(
        input_df=read_table(input_path),
        level_config=config,
        level_model_path=args.model_path,
        category_config_path=args.category_config or category_config,
        category_model_path=args.category_model_path or category_model,
    )
    output_path = resolve_feature_matrix_output_path(args.output_file)
    write_table(feature_matrix, output_path)
    print(f"Feature matrix saved to: {output_path}")
    print(f"Rows exported: {len(feature_matrix)}")


if __name__ == "__main__":
    main()
