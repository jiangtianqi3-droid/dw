from __future__ import annotations

import argparse

from src.data.io import read_table, write_table
from src.utils.config import load_config, resolve_path
from src.utils.feedback import append_feedback_samples, build_retrain_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare reviewed feedback data for retraining.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument(
        "--feedback-file",
        type=str,
        required=True,
        help="Prediction export file after manual review, CSV or JSONL.",
    )
    parser.add_argument(
        "--original-file",
        type=str,
        default=None,
        help="Optional original labeled file to overlay reviewed labels onto.",
    )
    parser.add_argument(
        "--base-train-file",
        type=str,
        default=None,
        help="Optional existing training file to append reviewed samples to.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for retraining data. Defaults to feedback.retrain.output_file in config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    feedback_path = resolve_path(args.feedback_file)
    if feedback_path is None or not feedback_path.exists():
        raise FileNotFoundError(f"Feedback file not found: {args.feedback_file}")

    retrain_output = args.output_file or config.get("feedback", {}).get("retrain", {}).get(
        "output_file",
        "artifacts/feedback/retrain_dataset.csv",
    )
    output_path = resolve_path(retrain_output)
    label_field = config["task"]["label_field"]

    feedback_df = read_table(feedback_path)
    original_df = read_table(resolve_path(args.original_file)) if args.original_file else feedback_df.copy()

    retrain_df = build_retrain_dataset(
        original_df=original_df,
        feedback_df=feedback_df,
        config=config,
        label_field=label_field,
    )

    if args.base_train_file:
        base_train_path = resolve_path(args.base_train_file)
        if base_train_path is None or not base_train_path.exists():
            raise FileNotFoundError(f"Base training file not found: {args.base_train_file}")
        base_train_df = read_table(base_train_path)
        retrain_df = append_feedback_samples(base_train_df, retrain_df, config)

    write_table(retrain_df, output_path)
    print(f"Retraining data saved to: {output_path}")
    print(f"Prepared samples: {len(retrain_df)}")


if __name__ == "__main__":
    main()
