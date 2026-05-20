from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from src.data.dataset_builder import prepare_data_for_training
from src.utils.common import save_json
from src.utils.config import load_config, resolve_path
from src.utils.metrics import build_compute_metrics
from src.utils.reporting import export_single_label_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained baseline model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model directory.")
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
        help="Dataset split for evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model_path = resolve_path(args.model_path)

    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    prepared = prepare_data_for_training(config, tokenizer)

    with TemporaryDirectory(prefix="eval_tmp_", dir=str(Path(model_path).parent)) as temp_dir:
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=temp_dir,
                per_device_eval_batch_size=int(config["train"].get("per_device_eval_batch_size", 8)),
                report_to=[],
            ),
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=build_compute_metrics(
                task_type=config["task"]["type"],
                threshold=float(config.get("predict", {}).get("threshold", 0.5)),
            ),
        )

        prediction_output = trainer.predict(prepared.dataset_dict[args.split], metric_key_prefix=args.split)
    metrics = prediction_output.metrics
    save_json(metrics, Path(model_path) / f"{args.split}_metrics_reloaded.json")

    if config["task"]["type"] == "single_label_classification":
        export_single_label_analysis(
            predictions=prediction_output.predictions,
            label_ids=prediction_output.label_ids,
            id2label=prepared.id2label,
            output_dir=model_path,
            split_name=args.split,
        )

    print(f"{args.split} metrics: {metrics}")


if __name__ == "__main__":
    main()
