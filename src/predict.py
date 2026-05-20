from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.io import read_table, write_table
from src.utils.config import load_config, resolve_path
from src.utils.decision_support import build_enriched_predictions, get_category_model_settings
from src.utils.feedback import attach_feedback_columns, get_prediction_export_config
from src.utils.model_runtime import load_model_runtime, predict_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prediction for technical supervision issues.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to saved model directory.")
    parser.add_argument("--text", type=str, default=None, help="Single text for prediction.")
    parser.add_argument(
        "--extra-fields-json",
        type=str,
        default=None,
        help="Optional JSON object for structured fields in single-text prediction.",
    )
    parser.add_argument("--input-file", type=str, default=None, help="Batch input file, CSV or JSONL.")
    parser.add_argument("--output-file", type=str, default=None, help="Batch output file, CSV or JSONL.")
    parser.add_argument("--category-config", type=str, default=None, help="Optional category-task config.")
    parser.add_argument("--category-model-path", type=str, default=None, help="Optional category model path.")
    return parser.parse_args()


def resolve_output_path(config: dict, output_file: str | None, input_file: str | None) -> Path:
    if output_file:
        return resolve_path(output_file)

    output_dir = resolve_path(config["predict"]["output_dir"])
    suffix = ".csv"
    if input_file and Path(input_file).suffix.lower() == ".jsonl":
        suffix = ".jsonl"
    return Path(output_dir) / f"predictions{suffix}"


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if not args.text and not args.input_file:
        raise ValueError("Provide either --text or --input-file.")

    text_field = config["task"]["text_field"]
    task_type = config["task"]["type"]
    if task_type != "single_label_classification":
        raise NotImplementedError("Prediction baseline currently runs single-label tasks by default.")

    feedback_config = get_prediction_export_config(config)
    runtime = load_model_runtime(config, args.model_path)
    category_config, category_model_path = get_category_model_settings(config)
    category_config = args.category_config or category_config
    category_model_path = args.category_model_path or category_model_path

    if args.text:
        extra_fields = {}
        if args.extra_fields_json:
            extra_fields = json.loads(args.extra_fields_json)
            if not isinstance(extra_fields, dict):
                raise ValueError("--extra-fields-json must be a JSON object.")

        single_record = {text_field: args.text, **extra_fields}
        single_output = pd.DataFrame([single_record])
        single_output, predictions, _ = predict_dataframe(single_output, runtime=runtime)
        result = predictions[0]
        single_output["extra_fields"] = json.dumps(extra_fields, ensure_ascii=False)
        single_output["predicted_label_id"] = result["predicted_label_id"]
        single_output["predicted_label"] = result["predicted_label"]
        single_output["confidence"] = result["confidence"]
        single_output = attach_feedback_columns(
            dataframe=single_output,
            model_path=str(runtime.model_path),
            config=config,
            include_model_input=bool(feedback_config.get("include_model_input", True)),
        )
        single_output = build_enriched_predictions(
            dataframe=single_output,
            config=config,
            category_config_path=category_config,
            category_model_path=category_model_path,
            device=runtime.device,
        )
        print(
            json.dumps(
                {
                    **single_output.iloc[0].to_dict(),
                    "extra_fields": extra_fields,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
        return

    input_path = resolve_path(args.input_file)
    dataframe = read_table(input_path)
    if text_field not in dataframe.columns:
        raise ValueError(f"Missing text field '{text_field}' in batch input.")

    dataframe, predictions, _ = predict_dataframe(dataframe, runtime=runtime)
    dataframe["predicted_label_id"] = [item["predicted_label_id"] for item in predictions]
    dataframe["predicted_label"] = [item["predicted_label"] for item in predictions]
    dataframe["confidence"] = [item["confidence"] for item in predictions]
    dataframe = attach_feedback_columns(
        dataframe=dataframe,
        model_path=str(runtime.model_path),
        config=config,
        include_model_input=bool(feedback_config.get("include_model_input", True)),
    )
    dataframe = build_enriched_predictions(
        dataframe=dataframe,
        config=config,
        category_config_path=category_config,
        category_model_path=category_model_path,
        device=runtime.device,
    )

    output_path = resolve_output_path(config, args.output_file, args.input_file)
    write_table(dataframe, output_path)
    print(f"Batch prediction saved to: {output_path}")


if __name__ == "__main__":
    main()
