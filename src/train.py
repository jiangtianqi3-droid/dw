from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

from src.data.dataset_builder import prepare_data_for_training
from src.models.factory import create_model, create_tokenizer
from src.utils.common import ensure_dir, save_json, set_global_seed
from src.utils.config import load_config, resolve_path, save_yaml
from src.utils.labels import save_label_mapping
from src.utils.metrics import build_compute_metrics
from src.utils.reporting import export_single_label_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a baseline issue classifier.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config.")
    return parser.parse_args()


def build_training_arguments(config: dict) -> TrainingArguments:
    train_config = config["train"]
    signature = inspect.signature(TrainingArguments.__init__)
    # 兼容不同版本 Transformers 的参数差异，尽量让同一份配置可复用。
    kwargs = {
        "output_dir": str(resolve_path(train_config["output_dir"])),
        "logging_dir": str(resolve_path(train_config["logging_dir"])),
        "num_train_epochs": float(train_config.get("num_train_epochs", 1)),
        "per_device_train_batch_size": int(train_config.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(train_config.get("per_device_eval_batch_size", 8)),
        "gradient_accumulation_steps": int(train_config.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(train_config.get("learning_rate", 2e-5)),
        "weight_decay": float(train_config.get("weight_decay", 0.0)),
        "save_strategy": str(train_config.get("save_strategy", "epoch")),
        "save_total_limit": int(train_config.get("save_total_limit", 2)),
        "load_best_model_at_end": bool(train_config.get("load_best_model_at_end", True)),
        "greater_is_better": bool(train_config.get("greater_is_better", True)),
        "logging_steps": int(train_config.get("logging_steps", 10)),
        "seed": int(train_config.get("seed", 42)),
        "fp16": bool(train_config.get("fp16", False)),
        "dataloader_pin_memory": bool(train_config.get("dataloader_pin_memory", True)),
        "report_to": train_config.get("report_to", []),
    }

    if "warmup_steps" in train_config:
        kwargs["warmup_steps"] = int(train_config.get("warmup_steps", 0))
    else:
        kwargs["warmup_ratio"] = float(train_config.get("warmup_ratio", 0.0))

    metric_for_best_model = train_config.get("metric_for_best_model", "macro_f1")
    if not str(metric_for_best_model).startswith("eval_"):
        metric_for_best_model = f"eval_{metric_for_best_model}"
    kwargs["metric_for_best_model"] = metric_for_best_model

    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = str(train_config.get("evaluation_strategy", "epoch"))
    else:
        kwargs["evaluation_strategy"] = str(train_config.get("evaluation_strategy", "epoch"))

    if "overwrite_output_dir" in signature.parameters and "overwrite_output_dir" in train_config:
        kwargs["overwrite_output_dir"] = bool(train_config.get("overwrite_output_dir", True))

    if "save_safetensors" in signature.parameters and "save_safetensors" in train_config:
        kwargs["save_safetensors"] = bool(train_config.get("save_safetensors", True))

    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config["train"].get("seed", 42))
    set_global_seed(seed)

    # 数据准备阶段会同时完成切分、预处理、标签映射和 tokenizer 编码。
    tokenizer = create_tokenizer(config)
    prepared = prepare_data_for_training(config, tokenizer)
    model = create_model(config, prepared.label2id, prepared.id2label)

    training_args = build_training_arguments(config)
    compute_metrics = build_compute_metrics(
        task_type=config["task"]["type"],
        threshold=float(config.get("predict", {}).get("threshold", 0.5)),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared.dataset_dict["train"],
        eval_dataset=prepared.dataset_dict["valid"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    output_dir = Path(training_args.output_dir)
    ensure_dir(output_dir)

    best_model_dir = output_dir / "best_model"
    trainer.save_model(best_model_dir)
    tokenizer.save_pretrained(best_model_dir)
    save_label_mapping(prepared.label2id, prepared.id2label, best_model_dir / "label_mapping.json")
    save_label_mapping(prepared.label2id, prepared.id2label, output_dir / "label_mapping.json")

    train_metrics = train_result.metrics
    valid_output = trainer.predict(prepared.dataset_dict["valid"], metric_key_prefix="valid")
    test_output = trainer.predict(prepared.dataset_dict["test"], metric_key_prefix="test")
    valid_metrics = valid_output.metrics
    test_metrics = test_output.metrics

    save_json(train_metrics, output_dir / "train_metrics.json")
    save_json(valid_metrics, output_dir / "valid_metrics.json")
    save_json(test_metrics, output_dir / "test_metrics.json")
    save_yaml(config, output_dir / "resolved_config.yaml")

    if config["task"]["type"] == "single_label_classification":
        # 单标签任务额外导出分类报告和混淆矩阵，便于直接做误差分析。
        export_single_label_analysis(
            predictions=valid_output.predictions,
            label_ids=valid_output.label_ids,
            id2label=prepared.id2label,
            output_dir=output_dir,
            split_name="valid",
        )
        export_single_label_analysis(
            predictions=test_output.predictions,
            label_ids=test_output.label_ids,
            id2label=prepared.id2label,
            output_dir=output_dir,
            split_name="test",
        )

    if trainer.state.best_model_checkpoint:
        save_json(
            {"best_model_checkpoint": trainer.state.best_model_checkpoint},
            output_dir / "best_checkpoint.json",
        )

    print("Training completed.")
    print(f"Best model saved to: {best_model_dir}")
    print(f"Validation metrics: {valid_metrics}")
    print(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
