from __future__ import annotations

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from src.models.interfaces import load_task_definitions


def create_tokenizer(config: dict):
    model_name_or_path = config["model"]["model_name_or_path"]
    return AutoTokenizer.from_pretrained(model_name_or_path)


def create_model(config: dict, label2id: dict[str, int], id2label: dict[int, str]):
    task_definitions = load_task_definitions(config)

    if len(task_definitions) > 1:
        # 当前 baseline 只支持“一个编码器 + 一个分类头”，多任务头留到后续版本。
        raise NotImplementedError(
            "Multi-task classification heads are reserved for later versions. "
            "Current baseline supports one active task."
        )

    task = task_definitions[0]
    model_name_or_path = config["model"]["model_name_or_path"]

    # 这里创建的是标准 Hugging Face SequenceClassification 结构：
    # 预训练编码器 + 单个分类头。
    hf_config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=len(label2id),
        label2id=label2id,
        id2label={str(key): value for key, value in id2label.items()},
        problem_type="multi_label_classification"
        if task.task_type == "multi_label_classification"
        else "single_label_classification",
    )

    if task.task_type not in {"single_label_classification", "multi_label_classification"}:
        raise NotImplementedError(f"Unsupported task type: {task.task_type}")

    return AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=hf_config,
    )
