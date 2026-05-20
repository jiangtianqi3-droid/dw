from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

from src.data.io import read_table, write_table
from src.data.preprocess import InputBuilder, TextPreprocessor, get_model_input_field
from src.utils.config import resolve_path
from src.utils.common import load_json, save_json
from src.utils.labels import build_label_mappings


@dataclass
class PreparedData:
    dataset_dict: DatasetDict
    label2id: dict[str, int]
    id2label: dict[int, str]


def _compute_file_sha256(path: str | Path) -> str:
    hasher = sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _validate_columns(dataframe: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _can_stratify(labels: pd.Series) -> bool:
    if labels.nunique() < 2:
        return False
    return labels.value_counts().min() >= 2


def _split_dataframe(
    dataframe: pd.DataFrame,
    label_field: str,
    split_config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_size = float(split_config.get("train_size", 0.8))
    valid_size = float(split_config.get("valid_size", 0.1))
    test_size = float(split_config.get("test_size", 0.1))

    if round(train_size + valid_size + test_size, 6) != 1.0:
        raise ValueError("train_size + valid_size + test_size must equal 1.0")

    random_state = int(split_config.get("random_state", 42))
    stratify = bool(split_config.get("stratify", True))
    y = dataframe[label_field] if stratify and _can_stratify(dataframe[label_field]) else None

    train_df, temp_df = train_test_split(
        dataframe,
        train_size=train_size,
        random_state=random_state,
        stratify=y,
    )

    valid_ratio_in_temp = valid_size / (valid_size + test_size)
    temp_y = temp_df[label_field] if stratify and _can_stratify(temp_df[label_field]) else None

    valid_df, test_df = train_test_split(
        temp_df,
        train_size=valid_ratio_in_temp,
        random_state=random_state,
        stratify=temp_y,
    )

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _apply_preprocess(
    dataframe: pd.DataFrame,
    preprocessor: TextPreprocessor,
    input_builder: InputBuilder,
    text_field: str,
) -> pd.DataFrame:
    return input_builder.transform_dataframe(dataframe, text_field=text_field, preprocessor=preprocessor)


def _build_split_paths(split_dir: str | Path, split_file_type: str) -> dict[str, Path]:
    split_dir = Path(split_dir)
    suffix = f".{split_file_type.lstrip('.')}"
    return {
        "train": split_dir / f"train{suffix}",
        "valid": split_dir / f"valid{suffix}",
        "test": split_dir / f"test{suffix}",
    }


def _build_split_manifest(
    source_path: str | Path,
    file_type: str,
    split_config: dict[str, Any],
) -> dict[str, Any]:
    resolved_source = Path(source_path)
    return {
        "input_path": str(resolved_source),
        "file_type": file_type,
        "source_sha256": _compute_file_sha256(resolved_source),
        "split": {
            "train_size": float(split_config.get("train_size", 0.8)),
            "valid_size": float(split_config.get("valid_size", 0.1)),
            "test_size": float(split_config.get("test_size", 0.1)),
            "stratify": bool(split_config.get("stratify", True)),
            "random_state": int(split_config.get("random_state", 42)),
        },
    }


def _load_persisted_splits(
    split_paths: dict[str, Path],
    manifest_path: Path,
    source_path: str | Path,
    file_type: str,
    split_config: dict[str, Any],
) -> dict[str, pd.DataFrame] | None:
    if not all(path.exists() for path in split_paths.values()):
        return None

    rebuild_if_source_changed = bool(split_config.get("rebuild_if_source_changed", True))
    if rebuild_if_source_changed and manifest_path.exists():
        saved_manifest = load_json(manifest_path)
        current_manifest = _build_split_manifest(source_path, file_type, split_config)
        # 原始文件或切分参数变了，就强制重建 split，避免历史指标失真。
        if (
            saved_manifest.get("source_sha256") != current_manifest.get("source_sha256")
            or saved_manifest.get("split") != current_manifest.get("split")
            or saved_manifest.get("file_type") != current_manifest.get("file_type")
        ):
            return None

    return {
        split_name: read_table(path, file_type=path.suffix.lstrip("."))
        for split_name, path in split_paths.items()
    }


def _persist_splits(
    splits: dict[str, pd.DataFrame],
    split_paths: dict[str, Path],
    manifest_path: Path,
    source_path: str | Path,
    file_type: str,
    split_config: dict[str, Any],
) -> None:
    for split_name, dataframe in splits.items():
        write_table(dataframe, split_paths[split_name])

    save_json(
        _build_split_manifest(source_path, file_type, split_config),
        manifest_path,
    )


def _encode_single_label(
    dataframe: pd.DataFrame,
    label_field: str,
    label2id: dict[str, int],
) -> Dataset:
    encoded = dataframe.copy()
    # 同时保留原始标签文本和数值标签，便于训练与后续结果回查。
    encoded["label_text"] = encoded[label_field].astype(str)
    encoded["labels"] = encoded[label_field].map(label2id)

    if encoded["labels"].isna().any():
        unknown_labels = encoded.loc[encoded["labels"].isna(), label_field].unique().tolist()
        raise ValueError(f"Found labels outside mapping: {unknown_labels}")

    encoded["labels"] = encoded["labels"].astype(int)
    for column in encoded.columns:
        if column == "labels":
            continue
        encoded[column] = encoded[column].fillna("").astype(str)
    dataset = Dataset.from_pandas(encoded, preserve_index=False)
    if label_field in dataset.column_names:
        dataset = dataset.remove_columns([label_field])
    return dataset


def build_raw_splits(config: dict) -> tuple[dict[str, pd.DataFrame], dict[str, int], dict[int, str]]:
    task_config = config["task"]
    data_config = config["data"]
    preprocess_config = config.get("preprocess", {})

    text_field = task_config["text_field"]
    label_field = task_config["label_field"]
    task_type = task_config["type"]

    if task_type not in {"single_label_classification", "multi_label_classification"}:
        raise NotImplementedError(f"Unsupported task type for dataset building: {task_type}")

    preprocessor = TextPreprocessor.from_config(preprocess_config)
    input_builder = InputBuilder.from_config(config.get("input_builder", {}))

    if data_config.get("train_path") and data_config.get("valid_path") and data_config.get("test_path"):
        # 已有固定切分时直接读取，适合严格复现实验。
        train_df = read_table(resolve_path(data_config["train_path"]))
        valid_df = read_table(resolve_path(data_config["valid_path"]))
        test_df = read_table(resolve_path(data_config["test_path"]))
        _validate_columns(train_df, [text_field, label_field])
        _validate_columns(valid_df, [text_field, label_field])
        _validate_columns(test_df, [text_field, label_field])
    else:
        split_config = data_config.get("split", {})
        source_path = resolve_path(data_config["input_path"])
        file_type = data_config.get("file_type")
        split_dir = resolve_path(split_config.get("split_dir")) if split_config.get("split_dir") else None
        split_file_type = split_config.get("split_file_type", file_type or "csv")
        persisted_splits = None

        if bool(split_config.get("persist", True)) and split_dir is not None:
            split_paths = _build_split_paths(split_dir, split_file_type)
            manifest_path = Path(split_dir) / "split_manifest.json"
            persisted_splits = _load_persisted_splits(
                split_paths=split_paths,
                manifest_path=manifest_path,
                source_path=source_path,
                file_type=file_type or split_file_type,
                split_config=split_config,
            )

        if persisted_splits is not None:
            train_df = persisted_splits["train"]
            valid_df = persisted_splits["valid"]
            test_df = persisted_splits["test"]
        else:
            source_df = read_table(source_path, file_type)
            _validate_columns(source_df, [text_field, label_field])
            # 只有提供总表时，才按配置自动切分 train/valid/test。
            train_df, valid_df, test_df = _split_dataframe(
                source_df,
                label_field=label_field,
                split_config=split_config,
            )
            if bool(split_config.get("persist", True)) and split_dir is not None:
                _persist_splits(
                    splits={"train": train_df, "valid": valid_df, "test": test_df},
                    split_paths=split_paths,
                    manifest_path=manifest_path,
                    source_path=source_path,
                    file_type=file_type or split_file_type,
                    split_config=split_config,
                )

        _validate_columns(train_df, [text_field, label_field])
        _validate_columns(valid_df, [text_field, label_field])
        _validate_columns(test_df, [text_field, label_field])

    # 先固化原始 split，再做输入拼接和文本归一化，避免持久化文件被“模型输入格式”污染。
    train_df = _apply_preprocess(train_df, preprocessor, input_builder, text_field)
    valid_df = _apply_preprocess(valid_df, preprocessor, input_builder, text_field)
    test_df = _apply_preprocess(test_df, preprocessor, input_builder, text_field)

    label_list = task_config.get("label_list") or train_df[label_field].astype(str).tolist()
    label2id, id2label = build_label_mappings(label_list)

    return {"train": train_df, "valid": valid_df, "test": test_df}, label2id, id2label


def tokenize_dataset_dict(
    raw_splits: dict[str, pd.DataFrame],
    tokenizer,
    config: dict,
    label2id: dict[str, int],
) -> DatasetDict:
    task_config = config["task"]
    model_config = config["model"]

    text_field = task_config["text_field"]
    label_field = task_config["label_field"]
    task_type = task_config["type"]
    max_length = int(model_config.get("max_length", 128))
    model_input_field = get_model_input_field(config, text_field)

    if task_type != "single_label_classification":
        raise NotImplementedError("This baseline only tokenizes single-label tasks by default.")

    dataset_dict = DatasetDict()

    for split_name, dataframe in raw_splits.items():
        dataset = _encode_single_label(
            dataframe=dataframe,
            label_field=label_field,
            label2id=label2id,
        )

        # 如果开启了 input_builder，这里喂给模型的是拼接后的 model_input，而不是原始 text。
        dataset = dataset.map(
            lambda batch: tokenizer(
                batch[model_input_field],
                truncation=True,
                max_length=max_length,
            ),
            batched=True,
            desc=f"Tokenizing {split_name} split",
        )
        dataset_dict[split_name] = dataset

    return dataset_dict


def prepare_data_for_training(config: dict, tokenizer) -> PreparedData:
    raw_splits, label2id, id2label = build_raw_splits(config)
    dataset_dict = tokenize_dataset_dict(raw_splits, tokenizer, config, label2id)
    return PreparedData(dataset_dict=dataset_dict, label2id=label2id, id2label=id2label)
