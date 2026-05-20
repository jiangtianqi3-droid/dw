from __future__ import annotations

from pathlib import Path

from src.utils.common import load_json, save_json


def build_label_mappings(labels: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    normalized = [str(label) for label in labels]
    unique_labels = sorted(set(normalized))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def save_label_mapping(
    label2id: dict[str, int],
    id2label: dict[int, str],
    path: str | Path,
) -> None:
    payload = {
        "label2id": label2id,
        "id2label": {str(key): value for key, value in id2label.items()},
    }
    save_json(payload, path)


def load_label_mapping(path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = load_json(path)
    label2id = {str(key): int(value) for key, value in payload["label2id"].items()}
    id2label = {int(key): str(value) for key, value in payload["id2label"].items()}
    return label2id, id2label
