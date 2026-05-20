from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.common import ensure_dir, save_json


def _extract_logits(predictions) -> np.ndarray:
    if isinstance(predictions, tuple):
        return np.asarray(predictions[0])
    return np.asarray(predictions)


def export_single_label_analysis(
    predictions,
    label_ids,
    id2label: dict[int, str],
    output_dir: str | Path,
    split_name: str,
) -> None:
    output_dir = ensure_dir(output_dir)
    logits = _extract_logits(predictions)
    pred_ids = np.argmax(logits, axis=-1)

    ordered_label_ids = sorted(id2label.keys())
    ordered_label_names = [id2label[label_id] for label_id in ordered_label_ids]

    report = classification_report(
        label_ids,
        pred_ids,
        labels=ordered_label_ids,
        target_names=ordered_label_names,
        output_dict=True,
        zero_division=0,
    )
    save_json(report, Path(output_dir) / f"{split_name}_classification_report.json")

    matrix = confusion_matrix(label_ids, pred_ids, labels=ordered_label_ids)
    matrix_df = pd.DataFrame(matrix, index=ordered_label_names, columns=ordered_label_names)
    matrix_df.to_csv(Path(output_dir) / f"{split_name}_confusion_matrix.csv", encoding="utf-8-sig")

    save_json(
        {
            "labels": ordered_label_names,
            "matrix": matrix.tolist(),
        },
        Path(output_dir) / f"{split_name}_confusion_matrix.json",
    )
