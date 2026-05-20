from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def build_compute_metrics(task_type: str, threshold: float = 0.5):
    if task_type == "single_label_classification":
        return _single_label_metrics

    if task_type == "multi_label_classification":
        return lambda eval_pred: _multi_label_metrics(eval_pred, threshold=threshold)

    raise NotImplementedError(f"Unsupported task type for metrics: {task_type}")


def _single_label_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


def _multi_label_metrics(eval_pred, threshold: float = 0.5):
    logits, labels = eval_pred
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    predictions = (probabilities >= threshold).astype(int)
    exact_match = (predictions == labels).all(axis=1).mean()
    return {
        "accuracy": float(exact_match),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }
