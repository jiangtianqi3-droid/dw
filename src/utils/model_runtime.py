from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.data.preprocess import InputBuilder, TextPreprocessor
from src.utils.config import load_config, resolve_path
from src.utils.labels import load_label_mapping


@dataclass
class ModelRuntime:
    config: dict[str, Any]
    model_path: Path
    tokenizer: Any
    model: Any
    id2label: dict[int, str]
    preprocessor: TextPreprocessor
    input_builder: InputBuilder
    text_field: str
    model_input_field: str
    max_length: int
    batch_size: int
    device: torch.device


def infer_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_runtime(
    config_or_path: dict[str, Any] | str,
    model_path: str | Path,
    batch_size: int | None = None,
    device: torch.device | None = None,
) -> ModelRuntime:
    config = load_config(config_or_path) if isinstance(config_or_path, str) else config_or_path
    resolved_model_path = resolve_path(str(model_path))
    if resolved_model_path is None or not resolved_model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    _, id2label = load_label_mapping(resolved_model_path / "label_mapping.json")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(resolved_model_path)
    runtime_device = device or infer_device()
    model.to(runtime_device)
    model.eval()

    text_field = config["task"]["text_field"]
    input_builder = InputBuilder.from_config(config.get("input_builder", {}))
    preprocessor = TextPreprocessor.from_config(config.get("preprocess", {}))
    return ModelRuntime(
        config=config,
        model_path=resolved_model_path,
        tokenizer=tokenizer,
        model=model,
        id2label=id2label,
        preprocessor=preprocessor,
        input_builder=input_builder,
        text_field=text_field,
        model_input_field=input_builder.get_output_field(text_field),
        max_length=int(config["model"].get("max_length", 128)),
        batch_size=int(batch_size or config.get("predict", {}).get("batch_size", 16)),
        device=runtime_device,
    )


def prepare_inference_dataframe(dataframe: pd.DataFrame, runtime: ModelRuntime) -> pd.DataFrame:
    if runtime.text_field not in dataframe.columns:
        raise ValueError(f"Missing text field '{runtime.text_field}' in inference input.")
    return runtime.input_builder.transform_dataframe(
        dataframe.copy(),
        text_field=runtime.text_field,
        preprocessor=runtime.preprocessor,
    )


def run_single_label_inference(
    texts: list[str],
    runtime: ModelRuntime,
    return_cls_vectors: bool = False,
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    results: list[dict[str, Any]] = []
    cls_vectors: list[np.ndarray] = []

    for start in range(0, len(texts), runtime.batch_size):
        batch_texts = texts[start : start + runtime.batch_size]
        inputs = runtime.tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=runtime.max_length,
            return_tensors="pt",
        )
        inputs = {key: value.to(runtime.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = runtime.model(**inputs, output_hidden_states=return_cls_vectors)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            confidences, pred_ids = torch.max(probabilities, dim=-1)
            if return_cls_vectors:
                cls_batch = outputs.hidden_states[-1][:, 0, :].detach().cpu().numpy()
                cls_vectors.extend(cls_batch)

        for pred_id, confidence in zip(pred_ids.tolist(), confidences.tolist()):
            results.append(
                {
                    "predicted_label_id": pred_id,
                    "predicted_label": runtime.id2label[pred_id],
                    "confidence": round(float(confidence), 6),
                }
            )

    if not return_cls_vectors:
        return results, None
    return results, np.asarray(cls_vectors, dtype=np.float32)


def predict_dataframe(
    dataframe: pd.DataFrame,
    runtime: ModelRuntime,
    return_cls_vectors: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]], np.ndarray | None]:
    prepared = prepare_inference_dataframe(dataframe, runtime)
    texts = prepared[runtime.model_input_field].astype(str).tolist()
    predictions, cls_vectors = run_single_label_inference(
        texts=texts,
        runtime=runtime,
        return_cls_vectors=return_cls_vectors,
    )
    return prepared, predictions, cls_vectors
