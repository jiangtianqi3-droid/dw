from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.config import resolve_path
from src.utils.decision_support import build_enriched_predictions
from src.utils.feedback import ensure_sample_ids
from src.utils.model_runtime import load_model_runtime, predict_dataframe


CLS_VECTOR_DIM = 768


def build_feature_matrix(
    input_df: pd.DataFrame,
    level_config: dict,
    level_model_path: str,
    category_config_path: str | None = None,
    category_model_path: str | None = None,
) -> pd.DataFrame:
    input_df = ensure_sample_ids(input_df.copy(), prefix="feature")
    level_runtime = load_model_runtime(level_config, level_model_path)
    prepared_df, level_predictions, cls_vectors = predict_dataframe(
        input_df,
        runtime=level_runtime,
        return_cls_vectors=True,
    )

    if cls_vectors is None or cls_vectors.shape[1] != CLS_VECTOR_DIM:
        raise ValueError(f"Unexpected CLS vector shape: {None if cls_vectors is None else cls_vectors.shape}")

    prepared_df["predicted_label_id"] = [item["predicted_label_id"] for item in level_predictions]
    prepared_df["predicted_label"] = [item["predicted_label"] for item in level_predictions]
    prepared_df["confidence"] = [item["confidence"] for item in level_predictions]
    enriched_df = build_enriched_predictions(
        dataframe=prepared_df,
        config=level_config,
        category_config_path=category_config_path,
        category_model_path=category_model_path,
        device=level_runtime.device,
    )

    output = pd.DataFrame(
        {
            "sample_id": enriched_df.get("sample_id", ""),
            "problem_text": enriched_df.get("text", ""),
            "device_type": enriched_df.get("device_type", ""),
            "supervision_major": enriched_df.get("specialty", ""),
            "supervision_stage": enriched_df.get("supervision_stage", ""),
            "problem_phase": enriched_df.get("problem_stage", ""),
            "pred_level": enriched_df.get("pred_level", ""),
            "pred_category": enriched_df.get("pred_category", ""),
            "level_confidence": enriched_df.get("level_confidence", ""),
            "category_confidence": enriched_df.get("category_confidence", ""),
            "need_review": enriched_df.get("need_review", False),
        }
    )

    for column_name in ["device_type", "supervision_major", "supervision_stage", "problem_phase"]:
        output[column_name] = output[column_name].fillna("").astype(str)

    cls_frame = pd.DataFrame(cls_vectors, columns=[f"cls_{idx}" for idx in range(CLS_VECTOR_DIM)])
    return pd.concat([output.reset_index(drop=True), cls_frame], axis=1)


def resolve_feature_matrix_output_path(output_file: str | None) -> Path:
    if output_file:
        return resolve_path(output_file)
    return resolve_path("outputs/feature_matrix.csv")
