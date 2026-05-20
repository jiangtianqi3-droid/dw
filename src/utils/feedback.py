from __future__ import annotations

from datetime import datetime

import pandas as pd


def get_prediction_export_config(config: dict) -> dict:
    return config.get("feedback", {}).get("prediction_export", {})


def get_retrain_config(config: dict) -> dict:
    return config.get("feedback", {}).get("retrain", {})


def ensure_sample_ids(dataframe: pd.DataFrame, prefix: str = "pred") -> pd.DataFrame:
    dataframe = dataframe.copy()
    if "sample_id" not in dataframe.columns:
        dataframe["sample_id"] = [f"{prefix}_{idx:06d}" for idx in range(1, len(dataframe) + 1)]
        return dataframe

    missing_mask = dataframe["sample_id"].isna() | (dataframe["sample_id"].astype(str).str.strip() == "")
    if missing_mask.any():
        missing_indices = dataframe.index[missing_mask].tolist()
        for offset, row_index in enumerate(missing_indices, start=1):
            dataframe.at[row_index, "sample_id"] = f"{prefix}_{offset:06d}"
    return dataframe


def attach_feedback_columns(
    dataframe: pd.DataFrame,
    model_path: str,
    config: dict,
    include_model_input: bool,
) -> pd.DataFrame:
    dataframe = ensure_sample_ids(dataframe)
    dataframe = dataframe.copy()

    export_config = get_prediction_export_config(config)
    if not bool(export_config.get("include_review_columns", True)):
        return dataframe

    threshold = float(export_config.get("review_confidence_threshold", 0.75))
    default_status = str(export_config.get("default_review_status", "pending"))
    default_comment = str(export_config.get("default_review_comment", ""))

    dataframe["model_version"] = model_path
    dataframe["prediction_time"] = datetime.now().isoformat(timespec="seconds")
    dataframe["need_manual_review"] = dataframe["confidence"].astype(float) < threshold
    dataframe["review_status"] = default_status
    dataframe["reviewed_label"] = ""
    dataframe["review_comment"] = default_comment
    dataframe["feedback_time"] = ""
    dataframe["final_label"] = dataframe["predicted_label"]

    if not include_model_input and "model_input" in dataframe.columns:
        dataframe = dataframe.drop(columns=["model_input"])

    return dataframe


def build_retrain_dataset(
    original_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
    config: dict,
    label_field: str,
) -> pd.DataFrame:
    retrain_config = get_retrain_config(config)
    review_status_field = retrain_config.get("review_status_field", "review_status")
    reviewed_label_field = retrain_config.get("reviewed_label_field", "reviewed_label")
    accepted_statuses = {str(item).lower() for item in retrain_config.get("accepted_statuses", ["accepted", "corrected"])}
    sample_id_field = retrain_config.get("sample_id_field", "sample_id")
    data_source_field = retrain_config.get("data_source_field", "data_source")

    original_df = ensure_sample_ids(original_df)
    feedback_df = ensure_sample_ids(feedback_df)

    normalized_feedback = feedback_df.copy()
    normalized_feedback[review_status_field] = normalized_feedback[review_status_field].fillna("").astype(str).str.lower()
    selected_feedback = normalized_feedback[
        normalized_feedback[review_status_field].isin(accepted_statuses)
        & normalized_feedback[reviewed_label_field].fillna("").astype(str).str.strip().ne("")
    ].copy()

    if selected_feedback.empty:
        if label_field in original_df.columns:
            return original_df
        empty_result = original_df.iloc[0:0].copy()
        empty_result[label_field] = []
        return empty_result

    original_df = original_df.drop(columns=[reviewed_label_field], errors="ignore")
    selected_feedback = selected_feedback[[sample_id_field, reviewed_label_field]].drop_duplicates(subset=[sample_id_field], keep="last")
    merged = original_df.merge(selected_feedback, on=sample_id_field, how="left")
    if label_field in merged.columns:
        merged[label_field] = merged[reviewed_label_field].fillna(merged[label_field])
    else:
        merged[label_field] = merged[reviewed_label_field]
        merged = merged[merged[label_field].fillna("").astype(str).str.strip().ne("")].copy()

    if data_source_field in merged.columns:
        reviewed_mask = merged[reviewed_label_field].fillna("").astype(str).str.strip().ne("")
        merged.loc[reviewed_mask, data_source_field] = (
            merged.loc[reviewed_mask, data_source_field]
            .fillna("")
            .astype(str)
            .map(lambda value: f"{value}|feedback_reviewed" if value else "feedback_reviewed")
        )
    else:
        merged[data_source_field] = "feedback_reviewed"

    merged = merged.drop(columns=[reviewed_label_field])
    return merged


def append_feedback_samples(
    base_df: pd.DataFrame,
    feedback_samples_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    retrain_config = get_retrain_config(config)
    sample_id_field = retrain_config.get("sample_id_field", "sample_id")

    combined = pd.concat([base_df.copy(), feedback_samples_df.copy()], ignore_index=True, sort=False)
    if sample_id_field in combined.columns:
        combined = combined.drop_duplicates(subset=[sample_id_field], keep="last").reset_index(drop=True)
    return combined
