from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_GRAPH_COLUMNS = [
    "feature_matrix_id",
    "issue_id",
    "issue_text",
    "issue_category",
    "issue_level",
    "device_type",
    "specialty",
    "supervision_stage",
    "problem_stage",
    "event_time",
    "source_unit",
    "data_source",
    "is_reviewed",
    "review_status",
    "predicted_label",
    "final_confirmed_label",
    "label_source",
    "confidence",
    "model_version",
    "prediction_time",
    "problem_reason",
    "rule_name",
    "priority_score",
    "priority_label",
    "priority_reason",
    "suggestion",
    "recommended_action",
    "kg_match_status",
    "kg_match_score",
    "kg_point_id",
    "kg_point_text",
    "kg_project",
    "kg_stage",
    "kg_severity",
    "kg_standard_refs",
    "kg_requirement_texts",
    "revision_need",
    "revision_need_type",
    "revision_priority_score",
    "revision_priority_label",
    "revision_reason",
]


def get_graph_export_config(config: dict) -> dict:
    return config.get("graph_export", {})


def normalize_scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _first_non_empty(record: dict, field_names: list[str]) -> str:
    for field_name in field_names:
        value = normalize_scalar(record.get(field_name, ""))
        if value:
            return value
    return ""


def _infer_label_role(config: dict, requested_role: str) -> str:
    if requested_role != "auto":
        return requested_role

    target_name = normalize_scalar(config.get("task", {}).get("target_name", "")).lower()
    label_field = normalize_scalar(config.get("task", {}).get("label_field", "")).lower()
    if "level" in target_name or "level" in label_field:
        return "level"
    return "category"


def _infer_reviewed(record: dict) -> bool:
    review_status = normalize_scalar(record.get("review_status", "")).lower()
    return review_status not in {"", "pending"}


def _infer_final_label(record: dict) -> str:
    review_status = normalize_scalar(record.get("review_status", "")).lower()
    reviewed_label = normalize_scalar(record.get("reviewed_label", ""))
    final_label = normalize_scalar(record.get("final_label", ""))
    raw_label = normalize_scalar(record.get("label", ""))
    predicted_label = normalize_scalar(record.get("predicted_label", ""))

    if review_status in {"accepted", "corrected"} and reviewed_label:
        return reviewed_label
    return final_label or raw_label or predicted_label


def _infer_label_source(record: dict) -> str:
    review_status = normalize_scalar(record.get("review_status", "")).lower()
    reviewed_label = normalize_scalar(record.get("reviewed_label", ""))
    predicted_label = normalize_scalar(record.get("predicted_label", ""))
    raw_label = normalize_scalar(record.get("label", ""))

    if review_status in {"accepted", "corrected"} and reviewed_label:
        return "manual"
    if predicted_label:
        return "model"
    if raw_label:
        return "dataset"
    return ""


def build_graph_export_dataframe(
    dataframe: pd.DataFrame,
    config: dict,
    label_role: str = "auto",
) -> pd.DataFrame:
    records = dataframe.to_dict(orient="records")
    resolved_role = _infer_label_role(config, label_role)
    rows: list[dict] = []

    for record in records:
        final_label = _infer_final_label(record)
        issue_category = final_label if resolved_role == "category" else ""
        issue_level = final_label if resolved_role == "level" else ""

        rows.append(
            {
                "issue_id": _first_non_empty(record, ["sample_id", "issue_id"]),
                "feature_matrix_id": _first_non_empty(record, ["sample_id", "feature_matrix_id", "issue_id"]),
                "issue_text": _first_non_empty(record, ["text", "issue_text"]),
                "issue_category": issue_category,
                "issue_level": issue_level,
                "device_type": _first_non_empty(record, ["device_type"]),
                "specialty": _first_non_empty(record, ["specialty"]),
                "supervision_stage": _first_non_empty(record, ["supervision_stage"]),
                "problem_stage": _first_non_empty(record, ["problem_stage", "stage_name"]),
                "event_time": _first_non_empty(record, ["event_time"]),
                "source_unit": _first_non_empty(record, ["source_unit"]),
                "data_source": _first_non_empty(record, ["data_source"]),
                "is_reviewed": _infer_reviewed(record),
                "review_status": _first_non_empty(record, ["review_status"]),
                "predicted_label": _first_non_empty(record, ["predicted_label"]),
                "final_confirmed_label": final_label,
                "label_source": _infer_label_source(record),
                "confidence": _first_non_empty(record, ["confidence"]),
                "model_version": _first_non_empty(record, ["model_version"]),
                "prediction_time": _first_non_empty(record, ["prediction_time"]),
                "problem_reason": _first_non_empty(record, ["problem_reason"]),
                "rule_name": _first_non_empty(record, ["rule_name"]),
                "priority_score": _first_non_empty(record, ["priority_score"]),
                "priority_label": _first_non_empty(record, ["priority_label"]),
                "priority_reason": _first_non_empty(record, ["priority_reason"]),
                "suggestion": _first_non_empty(record, ["suggestion"]),
                "recommended_action": _first_non_empty(record, ["recommended_action"]),
                "kg_match_status": _first_non_empty(record, ["kg_match_status"]),
                "kg_match_score": _first_non_empty(record, ["kg_match_score"]),
                "kg_point_id": _first_non_empty(record, ["kg_point_id"]),
                "kg_point_text": _first_non_empty(record, ["kg_point_text"]),
                "kg_project": _first_non_empty(record, ["kg_project"]),
                "kg_stage": _first_non_empty(record, ["kg_stage"]),
                "kg_severity": _first_non_empty(record, ["kg_severity"]),
                "kg_standard_refs": _first_non_empty(record, ["kg_standard_refs"]),
                "kg_requirement_texts": _first_non_empty(record, ["kg_requirement_texts"]),
                "revision_need": _first_non_empty(record, ["revision_need"]),
                "revision_need_type": _first_non_empty(record, ["revision_need_type"]),
                "revision_priority_score": _first_non_empty(record, ["revision_priority_score"]),
                "revision_priority_label": _first_non_empty(record, ["revision_priority_label"]),
                "revision_reason": _first_non_empty(record, ["revision_reason"]),
            }
        )

    return pd.DataFrame(rows, columns=DEFAULT_GRAPH_COLUMNS)


def resolve_graph_output_path(config: dict, input_file: str | Path, output_file: str | None) -> Path:
    if output_file:
        return Path(output_file)

    graph_config = get_graph_export_config(config)
    output_dir = Path(graph_config.get("output_dir", "artifacts/graph"))
    input_stem = Path(input_file).stem
    return output_dir / f"{input_stem}_graph_export.jsonl"
