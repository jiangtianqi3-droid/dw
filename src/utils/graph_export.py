from __future__ import annotations

import hashlib
import json
import re
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
    "kg_standard_id",
    "kg_standard_name",
    "kg_clause_id",
    "kg_clause_title",
    "kg_relation_type",
    "kg_relation_confidence",
    "graph_nodes",
    "graph_edges",
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


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _split_refs(value: object) -> list[str]:
    text = normalize_scalar(value)
    if not text:
        return []
    parts = re.split(r"[;；\n\r|]+", text)
    return [part.strip() for part in parts if part.strip()]


def _node(node_id: str, node_type: str, name: str, **properties: object) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "name": name,
        "properties": {key: value for key, value in properties.items() if normalize_scalar(value)},
    }


def _edge(
    source: str,
    target: str,
    relation_type: str,
    confidence: str = "",
    source_field: str = "",
    relation_reason: str = "",
) -> dict:
    return {
        "source": source,
        "target": target,
        "relation_type": relation_type,
        "confidence": confidence,
        "source_field": source_field,
        "relation_reason": relation_reason,
    }


def _standard_pairs(record: dict) -> list[tuple[str, str, str]]:
    standard_id = _first_non_empty(record, ["kg_standard_id", "standard_id"])
    standard_name = _first_non_empty(record, ["kg_standard_name", "standard_name"])
    if standard_id or standard_name:
        return [(standard_id or _stable_id("standard", standard_name), standard_name or standard_id, "kg_standard_id")]

    pairs: list[tuple[str, str, str]] = []
    for ref in _split_refs(record.get("kg_standard_refs", "")):
        pairs.append((_stable_id("standard", ref), ref, "kg_standard_refs"))
    return pairs


def build_graph_nodes_edges(record: dict, issue_id: str, issue_category: str, issue_level: str) -> tuple[str, str]:
    problem_id = issue_id or _stable_id("problem", _first_non_empty(record, ["text", "issue_text", "问题描述"]))
    issue_text = _first_non_empty(record, ["text", "issue_text", "问题描述"])
    device_type = _first_non_empty(record, ["device_type", "设备类型"])
    specialty = _first_non_empty(record, ["specialty", "监督专业", "major"])
    standard_pairs = _standard_pairs(record)
    clause_id = _first_non_empty(record, ["kg_clause_id", "clause_id"])
    clause_title = _first_non_empty(record, ["kg_clause_title", "clause_title"])
    confidence = _first_non_empty(record, ["kg_relation_confidence", "kg_match_score"])
    relation_reason = _first_non_empty(record, ["kg_relation_reason", "standard_revision_priority_reason", "revision_reason"])

    nodes: dict[str, dict] = {
        problem_id: _node(problem_id, "problem", problem_id, text=issue_text),
    }
    edges: list[dict] = []

    if issue_category:
        category_id = _stable_id("category", issue_category)
        nodes[category_id] = _node(category_id, "category", issue_category)
        edges.append(_edge(problem_id, category_id, "has_category", source_field="predicted_category"))

    if issue_level:
        severity_id = _stable_id("severity", issue_level)
        nodes[severity_id] = _node(severity_id, "severity", issue_level)
        edges.append(_edge(problem_id, severity_id, "has_severity", source_field="predicted_severity"))

    if device_type:
        device_id = _stable_id("device", device_type)
        nodes[device_id] = _node(device_id, "device", device_type)
        edges.append(_edge(problem_id, device_id, "has_device", source_field="device_type"))

    if specialty:
        major_id = _stable_id("major", specialty)
        nodes[major_id] = _node(major_id, "major", specialty)
        edges.append(_edge(problem_id, major_id, "has_major", source_field="specialty"))

    clause_node_id = ""
    if clause_id:
        clause_node_id = clause_id
        nodes[clause_node_id] = _node(clause_node_id, "clause", clause_title or clause_id)
        edges.append(
            _edge(
                problem_id,
                clause_node_id,
                "related_to_clause",
                confidence=confidence,
                source_field="kg_clause_id",
                relation_reason=relation_reason,
            )
        )

    for standard_id, standard_name, source_field in standard_pairs:
        nodes[standard_id] = _node(standard_id, "standard", standard_name)
        edges.append(
            _edge(
                problem_id,
                standard_id,
                "related_to_standard",
                confidence=confidence,
                source_field=source_field,
                relation_reason=relation_reason,
            )
        )
        if clause_node_id:
            edges.append(
                _edge(
                    clause_node_id,
                    standard_id,
                    "belongs_to_standard",
                    confidence=confidence,
                    source_field="kg_clause_id",
                    relation_reason=relation_reason,
                )
            )

    return (
        json.dumps(list(nodes.values()), ensure_ascii=False),
        json.dumps(edges, ensure_ascii=False),
    )


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
        issue_category = (
            _first_non_empty(record, ["predicted_category", "pred_category", "issue_category", "category"])
            or (final_label if resolved_role == "category" else "")
        )
        issue_level = (
            _first_non_empty(record, ["predicted_severity", "pred_level", "issue_level", "severity"])
            or (final_label if resolved_role == "level" else "")
        )
        issue_id = _first_non_empty(record, ["problem_id", "sample_id", "issue_id"])
        graph_nodes, graph_edges = build_graph_nodes_edges(record, issue_id, issue_category, issue_level)

        rows.append(
            {
                "issue_id": issue_id,
                "feature_matrix_id": _first_non_empty(record, ["sample_id", "feature_matrix_id", "issue_id", "problem_id"]),
                "issue_text": _first_non_empty(record, ["text", "issue_text", "问题描述"]),
                "issue_category": issue_category,
                "issue_level": issue_level,
                "device_type": _first_non_empty(record, ["device_type", "设备类型"]),
                "specialty": _first_non_empty(record, ["specialty", "监督专业", "major"]),
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
                "kg_standard_id": _first_non_empty(record, ["kg_standard_id", "standard_id"]),
                "kg_standard_name": _first_non_empty(record, ["kg_standard_name", "standard_name"]),
                "kg_clause_id": _first_non_empty(record, ["kg_clause_id", "clause_id"]),
                "kg_clause_title": _first_non_empty(record, ["kg_clause_title", "clause_title"]),
                "kg_relation_type": _first_non_empty(record, ["kg_relation_type"]),
                "kg_relation_confidence": _first_non_empty(record, ["kg_relation_confidence", "kg_match_score"]),
                "graph_nodes": graph_nodes,
                "graph_edges": graph_edges,
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
