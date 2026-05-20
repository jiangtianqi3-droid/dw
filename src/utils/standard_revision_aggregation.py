from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


AGGREGATION_OUTPUT_COLUMNS = [
    "rank",
    "kg_standard_id",
    "kg_standard_name",
    "kg_clause_id",
    "kg_clause_title",
    "clause_missing",
    "related_problem_count",
    "high_severity_problem_count",
    "medium_high_severity_problem_count",
    "revision_demand_count",
    "manual_review_count",
    "low_confidence_problem_count",
    "avg_relation_confidence",
    "max_relation_confidence",
    "avg_standard_revision_priority_score",
    "device_type_count",
    "major_count",
    "organization_count",
    "region_count",
    "aggregated_priority_score",
    "aggregated_priority_level",
    "aggregated_priority_reason",
    "representative_problem_ids",
    "representative_problem_texts",
]


FIELD_ALIASES: dict[str, list[str]] = {
    "problem_id": ["problem_id", "sample_id", "issue_id", "id"],
    "problem_text": ["问题描述", "problem_text", "text", "issue_text", "supervision_opinion"],
    "device_type": ["设备类型", "device_type"],
    "major": ["监督专业", "major", "specialty"],
    "organization": ["所属单位", "organization", "source_unit", "unit"],
    "region": ["地区", "region", "area"],
    "predicted_category": ["predicted_category", "pred_category", "issue_category", "category"],
    "predicted_severity": ["predicted_severity", "pred_level", "issue_level", "severity", "label"],
    "category_confidence": ["category_confidence", "pred_category_confidence"],
    "severity_confidence": ["severity_confidence", "level_confidence", "confidence"],
    "need_review": ["need_review", "need_manual_review"],
    "review_reason": ["review_reason"],
    "kg_standard_id": ["kg_standard_id", "standard_id"],
    "kg_standard_name": ["kg_standard_name", "standard_name"],
    "kg_standard_refs": ["kg_standard_refs", "standard_ref", "standard_refs"],
    "kg_clause_id": ["kg_clause_id", "clause_id"],
    "kg_clause_title": ["kg_clause_title", "clause_title"],
    "kg_relation_type": ["kg_relation_type"],
    "kg_relation_confidence": ["kg_relation_confidence", "kg_match_score", "relation_confidence"],
    "revision_demand": ["revision_demand", "revision_need"],
    "standard_revision_priority": ["standard_revision_priority", "revision_priority_label"],
    "standard_revision_priority_score": [
        "standard_revision_priority_score",
        "revision_priority_score",
        "priority_score",
    ],
    "standard_revision_priority_reason": ["standard_revision_priority_reason", "revision_reason"],
}


REQUIRED_REPORT_FIELDS = [
    "problem_id",
    "problem_text",
    "device_type",
    "major",
    "organization",
    "region",
    "predicted_severity",
    "category_confidence",
    "severity_confidence",
    "need_review",
    "kg_standard_id",
    "kg_standard_name",
    "kg_clause_id",
    "kg_clause_title",
    "kg_relation_confidence",
    "revision_demand",
    "standard_revision_priority_score",
]


HIGH_SEVERITY_KEYWORDS = {"重大", "严重", "高"}
MEDIUM_HIGH_SEVERITY_KEYWORDS = {"较大", "中高", "中"}
TRUE_VALUES = {"true", "1", "yes", "y", "是", "需", "需要", "命中", "有", "疑似", "true。"}


@dataclass(frozen=True)
class AggregationResult:
    dataframe: pd.DataFrame
    summary: dict[str, Any]
    missing_fields: list[str]


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def _normalize_bool(value: Any) -> bool:
    text = normalize_scalar(value).lower()
    if not text:
        return False
    if text in TRUE_VALUES:
        return True
    return any(keyword in text for keyword in ["需人工", "需复核", "标准缺失", "歧义", "冲突", "适用性不足"])


def _to_float(value: Any, default: float = 0.0) -> float:
    text = normalize_scalar(value)
    if not text:
        return default
    try:
        number = float(text.replace("%", ""))
    except ValueError:
        return default
    if "%" in text:
        number /= 100.0
    return number


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    if math.isnan(value):
        return lower
    return min(upper, max(lower, value))


def _score_0_to_1(value: Any) -> float:
    number = _to_float(value, 0.0)
    if number > 1.0:
        number /= 100.0
    return _clip(number)


def _severity_rank(value: Any) -> int:
    text = normalize_scalar(value)
    if any(keyword in text for keyword in HIGH_SEVERITY_KEYWORDS):
        return 3
    if any(keyword in text for keyword in MEDIUM_HIGH_SEVERITY_KEYWORDS):
        return 2
    return 1 if text else 0


def _is_high_severity(value: Any) -> bool:
    return _severity_rank(value) >= 3


def _is_medium_high_severity(value: Any) -> bool:
    return _severity_rank(value) >= 2


def _get_value(record: dict[str, Any], canonical_field: str) -> str:
    for alias in FIELD_ALIASES.get(canonical_field, [canonical_field]):
        if alias in record:
            value = normalize_scalar(record.get(alias))
            if value:
                return value
    return ""


def detect_missing_fields(columns: list[str]) -> list[str]:
    column_set = set(columns)
    missing: list[str] = []
    for canonical in REQUIRED_REPORT_FIELDS:
        if not any(alias in column_set for alias in FIELD_ALIASES.get(canonical, [canonical])):
            missing.append(canonical)
    return missing


def _split_standard_refs(value: Any) -> list[str]:
    text = normalize_scalar(value)
    if not text:
        return []
    parts = re.split(r"[;；\n\r|]+", text)
    return [part.strip() for part in parts if part.strip()]


def _stable_standard_id(standard_name: str) -> str:
    digest = hashlib.sha1(standard_name.encode("utf-8")).hexdigest()[:10]
    return f"standard_{digest}"


def _expand_records(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, raw_record in enumerate(dataframe.to_dict(orient="records"), start=1):
        record = {key: raw_record.get(key) for key in raw_record}
        standard_id = _get_value(record, "kg_standard_id")
        standard_name = _get_value(record, "kg_standard_name")
        standard_refs = _split_standard_refs(_get_value(record, "kg_standard_refs"))

        expanded_standards: list[tuple[str, str]] = []
        if standard_id or standard_name:
            expanded_standards.append((standard_id or _stable_standard_id(standard_name), standard_name or standard_id))
        elif standard_refs:
            expanded_standards.extend((_stable_standard_id(ref), ref) for ref in standard_refs)

        if not expanded_standards:
            continue

        base = {
            "problem_id": _get_value(record, "problem_id") or f"row_{index}",
            "problem_text": _get_value(record, "problem_text"),
            "device_type": _get_value(record, "device_type"),
            "major": _get_value(record, "major"),
            "organization": _get_value(record, "organization"),
            "region": _get_value(record, "region"),
            "predicted_severity": _get_value(record, "predicted_severity"),
            "need_review": _normalize_bool(_get_value(record, "need_review")),
            "category_confidence": _score_0_to_1(_get_value(record, "category_confidence")),
            "severity_confidence": _score_0_to_1(_get_value(record, "severity_confidence")),
            "kg_clause_id": _get_value(record, "kg_clause_id"),
            "kg_clause_title": "",
            "kg_relation_confidence": _score_0_to_1(_get_value(record, "kg_relation_confidence")),
            "revision_demand": _normalize_bool(_get_value(record, "revision_demand")),
            "standard_revision_priority_score": _score_0_to_1(_get_value(record, "standard_revision_priority_score")),
        }
        if base["kg_clause_id"]:
            base["kg_clause_title"] = _get_value(record, "kg_clause_title")
        if _get_value(record, "revision_demand") in {"标准缺失", "标准表述歧义", "标准冲突", "适用性不足"}:
            base["revision_demand"] = True

        for expanded_id, expanded_name in expanded_standards:
            expanded = dict(base)
            expanded["kg_standard_id"] = expanded_id
            expanded["kg_standard_name"] = expanded_name
            rows.append(expanded)
    return rows


def _distinct_count(records: list[dict[str, Any]], field: str) -> int:
    return len({normalize_scalar(record.get(field)) for record in records if normalize_scalar(record.get(field))})


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _priority_level(score: float) -> str:
    if score >= 80:
        return "高优先级"
    if score >= 60:
        return "中高优先级"
    if score >= 40:
        return "中优先级"
    return "低优先级"


def _low_confidence(record: dict[str, Any]) -> bool:
    confidence_values = [
        record.get("category_confidence", 0.0),
        record.get("severity_confidence", 0.0),
        record.get("kg_relation_confidence", 0.0),
    ]
    available = [value for value in confidence_values if value > 0]
    if not available:
        return False
    return min(available) < 0.6


def _representative_records(records: list[dict[str, Any]], limit: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(
        records,
        key=lambda record: (
            _severity_rank(record.get("predicted_severity")),
            record.get("kg_relation_confidence", 0.0),
            record.get("standard_revision_priority_score", 0.0),
            1 if normalize_scalar(record.get("problem_text")) else 0,
        ),
        reverse=True,
    )
    return ranked[:limit]


def _make_reason(
    count: int,
    high_count: int,
    medium_high_count: int,
    revision_count: int,
    avg_confidence: float,
    coverage_score: float,
    uncertainty_penalty: float,
    score: float,
) -> str:
    phrases: list[str] = []
    if count >= 5:
        phrases.append("关联问题数量较多")
    elif count >= 2:
        phrases.append("存在多条关联问题")
    else:
        phrases.append("当前关联问题数量较少")

    if high_count:
        phrases.append("高严重问题占比较高" if high_count / count >= 0.4 else "包含高严重问题")
    elif medium_high_count:
        phrases.append("包含中高严重问题")

    if revision_count:
        phrases.append("多条问题被识别为标准制修订需求" if revision_count >= 2 else "存在标准制修订需求")
    else:
        phrases.append("暂未集中命中标准制修订需求")

    if avg_confidence >= 0.75:
        phrases.append("图谱关联置信度较高")
    elif avg_confidence > 0:
        phrases.append("图谱关联置信度一般")

    if coverage_score >= 0.5:
        phrases.append("涉及设备、专业或单位地区范围较广")

    if uncertainty_penalty >= 0.4:
        phrases.append("但低置信度或需人工复核样本较多，建议专家复核后再定稿")

    advice = "建议优先纳入标准修订需求池。" if score >= 60 else "可作为后续标准修订储备项持续观察。"
    return "该标准/条款" + "，".join(phrases) + "，" + advice


def _aggregate_group(records: list[dict[str, Any]], total_count: int) -> dict[str, Any]:
    first = records[0]
    count = len(records)
    high_count = sum(1 for record in records if _is_high_severity(record.get("predicted_severity")))
    medium_high_count = sum(1 for record in records if _is_medium_high_severity(record.get("predicted_severity")))
    revision_count = sum(1 for record in records if record.get("revision_demand"))
    manual_review_count = sum(1 for record in records if record.get("need_review"))
    low_confidence_count = sum(1 for record in records if _low_confidence(record))
    relation_confidences = [record.get("kg_relation_confidence", 0.0) for record in records]
    priority_scores = [record.get("standard_revision_priority_score", 0.0) for record in records]
    avg_relation_confidence = _mean(relation_confidences)
    max_relation_confidence = max(relation_confidences or [0.0])
    avg_existing_priority = _mean(priority_scores)

    device_count = _distinct_count(records, "device_type")
    major_count = _distinct_count(records, "major")
    organization_count = _distinct_count(records, "organization")
    region_count = _distinct_count(records, "region")

    frequency_denominator = max(3, min(total_count, 10))
    frequency_score = _clip(count / frequency_denominator)
    severity_score = _clip((high_count + 0.6 * max(0, medium_high_count - high_count)) / count)
    revision_demand_score = _clip(revision_count / count)
    relation_confidence_score = _clip(avg_relation_confidence)
    coverage_score = _clip(
        (
            min(device_count / 3, 1)
            + min(major_count / 3, 1)
            + min(organization_count / 5, 1)
            + min(region_count / 5, 1)
        )
        / 4
    )
    uncertainty_penalty = _clip(max(manual_review_count / count, low_confidence_count / count))

    score_0_to_1 = (
        0.25 * frequency_score
        + 0.25 * severity_score
        + 0.20 * revision_demand_score
        + 0.15 * relation_confidence_score
        + 0.10 * coverage_score
        + 0.05 * avg_existing_priority
        - 0.10 * uncertainty_penalty
    )
    score = round(_clip(score_0_to_1) * 100, 2)
    representatives = _representative_records(records)

    return {
        "kg_standard_id": first["kg_standard_id"],
        "kg_standard_name": first["kg_standard_name"],
        "kg_clause_id": first["kg_clause_id"],
        "kg_clause_title": first["kg_clause_title"],
        "clause_missing": not bool(first["kg_clause_id"]),
        "related_problem_count": count,
        "high_severity_problem_count": high_count,
        "medium_high_severity_problem_count": medium_high_count,
        "revision_demand_count": revision_count,
        "manual_review_count": manual_review_count,
        "low_confidence_problem_count": low_confidence_count,
        "avg_relation_confidence": round(avg_relation_confidence, 4),
        "max_relation_confidence": round(max_relation_confidence, 4),
        "avg_standard_revision_priority_score": round(avg_existing_priority, 4),
        "device_type_count": device_count,
        "major_count": major_count,
        "organization_count": organization_count,
        "region_count": region_count,
        "aggregated_priority_score": score,
        "aggregated_priority_level": _priority_level(score),
        "aggregated_priority_reason": _make_reason(
            count=count,
            high_count=high_count,
            medium_high_count=medium_high_count,
            revision_count=revision_count,
            avg_confidence=avg_relation_confidence,
            coverage_score=coverage_score,
            uncertainty_penalty=uncertainty_penalty,
            score=score,
        ),
        "representative_problem_ids": "；".join(record["problem_id"] for record in representatives),
        "representative_problem_texts": "；".join(record["problem_text"] for record in representatives if record["problem_text"]),
    }


def aggregate_standard_revision_priority(
    dataframe: pd.DataFrame,
    top_k: int = 50,
    min_problem_count: int = 1,
) -> AggregationResult:
    expanded_records = _expand_records(dataframe)
    missing_fields = detect_missing_fields(list(dataframe.columns))

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in expanded_records:
        key = (
            record["kg_standard_id"],
            record["kg_standard_name"],
            record["kg_clause_id"],
            record["kg_clause_title"],
        )
        grouped.setdefault(key, []).append(record)

    rows = [
        _aggregate_group(records, total_count=len(expanded_records))
        for records in grouped.values()
        if len(records) >= min_problem_count
    ]
    rows = sorted(
        rows,
        key=lambda row: (
            row["aggregated_priority_score"],
            row["related_problem_count"],
            row["max_relation_confidence"],
        ),
        reverse=True,
    )[:top_k]

    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    output = pd.DataFrame(rows, columns=AGGREGATION_OUTPUT_COLUMNS)
    summary = {
        "total_related_problems": len(expanded_records),
        "aggregated_item_count": len(output),
        "standard_count": output["kg_standard_id"].nunique() if not output.empty else 0,
        "clause_count": int((~output["clause_missing"]).sum()) if not output.empty else 0,
        "high_priority_count": int((output["aggregated_priority_level"] == "高优先级").sum()) if not output.empty else 0,
        "medium_high_priority_count": int((output["aggregated_priority_level"] == "中高优先级").sum())
        if not output.empty
        else 0,
        "manual_review_problem_count": sum(1 for record in expanded_records if record.get("need_review")),
        "low_confidence_problem_count": sum(1 for record in expanded_records if _low_confidence(record)),
    }
    return AggregationResult(dataframe=output, summary=summary, missing_fields=missing_fields)


def build_standard_revision_report(result: AggregationResult) -> str:
    summary = result.summary
    dataframe = result.dataframe
    lines = [
        "# 标准/条款级标准修订优先级聚合报告",
        "",
        "## 1. 汇总",
        "",
        f"- 总关联问题数：{summary.get('total_related_problems', 0)}",
        f"- 参与聚合的标准数：{summary.get('standard_count', 0)}",
        f"- 参与聚合的条款数：{summary.get('clause_count', 0)}",
        f"- 高优先级标准/条款数量：{summary.get('high_priority_count', 0)}",
        f"- 中高优先级标准/条款数量：{summary.get('medium_high_priority_count', 0)}",
        "",
        "## 2. Top 10 标准/条款修订优先级",
        "",
    ]

    if dataframe.empty:
        lines.append("未生成聚合结果。请检查输入中是否包含标准或条款关联字段。")
    else:
        lines.append("| 排名 | 标准/条款 | 关联问题数 | 修订需求数 | 平均置信度 | 聚合分数 | 优先级 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- |")
        for _, row in dataframe.head(10).iterrows():
            title = row["kg_standard_name"]
            if not row["clause_missing"]:
                title = f"{title} / {row['kg_clause_title'] or row['kg_clause_id']}"
            lines.append(
                "| {rank} | {title} | {count} | {revision_count} | {confidence:.4f} | {score:.2f} | {level} |".format(
                    rank=row["rank"],
                    title=str(title).replace("|", "/"),
                    count=row["related_problem_count"],
                    revision_count=row["revision_demand_count"],
                    confidence=row["avg_relation_confidence"],
                    score=row["aggregated_priority_score"],
                    level=row["aggregated_priority_level"],
                )
            )

    lines.extend(["", "## 3. 缺失字段说明", ""])
    if result.missing_fields:
        lines.append("输入文件缺少以下字段或别名，已按空值或默认值参与统计：")
        for field in result.missing_fields:
            lines.append(f"- `{field}`")
    else:
        lines.append("输入文件包含本次聚合所需的主要字段。")

    lines.extend(["", "## 4. 低置信度与人工复核说明", ""])
    lines.append(f"- 需人工复核问题数：{summary.get('manual_review_problem_count', 0)}")
    lines.append(f"- 低置信度问题数：{summary.get('low_confidence_problem_count', 0)}")
    if summary.get("manual_review_problem_count", 0) or summary.get("low_confidence_problem_count", 0):
        lines.append("- 上述样本会降低聚合确定性，建议在形成正式标准修订清单前进行专家复核。")
    else:
        lines.append("- 未发现明显低置信度或需人工复核的样本。")

    lines.extend(
        [
            "",
            "## 5. 概念区分",
            "",
            "- `priority`：单条问题处理优先级。",
            "- `standard_revision_priority`：单条问题指向的标准修订优先级。",
            "- `aggregated_standard_revision_priority`：多条问题按标准/条款聚合后的标准修订优先级。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_json_summary(result: AggregationResult, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": result.summary,
        "missing_fields": result.missing_fields,
        "items": result.dataframe.to_dict(orient="records"),
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
