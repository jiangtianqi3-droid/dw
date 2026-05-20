from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


KG_REVISION_OUTPUT_COLUMNS = [
    "kg_match_status",
    "kg_match_score",
    "kg_point_id",
    "kg_point_text",
    "kg_project",
    "kg_stage",
    "kg_severity",
    "kg_standard_refs",
    "kg_requirement_texts",
    "kg_top_matches",
    "revision_need",
    "revision_need_type",
    "revision_priority_score",
    "revision_priority_label",
    "revision_reason",
]

STANDARD_PRIORITY_COLUMNS = [
    "standard_ref",
    "linked_issue_count",
    "revision_need_count",
    "major_or_above_issue_count",
    "max_issue_priority",
    "avg_issue_priority",
    "avg_kg_match_score",
    "standard_revision_priority_score",
    "standard_revision_priority_label",
    "example_issue_ids",
]

SUPPORTED_REVISION_TYPES = {
    "执行落实问题",
    "标准缺失",
    "标准表述歧义",
    "标准冲突",
    "适用性不足",
    "需人工判断",
}

STAGE_ALIASES = {
    "设备采购": "采购制造",
    "设备制造": "采购制造",
    "采购与制造": "采购制造",
    "采购制造": "采购制造",
    "设备安装": "安装调试",
    "设备调试": "安装调试",
    "安装调试": "安装调试",
    "工程设计": "工程设计",
    "规划可研": "规划可研",
    "设备验收": "竣工验收",
    "竣工验收": "竣工验收",
    "验收与交接": "竣工验收",
    "设备运维": "运维检修",
    "运维检修": "运维检修",
}


@dataclass(frozen=True)
class KGPoint:
    id: str
    label: str
    full_text: str
    equipment: str
    stage: str
    severity: str
    project: str
    standards: tuple[str, ...] = field(default_factory=tuple)
    requirements: tuple[str, ...] = field(default_factory=tuple)

    @property
    def searchable_text(self) -> str:
        return " ".join(
            item
            for item in [
                self.label,
                self.full_text,
                self.project,
                self.stage,
                self.severity,
                " ".join(self.standards),
                " ".join(self.requirements),
            ]
            if item
        )


@dataclass(frozen=True)
class KGMatch:
    point: KGPoint
    score: float
    reason: str


@dataclass
class KGIndex:
    points: list[KGPoint]
    supported_equipment: set[str]


def normalize_scalar(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def normalize_stage(value: Any) -> str:
    text = normalize_scalar(value)
    if not text:
        return ""
    if text in STAGE_ALIASES:
        return STAGE_ALIASES[text]
    for key, normalized in STAGE_ALIASES.items():
        if key in text:
            return normalized
    return text


def _clean_for_similarity(text: str) -> str:
    text = normalize_scalar(text).lower()
    text = re.sub(r"\d+(?:\.\d+)+", "", text)
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]+", "", text)
    return text


def _terms(text: str) -> set[str]:
    return set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", normalize_scalar(text).lower()))


def _ngrams(text: str, n: int = 2) -> set[str]:
    cleaned = _clean_for_similarity(text)
    if not cleaned:
        return set()
    if len(cleaned) <= n:
        return {cleaned}
    return {cleaned[index : index + n] for index in range(len(cleaned) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def text_similarity(left: str, right: str) -> float:
    left_clean = _clean_for_similarity(left)
    right_clean = _clean_for_similarity(right)
    if not left_clean or not right_clean:
        return 0.0
    if left_clean in right_clean or right_clean in left_clean:
        return 0.95
    char_score = _jaccard(_ngrams(left_clean), _ngrams(right_clean))
    term_score = _jaccard(_terms(left), _terms(right))
    return round(0.75 * char_score + 0.25 * term_score, 4)


def _node_text(node: dict[str, Any]) -> str:
    return normalize_scalar(node.get("full_text")) or normalize_scalar(node.get("label"))


def load_kg_index(path: str | Path) -> KGIndex:
    graph_path = Path(path)
    with graph_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    nodes = data.get("nodes", [])
    links = data.get("links", data.get("edges", []))
    node_map = {node["id"]: node for node in nodes if "id" in node}
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for link in links:
        source = link.get("source")
        target = link.get("target")
        if not source or not target:
            continue
        outgoing[source].append(link)
        incoming[target].append(link)

    supported_equipment = {
        normalize_scalar(node.get("label"))
        for node in nodes
        if node.get("node_type") == "设备类型" and normalize_scalar(node.get("label"))
    }

    points: list[KGPoint] = []
    for node in nodes:
        if node.get("node_type") != "监督要点":
            continue
        point_id = node["id"]
        project = ""
        for link in incoming.get(point_id, []):
            if link.get("edge_type") == "HAS_POINT":
                project_node = node_map.get(link.get("source"), {})
                project = normalize_scalar(project_node.get("label"))
                break

        severity = ""
        standards: list[str] = []
        requirements: list[str] = []
        for link in outgoing.get(point_id, []):
            target_node = node_map.get(link.get("target"), {})
            edge_type = link.get("edge_type")
            if edge_type == "HAS_SEVERITY":
                severity = normalize_scalar(target_node.get("label"))
            elif edge_type == "GOVERNED_BY":
                standards.append(
                    normalize_scalar(target_node.get("full_reference"))
                    or normalize_scalar(target_node.get("label"))
                )
            elif edge_type == "HAS_REQUIREMENT":
                requirements.append(_node_text(target_node))

        reference = normalize_scalar(node.get("reference"))
        if reference and reference not in standards:
            standards.append(reference)

        points.append(
            KGPoint(
                id=point_id,
                label=normalize_scalar(node.get("label")),
                full_text=_node_text(node),
                equipment=normalize_scalar(node.get("equipment")),
                stage=normalize_stage(node.get("stage")),
                severity=severity,
                project=project,
                standards=tuple(dict.fromkeys(item for item in standards if item)),
                requirements=tuple(dict.fromkeys(item for item in requirements if item)),
            )
        )

    return KGIndex(points=points, supported_equipment=supported_equipment)


def resolve_supported_equipment(record: dict[str, Any], supported_equipment: set[str]) -> str:
    candidates = [
        normalize_scalar(record.get("device_type")),
        normalize_scalar(record.get("rule_name")),
        normalize_scalar(record.get("text")),
    ]
    for candidate in candidates:
        for equipment in sorted(supported_equipment, key=len, reverse=True):
            if equipment and equipment in candidate:
                return equipment
    return ""


def _record_stage(record: dict[str, Any]) -> str:
    return normalize_stage(
        normalize_scalar(record.get("stage_name"))
        or normalize_scalar(record.get("supervision_stage"))
        or normalize_scalar(record.get("problem_stage"))
    )


def _record_level(record: dict[str, Any]) -> str:
    return (
        normalize_scalar(record.get("pred_level"))
        or normalize_scalar(record.get("rule_level"))
        or normalize_scalar(record.get("label"))
        or normalize_scalar(record.get("problem_level"))
    )


def _match_point(record: dict[str, Any], point: KGPoint, row_stage: str) -> KGMatch:
    checkpoint = normalize_scalar(record.get("checkpoint_text"))
    issue_text = normalize_scalar(record.get("text"))
    opinion_text = " ".join(
        item
        for item in [
            normalize_scalar(record.get("supervision_opinion")),
            normalize_scalar(record.get("actual_fix")),
        ]
        if item
    )
    rule_name = normalize_scalar(record.get("rule_name"))
    haystack = point.searchable_text

    checkpoint_score = text_similarity(checkpoint, f"{point.full_text} {' '.join(point.requirements)}")
    issue_score = text_similarity(issue_text, haystack)
    opinion_score = text_similarity(opinion_text, haystack)
    rule_score = text_similarity(rule_name, " ".join(point.standards))
    stage_score = 1.0 if row_stage and point.stage == row_stage else 0.0
    severity_score = 1.0 if _record_level(record) and point.severity == _record_level(record) else 0.0

    score = (
        0.50 * checkpoint_score
        + 0.24 * issue_score
        + 0.10 * opinion_score
        + 0.06 * rule_score
        + 0.06 * stage_score
        + 0.04 * severity_score
    )
    exact_reason = ""
    checkpoint_clean = _clean_for_similarity(checkpoint)
    point_clean = _clean_for_similarity(f"{point.full_text} {' '.join(point.requirements)}")
    if checkpoint_clean and checkpoint_clean in point_clean:
        score = max(score, 0.92)
        exact_reason = "监督要点文本直接命中"

    reason_parts = [
        exact_reason,
        f"checkpoint={checkpoint_score:.2f}",
        f"issue={issue_score:.2f}",
        f"stage={stage_score:.0f}",
        f"severity={severity_score:.0f}",
    ]
    return KGMatch(point=point, score=round(min(score, 1.0), 4), reason="; ".join(p for p in reason_parts if p))


def match_record_to_kg(
    record: dict[str, Any],
    kg_index: KGIndex,
    top_k: int = 3,
    min_score: float = 0.18,
) -> tuple[str, list[KGMatch]]:
    equipment = resolve_supported_equipment(record, kg_index.supported_equipment)
    if not equipment:
        return "unsupported_equipment", []

    row_stage = _record_stage(record)
    candidates = [point for point in kg_index.points if point.equipment == equipment]
    matches = sorted(
        (_match_point(record, point, row_stage=row_stage) for point in candidates),
        key=lambda item: item.score,
        reverse=True,
    )[:top_k]
    matches = [match for match in matches if match.score >= min_score]
    if not matches:
        return "low_score", []
    return "matched", matches


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def classify_revision_need(record: dict[str, Any], match_status: str, best_match: KGMatch | None) -> dict[str, Any]:
    text = " ".join(
        normalize_scalar(record.get(field))
        for field in [
            "text",
            "supervision_opinion",
            "actual_fix",
            "root_cause_analysis",
            "checkpoint_text",
            "pred_category",
        ]
    )
    standard_text = ""
    if best_match is not None:
        standard_text = " ".join([*best_match.point.standards, *best_match.point.requirements])

    if match_status != "matched" or best_match is None:
        need_type = "需人工判断"
        revision_need = False
        reason = "未能稳定关联到课题二图谱标准条款，建议人工判断是否形成修订需求。"
    elif _contains_any(text, ["标准缺失", "无标准", "没有标准", "无依据", "缺少依据", "未规定"]):
        need_type = "标准缺失"
        revision_need = True
        reason = "问题描述或整改意见出现标准缺失/依据不足信号。"
    elif _contains_any(text, ["冲突", "矛盾", "不一致", "交叉", "重复规定"]):
        need_type = "标准冲突"
        revision_need = True
        reason = "问题描述或整改意见出现标准冲突/交叉不一致信号。"
    elif _contains_any(text, ["歧义", "不明确", "不清晰", "理解偏差", "表述", "边界不清"]):
        need_type = "标准表述歧义"
        revision_need = True
        reason = "问题描述或整改意见出现标准表述不清或理解偏差信号。"
    elif _contains_any(text, ["不适用", "适用性", "现场实际", "特殊地区", "工况", "图纸修改", "选型"]):
        need_type = "适用性不足"
        revision_need = True
        reason = "问题与现场工况、设计选型或标准适用性相关，建议复核标准适用范围。"
    elif _contains_any(text, ["未按", "未见", "未进行", "未采用", "不符合", "不到位", "漏", "错位"]):
        need_type = "执行落实问题"
        revision_need = False
        reason = "问题更偏向标准或监督要求执行不到位，优先纳入整改闭环。"
    else:
        need_type = "需人工判断"
        revision_need = False
        reason = "当前规则未发现明确修订信号，建议结合人工复核确认。"

    priority_score = compute_revision_priority(record, revision_need, best_match.score if best_match else 0.0)
    return {
        "revision_need": revision_need,
        "revision_need_type": need_type,
        "revision_priority_score": priority_score,
        "revision_priority_label": priority_label(priority_score),
        "revision_reason": f"{reason} 关联标准依据：{standard_text[:120]}" if standard_text else reason,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return normalize_scalar(value).lower() in {"true", "1", "yes", "y"}


def compute_revision_priority(record: dict[str, Any], revision_need: bool, kg_score: float) -> float:
    issue_priority = _safe_float(record.get("priority_score"), 0.0)
    if issue_priority <= 0:
        issue_priority = {"高": 0.85, "中": 0.62, "低": 0.35}.get(normalize_scalar(record.get("priority_label")), 0.45)
    level = _record_level(record)
    level_factor = {"重大": 1.0, "较大": 0.72, "一般": 0.42}.get(level, 0.35)
    need_factor = 1.0 if revision_need else 0.0
    score = 0.48 * issue_priority + 0.24 * need_factor + 0.18 * kg_score + 0.10 * level_factor
    return round(min(score, 1.0), 4)


def priority_label(score: float) -> str:
    if score >= 0.75:
        return "高"
    if score >= 0.55:
        return "中"
    return "低"


def _matches_to_json(matches: list[KGMatch]) -> str:
    payload = [
        {
            "kg_point_id": match.point.id,
            "score": match.score,
            "kg_point_text": match.point.full_text,
            "kg_project": match.point.project,
            "kg_stage": match.point.stage,
            "kg_standard_refs": list(match.point.standards),
        }
        for match in matches
    ]
    return json.dumps(payload, ensure_ascii=False)


def enrich_dataframe_with_kg(
    dataframe: pd.DataFrame,
    kg_index: KGIndex,
    top_k: int = 3,
    min_score: float = 0.18,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in dataframe.to_dict(orient="records"):
        match_status, matches = match_record_to_kg(record, kg_index, top_k=top_k, min_score=min_score)
        best_match = matches[0] if matches else None
        revision = classify_revision_need(record, match_status, best_match)
        row = dict(record)
        if best_match is None:
            row.update(
                {
                    "kg_match_status": match_status,
                    "kg_match_score": 0.0,
                    "kg_point_id": "",
                    "kg_point_text": "",
                    "kg_project": "",
                    "kg_stage": "",
                    "kg_severity": "",
                    "kg_standard_refs": "",
                    "kg_requirement_texts": "",
                    "kg_top_matches": "[]",
                }
            )
        else:
            point = best_match.point
            row.update(
                {
                    "kg_match_status": match_status,
                    "kg_match_score": best_match.score,
                    "kg_point_id": point.id,
                    "kg_point_text": point.full_text,
                    "kg_project": point.project,
                    "kg_stage": point.stage,
                    "kg_severity": point.severity,
                    "kg_standard_refs": "；".join(point.standards),
                    "kg_requirement_texts": " || ".join(point.requirements),
                    "kg_top_matches": _matches_to_json(matches),
                }
            )
        row.update(revision)
        rows.append(row)

    return pd.DataFrame(rows)


def _split_standard_refs(value: Any) -> list[str]:
    text = normalize_scalar(value)
    if not text:
        return []
    return [item.strip() for item in re.split(r"[；;]\s*|\n+", text) if item.strip()]


def build_standard_priority_report(dataframe: pd.DataFrame) -> pd.DataFrame:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in dataframe.to_dict(orient="records"):
        for standard_ref in _split_standard_refs(record.get("kg_standard_refs")):
            buckets[standard_ref].append(record)

    rows: list[dict[str, Any]] = []
    for standard_ref, records in buckets.items():
        linked_count = len(records)
        revision_need_count = sum(1 for row in records if _as_bool(row.get("revision_need")))
        major_count = sum(1 for row in records if _record_level(row) in {"重大", "较大"})
        issue_scores = [_safe_float(row.get("priority_score"), 0.0) for row in records]
        revision_scores = [_safe_float(row.get("revision_priority_score"), 0.0) for row in records]
        kg_scores = [_safe_float(row.get("kg_match_score"), 0.0) for row in records]
        max_issue_priority = max(issue_scores or [0.0])
        avg_issue_priority = sum(issue_scores) / linked_count if linked_count else 0.0
        avg_kg_score = sum(kg_scores) / linked_count if linked_count else 0.0
        count_score = min(linked_count / 5.0, 1.0)
        need_rate = revision_need_count / linked_count if linked_count else 0.0
        major_rate = major_count / linked_count if linked_count else 0.0
        aggregate_score = (
            0.24 * count_score
            + 0.22 * need_rate
            + 0.20 * max_issue_priority
            + 0.16 * avg_issue_priority
            + 0.10 * major_rate
            + 0.08 * avg_kg_score
        )
        example_ids = [
            normalize_scalar(row.get("sample_id")) or normalize_scalar(row.get("issue_id"))
            for row in sorted(records, key=lambda item: _safe_float(item.get("revision_priority_score"), 0.0), reverse=True)[:5]
        ]
        rows.append(
            {
                "standard_ref": standard_ref,
                "linked_issue_count": linked_count,
                "revision_need_count": revision_need_count,
                "major_or_above_issue_count": major_count,
                "max_issue_priority": round(max_issue_priority, 4),
                "avg_issue_priority": round(avg_issue_priority, 4),
                "avg_kg_match_score": round(avg_kg_score, 4),
                "standard_revision_priority_score": round(max(aggregate_score, max(revision_scores or [0.0]) * 0.8), 4),
                "standard_revision_priority_label": priority_label(round(max(aggregate_score, max(revision_scores or [0.0]) * 0.8), 4)),
                "example_issue_ids": "；".join(example_ids),
            }
        )

    return pd.DataFrame(rows, columns=STANDARD_PRIORITY_COLUMNS).sort_values(
        by=["standard_revision_priority_score", "linked_issue_count"],
        ascending=[False, False],
        kind="mergesort",
    )


def build_markdown_report(dataframe: pd.DataFrame, standard_report: pd.DataFrame, kg_graph_path: str | Path) -> str:
    total = len(dataframe)
    status_counts = Counter(dataframe.get("kg_match_status", pd.Series(dtype=str)).fillna("").astype(str))
    matched_count = int(status_counts.get("matched", 0))
    match_rate = matched_count / total if total else 0.0
    revision_need_count = int(dataframe.get("revision_need", pd.Series(dtype=bool)).map(_as_bool).sum())
    type_counts = Counter(dataframe.get("revision_need_type", pd.Series(dtype=str)).fillna("").astype(str))

    lines = [
        "# 知识图谱关联与标准修订优先级验证报告",
        "",
        "## 1. 输入图谱",
        f"- 课题二图谱文件：`{kg_graph_path}`",
        "",
        "## 2. 样本概览",
        f"- 样本总数：{total}",
        f"- 成功关联课题二监督要点：{matched_count}",
        f"- 匹配率：{match_rate:.1%}",
        f"- 疑似标准修订需求数：{revision_need_count}",
        "",
        "## 3. 匹配状态分布",
    ]
    for status, count in status_counts.most_common():
        lines.append(f"- {status or '空'}：{count}")

    lines.extend(["", "## 4. 修订需求类型分布"])
    for need_type, count in type_counts.most_common():
        lines.append(f"- {need_type or '空'}：{count}")

    lines.extend(["", "## 5. 标准修订优先级 Top 10"])
    if standard_report.empty:
        lines.append("- 暂无可聚合的关联标准。")
    else:
        for row in standard_report.head(10).to_dict(orient="records"):
            lines.append(
                "- "
                f"{row['standard_ref']} | "
                f"优先级={row['standard_revision_priority_label']}({row['standard_revision_priority_score']}) | "
                f"关联问题={row['linked_issue_count']} | 修订需求={row['revision_need_count']}"
            )

    lines.extend(["", "## 6. 典型匹配样本"])
    matched = dataframe[dataframe.get("kg_match_status", "") == "matched"].sort_values(
        by="kg_match_score",
        ascending=False,
        kind="mergesort",
    )
    if matched.empty:
        lines.append("- 暂无成功匹配样本。")
    else:
        for row in matched.head(5).to_dict(orient="records"):
            lines.append(
                "- "
                f"{normalize_scalar(row.get('sample_id'))} | "
                f"score={_safe_float(row.get('kg_match_score')):.4f} | "
                f"问题={normalize_scalar(row.get('text'))[:60]} | "
                f"要点={normalize_scalar(row.get('kg_point_text'))[:60]}"
            )

    lines.extend(
        [
            "",
            "## 7. 局限说明",
            "- 第一版使用离线 `kg_graph.json`，不依赖 Neo4j/FastAPI。",
            "- 课题二当前主要覆盖组合电器、隔离开关；其他设备输出 `unsupported_equipment`。",
            "- 标准修订需求识别为规则版本，适合形成验证闭环，正式结论仍需专家复核。",
        ]
    )
    return "\n".join(lines) + "\n"


def write_text(path: str | Path, content: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
