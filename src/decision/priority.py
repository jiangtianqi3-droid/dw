from __future__ import annotations

from typing import Any


DEFAULT_PRIORITY_RULES = {
    "weights": {
        "level_score": 0.40,
        "device_score": 0.25,
        "category_score": 0.20,
        "repeat_score": 0.10,
        "confidence_score": 0.05,
    },
    "level_scores": {
        "重大": 1.0,
        "较大": 0.7,
        "一般": 0.4,
        "未知": 0.3,
    },
    "device_scores": [
        {"keywords": ["变压器", "主变", "组合电器", "断路器", "隔离开关"], "score": 1.0},
        {"keywords": ["母线", "电缆", "继电保护", "避雷器"], "score": 0.8},
    ],
    "device_default_score": 0.5,
    "category_scores": {
        "调试与试验": 0.9,
        "安装与施工": 0.8,
        "采购与制造": 0.75,
        "设计与选型": 0.7,
        "验收与交接": 0.65,
        "运维与环境": 0.6,
        "资料与标识": 0.4,
        "未知": 0.3,
    },
    "confidence_default": 0.5,
    "repeat_thresholds": {
        "high": 3,
        "medium": 1,
        "high_score": 1.0,
        "medium_score": 0.7,
        "default_score": 0.4,
    },
    "label_thresholds": {
        "high": 0.80,
        "medium": 0.60,
    },
}


def _normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return default
    return text


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _resolve_level(row: dict[str, Any]) -> str:
    return _normalize_text(
        row.get("pred_level")
        or row.get("final_level")
        or row.get("issue_level")
        or row.get("predicted_label")
        or row.get("label"),
        default="未知",
    )


def _resolve_category(row: dict[str, Any]) -> str:
    return _normalize_text(
        row.get("pred_category")
        or row.get("final_category")
        or row.get("issue_category"),
        default="未知",
    )


def _resolve_device_score(device_type: str, rules: dict[str, Any]) -> float:
    for item in rules["device_scores"]:
        if any(keyword in device_type for keyword in item["keywords"]):
            return float(item["score"])
    return float(rules["device_default_score"])


def _resolve_repeat_score(row: dict[str, Any], rules: dict[str, Any]) -> float:
    repeat_value = row.get("repeat_count", row.get("historical_count", 0))
    repeat_count = int(_safe_float(repeat_value, 0.0))
    threshold_rules = rules["repeat_thresholds"]
    if repeat_count >= int(threshold_rules["high"]):
        return float(threshold_rules["high_score"])
    if repeat_count >= int(threshold_rules["medium"]):
        return float(threshold_rules["medium_score"])
    return float(threshold_rules["default_score"])


def _resolve_confidence_score(row: dict[str, Any], rules: dict[str, Any]) -> float:
    default_value = float(rules["confidence_default"])
    level_confidence = _safe_float(row.get("level_confidence"), default_value)
    category_confidence = _safe_float(row.get("category_confidence"), default_value)
    return (level_confidence + category_confidence) / 2.0


def _resolve_priority_label(score: float, rules: dict[str, Any]) -> str:
    thresholds = rules["label_thresholds"]
    if score >= float(thresholds["high"]):
        return "高"
    if score >= float(thresholds["medium"]):
        return "中"
    return "低"


def compute_priority_score(row: dict[str, Any], rules: dict[str, Any] | None = None) -> dict[str, Any]:
    rules = rules or DEFAULT_PRIORITY_RULES
    level = _resolve_level(row)
    category = _resolve_category(row)
    device_type = _normalize_text(row.get("device_type"), default="其他设备")

    level_score = float(rules["level_scores"].get(level, rules["level_scores"]["未知"]))
    device_score = _resolve_device_score(device_type, rules)
    category_score = float(rules["category_scores"].get(category, rules["category_scores"]["未知"]))
    repeat_score = _resolve_repeat_score(row, rules)
    confidence_score = _resolve_confidence_score(row, rules)

    weights = rules["weights"]
    priority_score = (
        float(weights["level_score"]) * level_score
        + float(weights["device_score"]) * device_score
        + float(weights["category_score"]) * category_score
        + float(weights["repeat_score"]) * repeat_score
        + float(weights["confidence_score"]) * confidence_score
    )
    priority_score = round(priority_score, 4)
    priority_label = _resolve_priority_label(priority_score, rules)

    priority_reason = (
        f"问题等级为{level}，涉及设备为{device_type}，"
        f"问题类别为{category}，综合历史重复情况和模型置信度后，建议按{priority_label}优先级处理。"
    )

    return {
        "priority_score": priority_score,
        "priority_label": priority_label,
        "priority_reason": priority_reason,
        "level_score": round(level_score, 4),
        "device_score": round(device_score, 4),
        "category_score": round(category_score, 4),
        "repeat_score": round(repeat_score, 4),
        "confidence_score": round(confidence_score, 4),
    }
