from __future__ import annotations

from typing import Any


DEFAULT_SUGGESTION_RULES = {
    "category_rules": {
        "设计与选型": {
            "suggestion": "建议复核设计依据、设备选型参数、标准适用性和现场工况匹配性。",
            "basis": "该问题属于设计与选型类，通常与前期设计输入、技术参数和标准适用性有关。",
            "action": "组织设计、建设和运维相关人员联合复核设计文件，并形成修订建议。",
        },
        "采购与制造": {
            "suggestion": "建议核查采购技术协议、出厂试验记录、制造质量证明文件和到货验收记录。",
            "basis": "该问题属于采购与制造类，重点风险集中在技术协议兑现和制造质量一致性。",
            "action": "核对采购技术协议和质保文件，必要时要求厂家补充说明或重新提供证明材料。",
        },
        "安装与施工": {
            "suggestion": "建议复核施工工艺、安装记录、隐蔽工程记录和现场复验结果。",
            "basis": "该问题属于安装与施工类，通常与施工过程控制和现场记录完整性有关。",
            "action": "组织现场复查，核验施工工艺执行情况，并补齐关键安装记录。",
        },
        "调试与试验": {
            "suggestion": "建议核查试验项目完整性、试验数据有效性、试验标准依据和缺项补试安排。",
            "basis": "该问题属于调试与试验类，通常直接影响设备投运前的状态判断。",
            "action": "梳理试验项目清单，尽快完成缺项补试并复核试验依据。",
        },
        "验收与交接": {
            "suggestion": "建议核查验收资料完整性、交接手续、遗留问题闭环和责任单位确认情况。",
            "basis": "该问题属于验收与交接类，重点在于资料闭环和责任边界是否明确。",
            "action": "补齐验收交接资料，明确责任单位和问题闭环时限。",
        },
        "资料与标识": {
            "suggestion": "建议补充资料归档、统一设备编号、完善标识标牌和台账一致性校验。",
            "basis": "该问题属于资料与标识类，通常影响后续运维和追溯管理。",
            "action": "开展资料归档和标识专项核查，统一编号规则并校验台账一致性。",
        },
        "运维与环境": {
            "suggestion": "建议核查运行环境、巡视记录、缺陷处理记录和运维责任落实情况。",
            "basis": "该问题属于运维与环境类，通常反映运行管理或环境约束未被充分控制。",
            "action": "结合现场巡视与缺陷记录开展复查，明确运维责任并跟踪整改。",
        },
        "未知": {
            "suggestion": "建议结合问题描述补充上下文信息，再进一步判断整改方向。",
            "basis": "当前问题类别信息不足，建议先补全业务上下文。",
            "action": "补充设备、环节和责任单位信息后，再组织针对性分析。",
        },
    },
    "high_priority_action": "建议纳入重点督办清单，并要求责任单位限期反馈整改结果。",
    "need_review_action": "该样本模型置信度不足，建议人工复核后再进入正式统计。",
}


def _normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return default
    return text


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalize_text(value).lower() in {"true", "1", "yes", "y"}


def generate_suggestion(row: dict[str, Any], rules: dict[str, Any] | None = None) -> dict[str, str]:
    rules = rules or DEFAULT_SUGGESTION_RULES
    category = _normalize_text(row.get("pred_category") or row.get("issue_category"), default="未知")
    category_rule = rules["category_rules"].get(category, rules["category_rules"]["未知"])

    suggestion = category_rule["suggestion"]
    suggestion_basis = category_rule["basis"]
    recommended_actions = [category_rule["action"]]

    if _normalize_text(row.get("priority_label")) == "高":
        recommended_actions.append(rules["high_priority_action"])

    if _normalize_bool(row.get("need_review") or row.get("need_manual_review")):
        recommended_actions.append(rules["need_review_action"])

    return {
        "suggestion": suggestion,
        "suggestion_basis": suggestion_basis,
        "recommended_action": " ".join(action for action in recommended_actions if action),
    }
