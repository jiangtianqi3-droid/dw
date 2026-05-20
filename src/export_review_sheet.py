from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path

import pandas as pd

from src.data.io import read_table
from src.utils.config import resolve_path


DEFAULT_PRIORITY_COLUMNS = [
    "sample_id",
    "need_manual_review",
    "need_review",
    "confidence",
    "predicted_label",
    "pred_level",
    "pred_category",
    "level_confidence",
    "category_confidence",
    "priority_score",
    "priority_label",
    "priority_reason",
    "suggestion",
    "suggested_basis",
    "suggestion_basis",
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
    "final_label",
    "label",
    "review_status",
    "reviewed_label",
    "review_comment",
    "feedback_time",
    "text",
    "device_type",
    "specialty",
    "supervision_stage",
    "problem_stage",
    "problem_reason",
    "rule_name",
    "checkpoint_text",
    "supervision_opinion",
    "actual_fix",
    "event_time",
    "data_source",
    "model_version",
    "prediction_time",
]

DISPLAY_NAME_MAP = {
    "sample_id": "样本编号",
    "need_manual_review": "是否建议人工复核",
    "need_review": "是否进入复核流程",
    "confidence": "等级模型置信度",
    "predicted_label": "原始预测标签",
    "pred_level": "预测问题等级",
    "pred_category": "预测问题类别",
    "level_confidence": "问题等级置信度",
    "category_confidence": "问题类别置信度",
    "priority_score": "优先级得分",
    "priority_label": "优先级标签",
    "priority_reason": "优先级判定原因",
    "suggestion": "优化建议",
    "suggested_basis": "建议依据",
    "suggestion_basis": "建议依据",
    "recommended_action": "建议动作",
    "kg_match_status": "图谱匹配状态",
    "kg_match_score": "图谱匹配得分",
    "kg_point_id": "关联监督要点ID",
    "kg_point_text": "关联监督要点",
    "kg_project": "关联监督项目",
    "kg_stage": "关联技术阶段",
    "kg_severity": "图谱问题分级",
    "kg_standard_refs": "关联标准依据",
    "kg_requirement_texts": "关联监督要求",
    "revision_need": "是否疑似修订需求",
    "revision_need_type": "修订需求类型",
    "revision_priority_score": "修订优先级得分",
    "revision_priority_label": "修订优先级标签",
    "revision_reason": "修订需求判定原因",
    "final_label": "当前最终标签",
    "label": "原始标签",
    "review_status": "复核状态",
    "reviewed_label": "人工确认标签",
    "review_comment": "复核备注",
    "feedback_time": "复核时间",
    "text": "问题文本",
    "device_type": "设备类型",
    "specialty": "监督专业",
    "supervision_stage": "监督环节",
    "problem_stage": "问题阶段",
    "problem_reason": "问题原因",
    "rule_name": "细则名称",
    "checkpoint_text": "监督要点",
    "supervision_opinion": "监督意见",
    "actual_fix": "实际整改措施",
    "event_time": "发现时间",
    "data_source": "数据来源",
    "model_version": "模型版本",
    "prediction_time": "预测时间",
}

SUMMARY_ITEM_DISPLAY_MAP = {
    "total_rows": "样本总数",
    "need_manual_review_true": "建议人工复核数量",
    "need_manual_review_false": "无需人工复核数量",
    "need_review_true": "进入复核流程数量",
    "review_status_pending": "待复核数量",
    "high_priority_and_need_review": "高优先级且需复核数量",
}

LONG_TEXT_COLUMNS = {
    "问题文本",
    "监督要点",
    "监督意见",
    "实际整改措施",
    "优先级判定原因",
    "优化建议",
    "建议依据",
    "建议动作",
    "关联监督要点",
    "关联标准依据",
    "关联监督要求",
    "修订需求判定原因",
}

SUMMARY_COLUMNS = {"item": "统计项", "value": "数值"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a human-friendly Excel review sheet from prediction results.")
    parser.add_argument("--input-file", type=str, required=True, help="Prediction export file, CSV/JSONL/XLSX.")
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output Excel file. Defaults to '<input_stem>_human.xlsx' in the same directory.",
    )
    return parser.parse_args()


def _resolve_output_path(input_path: Path, output_file: str | None) -> Path:
    if output_file:
        return resolve_path(output_file)
    return input_path.with_name(f"{input_path.stem}_human.xlsx")


def _normalize_review_flag(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _build_review_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"predicted_label", "confidence", "need_manual_review", "review_status", "reviewed_label"}
    missing = sorted(required_columns - set(dataframe.columns))
    if missing:
        dataframe = dataframe.copy()
        if "predicted_label" in missing:
            fallback_label = None
            for candidate in ["pred_level", "pred_category", "final_label", "label"]:
                if candidate in dataframe.columns:
                    fallback_label = dataframe[candidate]
                    break
            dataframe["predicted_label"] = fallback_label if fallback_label is not None else ""
        if "confidence" in missing:
            fallback_confidence = None
            for candidate in ["level_confidence", "category_confidence", "kg_match_score"]:
                if candidate in dataframe.columns:
                    fallback_confidence = dataframe[candidate]
                    break
            dataframe["confidence"] = fallback_confidence if fallback_confidence is not None else pd.NA
        if "need_manual_review" in missing:
            dataframe["need_manual_review"] = dataframe["need_review"] if "need_review" in dataframe.columns else False
        if "review_status" in missing:
            dataframe["review_status"] = "pending"
        if "reviewed_label" in missing:
            dataframe["reviewed_label"] = ""

    selected_columns = [column for column in DEFAULT_PRIORITY_COLUMNS if column in dataframe.columns]
    review_df = dataframe[selected_columns].copy()
    review_df["need_manual_review"] = _normalize_review_flag(review_df["need_manual_review"])
    if "need_review" in review_df.columns:
        review_df["need_review"] = _normalize_review_flag(review_df["need_review"])
    review_df["confidence"] = pd.to_numeric(review_df["confidence"], errors="coerce")
    for column in ["level_confidence", "category_confidence", "priority_score", "kg_match_score", "revision_priority_score"]:
        if column in review_df.columns:
            review_df[column] = pd.to_numeric(review_df[column], errors="coerce")

    # 人工复核时优先看需要复核、置信度最低的样本。
    return review_df.sort_values(
        by=[column for column in ["need_review", "need_manual_review", "priority_score", "confidence"] if column in review_df.columns],
        ascending=[False, False, False, True][: len([column for column in ["need_review", "need_manual_review", "priority_score", "confidence"] if column in review_df.columns])],
        kind="mergesort",
    ).reset_index(drop=True)


def _build_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    need_review = _normalize_review_flag(dataframe["need_manual_review"])
    review_flag = _normalize_review_flag(dataframe["need_review"]) if "need_review" in dataframe.columns else need_review
    review_status = dataframe["review_status"].fillna("").astype(str)
    rows = [
        {"item": "total_rows", "value": int(len(dataframe))},
        {"item": "need_manual_review_true", "value": int(need_review.sum())},
        {"item": "need_manual_review_false", "value": int((~need_review).sum())},
        {"item": "need_review_true", "value": int(review_flag.sum())},
        {"item": "review_status_pending", "value": int((review_status == "pending").sum())},
    ]

    if "priority_label" in dataframe.columns:
        priority_series = dataframe["priority_label"].fillna("").astype(str)
        for label in ["高", "中", "低"]:
            rows.append({"item": f"priority_{label}", "value": int((priority_series == label).sum())})
        rows.append(
            {
                "item": "high_priority_and_need_review",
                "value": int(((priority_series == "高") & review_flag).sum()),
            }
        )

    if "revision_need" in dataframe.columns:
        revision_need = _normalize_review_flag(dataframe["revision_need"])
        rows.append({"item": "revision_need_true", "value": int(revision_need.sum())})
        if "revision_need_type" in dataframe.columns:
            type_counts = dataframe["revision_need_type"].fillna("").astype(str).value_counts()
            for need_type, count in type_counts.items():
                if need_type:
                    rows.append({"item": f"revision::{need_type}", "value": int(count)})

    if "kg_match_status" in dataframe.columns:
        status_counts = dataframe["kg_match_status"].fillna("").astype(str).value_counts()
        for status, count in status_counts.items():
            if status:
                rows.append({"item": f"kg_match::{status}", "value": int(count)})

    if "pred_category" in dataframe.columns:
        category_counts = dataframe["pred_category"].fillna("").astype(str).value_counts()
        for category, count in category_counts.items():
            if category:
                rows.append({"item": f"category::{category}", "value": int(count)})

    if "pred_level" in dataframe.columns:
        level_counts = dataframe["pred_level"].fillna("").astype(str).value_counts()
        for level, count in level_counts.items():
            if level:
                rows.append({"item": f"level::{level}", "value": int(count)})

    summary_df = pd.DataFrame(rows)
    summary_df["item"] = summary_df["item"].map(lambda value: _translate_summary_item(str(value)))
    return summary_df.rename(columns=SUMMARY_COLUMNS)


def _translate_summary_item(item: str) -> str:
    if item in SUMMARY_ITEM_DISPLAY_MAP:
        return SUMMARY_ITEM_DISPLAY_MAP[item]
    if item.startswith("priority_"):
        return f"{item.split('_', 1)[1]}优先级数量"
    if item.startswith("category::"):
        return f"问题类别::{item.split('::', 1)[1]}"
    if item.startswith("level::"):
        return f"问题等级::{item.split('::', 1)[1]}"
    if item.startswith("revision::"):
        return f"修订需求类型::{item.split('::', 1)[1]}"
    if item.startswith("kg_match::"):
        return f"图谱匹配状态::{item.split('::', 1)[1]}"
    if item == "revision_need_true":
        return "疑似标准修订需求数量"
    return item


def _localize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    rename_map = {column: DISPLAY_NAME_MAP[column] for column in dataframe.columns if column in DISPLAY_NAME_MAP}
    return dataframe.rename(columns=rename_map)


def _format_workbook(writer: pd.ExcelWriter, sheet_names: list[str]) -> None:
    workbook = writer.book
    for sheet_name in sheet_names:
        sheet = workbook[sheet_name]
        sheet.freeze_panes = "A2"
        sheet.auto_filter.ref = sheet.dimensions
        for column_cells in sheet.columns:
            header = str(column_cells[0].value or "")
            if header in LONG_TEXT_COLUMNS:
                width = 60
            elif header == "复核备注":
                width = 36
            elif header == "模型版本":
                width = 48
            else:
                width = min(max(len(str(header or "")) + 4, 12), 28)
            sheet.column_dimensions[column_cells[0].column_letter].width = width
        for row in sheet.iter_rows():
            for cell in row:
                updated_alignment = copy(cell.alignment)
                updated_alignment.wrap_text = True
                updated_alignment.vertical = "top"
                cell.alignment = updated_alignment


def export_review_sheet(input_file: str, output_file: str | None = None) -> Path:
    input_path = resolve_path(input_file)
    if input_path is None or not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    output_path = _resolve_output_path(input_path, output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = read_table(input_path)
    review_df = _build_review_dataframe(dataframe)
    review_mask_column = "need_review" if "need_review" in review_df.columns else "need_manual_review"
    need_review_df = review_df[review_df[review_mask_column]].copy()
    summary_df = _build_summary(review_df)
    review_df_localized = _localize_columns(review_df)
    need_review_localized = _localize_columns(need_review_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        need_review_localized.to_excel(writer, sheet_name="need_review", index=False)
        review_df_localized.to_excel(writer, sheet_name="all_predictions", index=False)
        _format_workbook(writer, ["summary", "need_review", "all_predictions"])

    return output_path


def main() -> None:
    args = parse_args()
    output_path = export_review_sheet(args.input_file, args.output_file)
    print(f"Human-friendly review sheet saved to: {output_path}")


if __name__ == "__main__":
    main()
