from __future__ import annotations

import argparse

import pandas as pd

from src.data.io import read_table, write_table
from src.utils.common import save_json
from src.utils.config import resolve_path


DEFAULT_RAW_INPUT_FILE = "data/real/raw/pytest_processed_1.xlsx"
DEFAULT_ALLOWED_LABELS = ["一般", "较大", "重大"]
UNKNOWN_TOKENS = {"", "nan", "none", "null", "未知"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare real supervision issue data from Excel.")
    parser.add_argument(
        "--input-file",
        type=str,
        default=DEFAULT_RAW_INPUT_FILE,
        help="Raw Excel/CSV/JSONL file. Defaults to the project-local real-data file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/real/processed/pytest_processed_1_problem_level_v1.csv",
        help="Prepared CSV/JSONL/XLSX file.",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default="data/real/processed/pytest_processed_1_problem_level_v1_stats.json",
        help="Path to output data statistics JSON.",
    )
    parser.add_argument(
        "--allowed-labels",
        type=str,
        default=",".join(DEFAULT_ALLOWED_LABELS),
        help="Comma-separated labels kept for training.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    text = str(value).replace("\r", "\n").strip()
    text = " ".join(part for part in text.split())
    return "" if text.lower() in UNKNOWN_TOKENS else text


def normalize_date(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    try:
        timestamp = pd.to_datetime(value)
    except (TypeError, ValueError):
        return normalize_text(value)
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m-%d")


def build_real_dataset(dataframe: pd.DataFrame, allowed_labels: list[str]) -> pd.DataFrame:
    # 先把原始 Excel 的中文列名统一映射成训练链路使用的英文列名。
    column_map = {
        "工程性质": "project_type",
        "设备名称": "device_name",
        "电压等级": "voltage_level",
        "设备型号": "device_model",
        "实物ID": "asset_id",
        "设备类型": "device_type",
        "问题原因": "problem_reason",
        "发现时间": "event_time",
        "问题产生阶段": "problem_stage",
        "问题产生环节": "supervision_stage",
        "细则问题等级": "rule_level",
        "问题等级": "label",
        "问题描述": "text",
        "是否质量问题": "is_quality_issue",
        "监督专业": "specialty",
        "问题原因分析": "root_cause_analysis",
        "监督意见": "supervision_opinion",
        "处理时间": "resolved_time",
        "实际整改措施": "actual_fix",
        "细则名称": "rule_name",
        "阶段名称": "stage_name",
        "所属环节": "stage_link",
        "条款序号": "clause_no",
        "监督要点序号": "checkpoint_no",
        "监督要点": "checkpoint_text",
    }

    missing_columns = [column for column in column_map if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in raw data: {missing_columns}")

    prepared = dataframe.rename(columns=column_map).copy()

    text_columns = [
        "project_type",
        "device_name",
        "voltage_level",
        "device_model",
        "asset_id",
        "device_type",
        "problem_reason",
        "problem_stage",
        "supervision_stage",
        "rule_level",
        "label",
        "text",
        "is_quality_issue",
        "specialty",
        "root_cause_analysis",
        "supervision_opinion",
        "actual_fix",
        "rule_name",
        "stage_name",
        "stage_link",
        "clause_no",
        "checkpoint_no",
        "checkpoint_text",
    ]
    for column in text_columns:
        prepared[column] = prepared[column].map(normalize_text)

    prepared["event_time"] = prepared["event_time"].map(normalize_date)
    prepared["resolved_time"] = prepared["resolved_time"].map(normalize_date)
    prepared["source_unit"] = "未知"
    prepared["data_source"] = "pytest_processed_1_real"

    valid_mask = prepared["text"].ne("") & prepared["label"].isin(allowed_labels)
    prepared = prepared.loc[valid_mask].copy()
    prepared = prepared.drop_duplicates(subset=["text", "label", "device_type", "specialty", "supervision_stage"])
    prepared = prepared.reset_index(drop=True)
    # sample_id 用于后续反馈回流、图谱导出和人工复核定位。
    prepared.insert(0, "sample_id", [f"real_v1_{idx:06d}" for idx in range(1, len(prepared) + 1)])
    return prepared


def build_stats(raw_df: pd.DataFrame, prepared_df: pd.DataFrame) -> dict:
    return {
        "raw_rows": int(len(raw_df)),
        "prepared_rows": int(len(prepared_df)),
        "dropped_rows": int(len(raw_df) - len(prepared_df)),
        "label_distribution": prepared_df["label"].value_counts().to_dict(),
        "specialty_distribution_top10": prepared_df["specialty"].value_counts().head(10).to_dict(),
        "supervision_stage_distribution_top10": prepared_df["supervision_stage"].value_counts().head(10).to_dict(),
    }


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input_file)
    output_path = resolve_path(args.output_file)
    stats_path = resolve_path(args.stats_file)

    if input_path is None or not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input_file}. "
            f"Expected a raw dataset under {DEFAULT_RAW_INPUT_FILE} or pass --input-file explicitly."
        )

    allowed_labels = [item.strip() for item in args.allowed_labels.split(",") if item.strip()]
    raw_df = read_table(input_path)
    prepared_df = build_real_dataset(raw_df, allowed_labels=allowed_labels)

    write_table(prepared_df, output_path)
    save_json(build_stats(raw_df, prepared_df), stats_path)

    print(f"Prepared real dataset saved to: {output_path}")
    print(f"Rows: raw={len(raw_df)}, prepared={len(prepared_df)}")
    print(f"Label distribution: {prepared_df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
