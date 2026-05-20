from __future__ import annotations

import argparse

import pandas as pd

from src.data.io import read_table, write_table
from src.data.prepare_real_data import DEFAULT_RAW_INPUT_FILE, build_real_dataset
from src.utils.common import save_json
from src.utils.config import resolve_path


DEFAULT_CATEGORY_LABELS = [
    "设计与选型",
    "采购与制造",
    "安装与施工",
    "调试与试验",
    "验收与交接",
    "资料与标识",
    "运维与环境",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare problem-category training data from real supervision records.")
    parser.add_argument(
        "--input-file",
        type=str,
        default=DEFAULT_RAW_INPUT_FILE,
        help="Raw Excel/CSV/JSONL file. Defaults to the project-local real-data file.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/real/processed/pytest_processed_1_problem_category_v1.csv",
        help="Prepared CSV/JSONL/XLSX file.",
    )
    parser.add_argument(
        "--stats-file",
        type=str,
        default="data/real/processed/pytest_processed_1_problem_category_v1_stats.json",
        help="Path to output data statistics JSON.",
    )
    return parser.parse_args()


def _normalize(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def assign_problem_category(record: pd.Series) -> str:
    text = _normalize(record.get("text", ""))
    problem_reason = _normalize(record.get("problem_reason", ""))
    problem_stage = _normalize(record.get("problem_stage", ""))
    supervision_stage = _normalize(record.get("supervision_stage", ""))
    rule_name = _normalize(record.get("rule_name", ""))
    checkpoint_text = _normalize(record.get("checkpoint_text", ""))
    merged = " ".join(
        item for item in [text, problem_reason, problem_stage, supervision_stage, rule_name, checkpoint_text] if item
    )

    # 规则顺序很重要：先判边界最清晰、最容易和其他类混开的类别。
    if any(
        keyword in merged
        for keyword in ["台账", "记录", "报告", "标识", "标牌", "铭牌", "签字", "签名", "编号", "归档", "资料", "文档"]
    ):
        return "资料与标识"

    if (
        problem_reason in {"工程设计", "工程规划", "设备选型"}
        or problem_stage in {"工程设计", "规划可研"}
        or supervision_stage in {"工程设计", "规划可研"}
        or any(keyword in merged for keyword in ["设计图纸", "选型", "布置", "容量配置", "设计不合理"])
    ):
        return "设计与选型"

    if (
        problem_reason in {"设备制造工艺", "设备材质", "出厂试验"}
        or problem_stage in {"设备制造", "设备采购"}
        or supervision_stage == "设备采购"
        or any(keyword in merged for keyword in ["出厂", "材质", "加工工艺", "采购技术协议", "制造"])
    ):
        return "采购与制造"

    if (
        problem_reason in {"设备运维", "运行环境"}
        or any(keyword in merged for keyword in ["渗漏", "温升", "腐蚀", "噪声", "振动", "环境", "通风", "消防"])
    ):
        return "运维与环境"

    if (
        problem_stage in {"设备验收", "竣工验收"}
        or supervision_stage == "竣工验收"
        or any(keyword in merged for keyword in ["交接试验", "启动验收", "投运前验收", "验收移交"])
    ):
        return "验收与交接"

    if (
        problem_reason == "现场安装"
        or problem_stage == "设备安装"
        or (
            supervision_stage == "安装调试"
            and any(
                keyword in merged
                for keyword in [
                    "接线",
                    "接地",
                    "封堵",
                    "螺栓",
                    "焊接",
                    "支架",
                    "基础",
                    "敷设",
                    "相序",
                    "防火封堵",
                    "端子",
                    "电缆",
                    "安装偏差",
                ]
            )
        )
        or any(keyword in merged for keyword in ["施工", "孔洞封堵", "电缆敷设", "螺栓松动", "焊缝", "二次接线"])
    ):
        return "安装与施工"

    if (
        problem_reason == "设备调试"
        or problem_stage == "设备调试"
        or any(
            keyword in merged
            for keyword in ["调试", "试验", "校验", "联调", "定值", "二次回路", "保护动作", "绝缘电阻", "耐压", "局放"]
        )
    ):
        return "调试与试验"

    if any(keyword in merged for keyword in ["验收", "投运前", "交接"]):
        return "验收与交接"

    return "安装与施工"


def build_problem_category_dataset(raw_dataframe: pd.DataFrame) -> pd.DataFrame:
    prepared = build_real_dataset(raw_dataframe, allowed_labels=["一般", "较大", "重大"])
    # 问题等级先保留下来，后面可作为多任务标签或误差分析辅助字段。
    prepared = prepared.rename(columns={"label": "problem_level"}).copy()
    prepared["label"] = prepared.apply(assign_problem_category, axis=1)
    prepared["data_source"] = prepared["data_source"].map(
        lambda value: f"{value}|problem_category_rule_v1" if value else "problem_category_rule_v1"
    )
    return prepared


def build_stats(prepared_df: pd.DataFrame) -> dict:
    return {
        "prepared_rows": int(len(prepared_df)),
        "category_distribution": prepared_df["label"].value_counts().to_dict(),
        "problem_level_distribution": prepared_df["problem_level"].value_counts().to_dict(),
        "cross_tab_category_vs_level": pd.crosstab(prepared_df["label"], prepared_df["problem_level"]).to_dict(),
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

    raw_df = read_table(input_path)
    prepared_df = build_problem_category_dataset(raw_df)
    write_table(prepared_df, output_path)
    save_json(build_stats(prepared_df), stats_path)

    print(f"Prepared problem-category dataset saved to: {output_path}")
    print(f"Rows: {len(prepared_df)}")
    print(f"Category distribution: {prepared_df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
