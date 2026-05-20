from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


LABELS = [
    "试验检测缺失",
    "标准执行不到位",
    "记录台账不规范",
    "验收把关不严",
]

DEVICE_OPTIONS = [
    ("主变压器", "一次设备", ["试验检测", "大修验收", "状态监测"]),
    ("GIS设备", "一次设备", ["交接验收", "试验检测", "设备检修"]),
    ("断路器", "一次设备", ["试验检测", "检修验收", "缺陷处理"]),
    ("避雷器", "一次设备", ["在线监测", "试验检测", "异常处置"]),
    ("继电保护装置", "二次设备", ["定期检验", "工程验收", "运行操作"]),
    ("二次回路", "二次设备", ["现场作业", "技改验收", "运行维护"]),
    ("直流系统", "二次设备", ["运维监督", "日常巡检", "试验检测"]),
    ("防误闭锁装置", "二次设备", ["改造验收", "运行维护", "联调联试"]),
    ("电缆线路", "输电", ["试验检测", "检修验收", "隐患排查"]),
    ("架空线路", "输电", ["缺陷管理", "竣工验收", "日常巡检"]),
    ("接地网", "土建", ["现场检查", "隐蔽验收", "整改复验"]),
    ("开关柜", "配电", ["到货验收", "交接验收", "投运验收"]),
    ("配电自动化终端", "配电", ["到货验收", "功能验收", "运行维护"]),
    ("电能表", "营销", ["计划管理", "轮换验收", "抽检监督"]),
]

SOURCE_UNITS = [
    "运检一班",
    "运维二班",
    "检修中心",
    "继保班",
    "试验班",
    "输电运检室",
    "配电工程部",
    "设备检修部",
    "设备管理部",
    "基建项目部",
    "变电运维站",
    "营销运维班",
]


@dataclass(frozen=True)
class LabelTemplate:
    observations: list[str]
    behaviors: list[str]
    objects: list[str]
    suffixes: list[str]


TEMPLATES = {
    "试验检测缺失": LabelTemplate(
        observations=["未开展", "缺少", "漏做", "未覆盖", "仅完成部分", "超周期仍未实施"],
        behaviors=["预防性试验", "交接试验", "专项复验", "功能校验", "状态检测", "耐压试验", "核容试验"],
        objects=["关键项目", "规定项目", "高风险回路", "全部间隔", "必要工况", "重点测点"],
        suffixes=[
            "与现行标准要求不符",
            "未见补测安排",
            "导致结果无法满足验收依据",
            "且原始记录无说明",
            "现场未提供替代依据",
        ],
    ),
    "标准执行不到位": LabelTemplate(
        observations=["未按", "仍沿用", "简化处理", "执行频次低于", "抽检比例低于", "未落实"],
        behaviors=["现行标准", "最新标准条款", "双签要求", "周期要求", "作业流程", "判定规则", "管理模板"],
        objects=["实施步骤", "审核流程", "巡检周期", "抽检口径", "验收规则", "填报要求"],
        suffixes=[
            "与标准要求存在偏差",
            "现场人员解释依据不充分",
            "相关班组未完成切换宣贯",
            "仍按历史经验执行",
            "导致执行口径不一致",
        ],
    ),
    "记录台账不规范": LabelTemplate(
        observations=["未填写", "漏填", "缺少", "未签字确认", "时间不一致", "仅保留结论"],
        behaviors=["原始记录", "巡检台账", "验收问题清单", "异常处置单", "操作记录", "检测报告"],
        objects=["处理依据", "复验结论", "关键字段", "责任人信息", "检查明细", "时间戳"],
        suffixes=[
            "台账完整性不足",
            "无法支撑后续追溯",
            "扫描件质量较差",
            "整改闭环信息缺失",
            "不满足留痕管理要求",
        ],
    ),
    "验收把关不严": LabelTemplate(
        observations=["仅抽查", "未核对", "未组织", "资料未齐套即", "未完成复验即", "未覆盖"],
        behaviors=["投运验收", "检修验收", "工程验收", "功能验收", "联动验收", "隐蔽验收"],
        objects=["关键风险点", "图纸一致性", "全量回路", "专项复验", "关键试验结果", "必要见证资料"],
        suffixes=[
            "仍出具通过意见",
            "验收把关流于形式",
            "会议纪要未见异议记录",
            "未形成风险复核闭环",
            "关键见证点缺失",
        ],
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic issue classification data.")
    parser.add_argument("--output-dir", type=str, default="data/generated", help="Output directory.")
    parser.add_argument("--samples-per-label", type=int, default=200, help="Samples per label.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _pick_device(randomizer: random.Random) -> tuple[str, str, str]:
    device_type, specialty, stages = randomizer.choice(DEVICE_OPTIONS)
    return device_type, specialty, randomizer.choice(stages)


def _build_issue_text(
    label: str,
    device_type: str,
    specialty: str,
    supervision_stage: str,
    randomizer: random.Random,
) -> str:
    template = TEMPLATES[label]
    observation = randomizer.choice(template.observations)
    behavior = randomizer.choice(template.behaviors)
    obj = randomizer.choice(template.objects)
    suffix = randomizer.choice(template.suffixes)

    patterns = [
        f"{device_type}{behavior}{observation}{obj}，{suffix}",
        f"{supervision_stage}过程中{device_type}{behavior}{observation}{obj}，{suffix}",
        f"{specialty}专业{device_type}{behavior}{observation}{obj}，{suffix}",
        f"{device_type}在{supervision_stage}环节{behavior}{observation}{obj}，{suffix}",
        f"现场检查发现{device_type}{behavior}{observation}{obj}，{suffix}",
    ]
    return randomizer.choice(patterns)


def _build_record(index: int, label: str, randomizer: random.Random) -> dict:
    device_type, specialty, supervision_stage = _pick_device(randomizer)
    source_unit = randomizer.choice(SOURCE_UNITS)
    event_time = date(2024, 1, 1) + timedelta(days=randomizer.randint(0, 540))

    return {
        "sample_id": f"synthetic_v1_{label}_{index:04d}",
        "text": _build_issue_text(label, device_type, specialty, supervision_stage, randomizer),
        "label": label,
        "device_type": device_type,
        "specialty": specialty,
        "supervision_stage": supervision_stage,
        "event_time": event_time.isoformat(),
        "source_unit": source_unit,
        "data_source": "synthetic_v1",
    }


def build_dataset(samples_per_label: int, seed: int) -> pd.DataFrame:
    randomizer = random.Random(seed)
    records: list[dict] = []

    for label in LABELS:
        for index in range(samples_per_label):
            records.append(_build_record(index=index + 1, label=label, randomizer=randomizer))

    randomizer.shuffle(records)
    return pd.DataFrame(records)


def save_dataset(dataframe: pd.DataFrame, output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "issues_synthetic_v1.csv"
    jsonl_path = output_dir / "issues_synthetic_v1.jsonl"

    dataframe.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with jsonl_path.open("w", encoding="utf-8") as file:
        for record in dataframe.to_dict(orient="records"):
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    return csv_path, jsonl_path


def main() -> None:
    args = parse_args()
    dataframe = build_dataset(samples_per_label=args.samples_per_label, seed=args.seed)
    csv_path, jsonl_path = save_dataset(dataframe, args.output_dir)

    print(f"Generated {len(dataframe)} samples.")
    print(f"CSV saved to: {csv_path}")
    print(f"JSONL saved to: {jsonl_path}")


if __name__ == "__main__":
    main()
