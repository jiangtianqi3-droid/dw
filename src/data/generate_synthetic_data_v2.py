from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


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

TEST_ITEMS = ["交接试验", "预防性试验", "耐压试验", "局放检测", "功能校验", "核容试验", "联调联试"]
RECORD_DOCS = ["原始记录", "试验报告", "巡检台账", "验收问题清单", "异常处置单", "操作记录"]
RECORD_FIELDS = ["责任人", "时间戳", "处理依据", "复验结论", "检查明细", "签字页"]
STANDARD_RULES = ["抽检比例", "执行周期", "双签要求", "判定口径", "填报模板", "见证点要求"]
ACCEPTANCE_ACTIONS = ["投运验收", "检修验收", "工程验收", "隐蔽验收", "功能验收", "联动验收"]
OBJECTS = ["关键风险点", "关键试验结果", "专项复验项", "全量回路", "图纸一致性", "见证资料"]

COMMON_INTROS = [
    "现场抽查发现",
    "专项监督中发现",
    "复核时发现",
    "监督检查显示",
    "班组自查反映",
]

DISTRACTOR_POOL = [
    "相关台账已补录但未形成有效闭环",
    "班组口头说明此前一直按旧版要求执行",
    "验收资料已提交但未说明复核依据",
    "现场人员表示后续计划补测",
    "报告中已写明整改计划但未说明完成时间",
]


@dataclass(frozen=True)
class Context:
    device_type: str
    specialty: str
    supervision_stage: str
    source_unit: str
    event_time: str
    test_item: str
    record_doc: str
    record_field: str
    standard_rule: str
    acceptance_action: str
    obj: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate harder synthetic data and hard test set.")
    parser.add_argument("--output-dir", type=str, default="data/generated", help="Output directory.")
    parser.add_argument("--samples-per-label", type=int, default=300, help="Training samples per label.")
    parser.add_argument("--hard-test-per-label", type=int, default=40, help="Hard test samples per label.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _pick_context(randomizer: random.Random) -> Context:
    device_type, specialty, stages = randomizer.choice(DEVICE_OPTIONS)
    return Context(
        device_type=device_type,
        specialty=specialty,
        supervision_stage=randomizer.choice(stages),
        source_unit=randomizer.choice(SOURCE_UNITS),
        event_time=(date(2024, 1, 1) + timedelta(days=randomizer.randint(0, 700))).isoformat(),
        test_item=randomizer.choice(TEST_ITEMS),
        record_doc=randomizer.choice(RECORD_DOCS),
        record_field=randomizer.choice(RECORD_FIELDS),
        standard_rule=randomizer.choice(STANDARD_RULES),
        acceptance_action=randomizer.choice(ACCEPTANCE_ACTIONS),
        obj=randomizer.choice(OBJECTS),
    )


def _compose_text(label: str, context: Context, randomizer: random.Random) -> str:
    intro = randomizer.choice(COMMON_INTROS)
    distractor = randomizer.choice(DISTRACTOR_POOL)

    if label == "试验检测缺失":
        core_options = [
            f"{context.device_type}{context.test_item}虽已立项，但关键项目实际未实施",
            f"{context.device_type}{context.test_item}仅覆盖部分{context.obj}，其余项目未见补测",
            f"{context.supervision_stage}后{context.device_type}仍未补做{context.test_item}",
            f"{context.device_type}{context.test_item}记录已形成，但实测项目存在漏检漏试",
            f"{context.device_type}{context.test_item}执行到一半即结束，关键工况未覆盖",
        ]
    elif label == "标准执行不到位":
        core_options = [
            f"{context.device_type}{context.test_item}虽已完成，但{context.standard_rule}仍沿用旧版要求",
            f"{context.device_type}相关工作已开展，但{context.standard_rule}未按现行标准执行",
            f"{context.supervision_stage}环节已安排检查，但现场仍以经验替代标准条款",
            f"{context.device_type}并非未做工作，而是执行口径与现行标准不一致",
            f"{context.device_type}{context.standard_rule}落实不到位，班组未完成新标准切换",
        ]
    elif label == "记录台账不规范":
        core_options = [
            f"{context.device_type}{context.record_doc}缺少{context.record_field}，无法支撑后续追溯",
            f"{context.device_type}{context.record_doc}仅保留结论，未附对应检查明细",
            f"{context.device_type}{context.record_doc}与现场时间不一致，且未签字确认",
            f"{context.supervision_stage}形成的{context.record_doc}存在漏填漏签问题",
            f"{context.device_type}{context.test_item}已实施，但{context.record_doc}关键信息缺失",
        ]
    else:
        core_options = [
            f"{context.device_type}{context.acceptance_action}时未核对{context.obj}即出具通过意见",
            f"{context.device_type}{context.acceptance_action}仅抽查部分{context.obj}便办理验收",
            f"{context.supervision_stage}结束后未组织专项复验即完成验收签认",
            f"{context.device_type}相关资料虽已提交，但验收把关流于形式",
            f"{context.device_type}{context.acceptance_action}未覆盖关键见证点仍予放行",
        ]

    style = randomizer.randint(0, 4)
    core = randomizer.choice(core_options)

    if style == 0:
        return f"{intro}，{core}。"
    if style == 1:
        return f"{intro}，{distractor}，但{core}。"
    if style == 2:
        return f"{context.specialty}专业{context.device_type}在{context.supervision_stage}环节中，{core}。"
    if style == 3:
        return f"{context.device_type}{context.test_item}相关资料表面齐全，实际情况是：{core}。"
    return f"{context.source_unit}反馈，{core}；同时{distractor}。"


def _build_record(index: int, label: str, randomizer: random.Random, data_source: str) -> dict:
    context = _pick_context(randomizer)

    record = {
        "sample_id": f"{data_source}_{label}_{index:04d}",
        "text": _compose_text(label, context, randomizer),
        "label": label,
        "device_type": context.device_type,
        "specialty": context.specialty,
        "supervision_stage": context.supervision_stage,
        "event_time": context.event_time,
        "source_unit": context.source_unit,
        "data_source": data_source,
    }

    if randomizer.random() < 0.12:
        record["device_type"] = ""
    if randomizer.random() < 0.12:
        record["specialty"] = ""
    if randomizer.random() < 0.18:
        record["supervision_stage"] = ""

    return record


def build_synthetic_v2(samples_per_label: int, seed: int) -> pd.DataFrame:
    randomizer = random.Random(seed)
    records: list[dict] = []
    seen_texts: set[str] = set()

    for label in LABELS:
        index = 0
        attempts = 0
        while index < samples_per_label:
            record = _build_record(index + 1, label, randomizer, data_source="synthetic_v2")
            attempts += 1
            if record["text"] in seen_texts:
                if attempts > samples_per_label * 20:
                    raise RuntimeError(f"Too many duplicate texts while generating label: {label}")
                continue
            seen_texts.add(record["text"])
            records.append(record)
            index += 1

    randomizer.shuffle(records)
    return pd.DataFrame(records)


def _build_hard_text(label: str, context: Context, randomizer: random.Random) -> str:
    if label == "试验检测缺失":
        options = [
            f"{context.device_type}{context.test_item}台账和报告均已建立，但{context.obj}对应检测项目实际未做，后续也未补测。",
            f"{context.device_type}{context.acceptance_action}资料齐全，不过支撑结论的{context.test_item}关键项目并未实施。",
            f"{context.device_type}并非记录缺失，而是{context.test_item}本身只做了部分回路，关键工况未覆盖。",
        ]
    elif label == "标准执行不到位":
        options = [
            f"{context.device_type}{context.test_item}已经实施且留痕完整，但{context.standard_rule}仍按旧版标准执行。",
            f"{context.device_type}并非未做试验，而是{context.standard_rule}与现行标准条款不一致。",
            f"{context.device_type}{context.acceptance_action}流程齐全，不过现场执行仍沿用历史口径替代标准要求。",
        ]
    elif label == "记录台账不规范":
        options = [
            f"{context.device_type}{context.test_item}已经完成且满足标准要求，但{context.record_doc}缺少{context.record_field}。",
            f"{context.device_type}实际工作已开展，问题在于{context.record_doc}只写结论未附过程明细。",
            f"{context.device_type}并非验收把关问题，而是{context.record_doc}时间、签字与现场不一致。",
        ]
    else:
        options = [
            f"{context.device_type}{context.test_item}和相关记录均已提供，但验收人员未核对{context.obj}就签署通过意见。",
            f"{context.device_type}并非标准执行偏差，而是在{context.acceptance_action}时仅抽查部分{context.obj}便放行。",
            f"{context.device_type}相关台账和试验资料都有，问题是验收环节未组织必要复验即办理通过。",
        ]
    return randomizer.choice(options)


def build_hard_test(samples_per_label: int, seed: int) -> pd.DataFrame:
    randomizer = random.Random(seed + 1000)
    records: list[dict] = []
    seen_texts: set[str] = set()

    for label in LABELS:
        index = 0
        attempts = 0
        while index < samples_per_label:
            context = _pick_context(randomizer)
            record = {
                "sample_id": f"hard_test_v1_{label}_{index + 1:04d}",
                "text": _build_hard_text(label, context, randomizer),
                "label": label,
                "device_type": context.device_type if randomizer.random() >= 0.2 else "",
                "specialty": context.specialty if randomizer.random() >= 0.2 else "",
                "supervision_stage": context.supervision_stage if randomizer.random() >= 0.25 else "",
                "event_time": context.event_time,
                "source_unit": context.source_unit,
                "data_source": "hard_test_v1",
            }
            attempts += 1
            if record["text"] in seen_texts:
                if attempts > samples_per_label * 30:
                    raise RuntimeError(f"Too many duplicate texts while generating hard label: {label}")
                continue
            seen_texts.add(record["text"])
            records.append(record)
            index += 1

    randomizer.shuffle(records)
    return pd.DataFrame(records)


def _save_jsonl(dataframe: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in dataframe.to_dict(orient="records"):
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_outputs(
    synthetic_v2_df: pd.DataFrame,
    hard_test_df: pd.DataFrame,
    output_dir: str | Path,
    seed: int,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df, temp_df = train_test_split(
        synthetic_v2_df,
        train_size=0.8,
        random_state=seed,
        stratify=synthetic_v2_df["label"],
    )
    valid_df, test_df = train_test_split(
        temp_df,
        train_size=0.5,
        random_state=seed,
        stratify=temp_df["label"],
    )

    paths = {
        "synthetic_v2_csv": output_dir / "issues_synthetic_v2.csv",
        "synthetic_v2_jsonl": output_dir / "issues_synthetic_v2.jsonl",
        "synthetic_v2_train_csv": output_dir / "train_synthetic_v2.csv",
        "synthetic_v2_valid_csv": output_dir / "valid_synthetic_v2.csv",
        "synthetic_v2_test_csv": output_dir / "test_synthetic_v2.csv",
        "hard_test_csv": output_dir / "hard_test_v1.csv",
        "hard_test_jsonl": output_dir / "hard_test_v1.jsonl",
    }

    synthetic_v2_df.to_csv(paths["synthetic_v2_csv"], index=False, encoding="utf-8-sig")
    train_df.reset_index(drop=True).to_csv(paths["synthetic_v2_train_csv"], index=False, encoding="utf-8-sig")
    valid_df.reset_index(drop=True).to_csv(paths["synthetic_v2_valid_csv"], index=False, encoding="utf-8-sig")
    test_df.reset_index(drop=True).to_csv(paths["synthetic_v2_test_csv"], index=False, encoding="utf-8-sig")
    hard_test_df.to_csv(paths["hard_test_csv"], index=False, encoding="utf-8-sig")

    _save_jsonl(synthetic_v2_df, paths["synthetic_v2_jsonl"])
    _save_jsonl(hard_test_df, paths["hard_test_jsonl"])
    return paths


def main() -> None:
    args = parse_args()
    synthetic_v2_df = build_synthetic_v2(samples_per_label=args.samples_per_label, seed=args.seed)
    hard_test_df = build_hard_test(samples_per_label=args.hard_test_per_label, seed=args.seed)
    paths = save_outputs(synthetic_v2_df, hard_test_df, output_dir=args.output_dir, seed=args.seed)

    print(f"Synthetic v2 samples: {len(synthetic_v2_df)}")
    print(f"Hard test samples: {len(hard_test_df)}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
