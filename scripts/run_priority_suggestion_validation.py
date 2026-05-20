from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import read_table, write_table
from src.export_review_sheet import export_review_sheet
from src.utils.config import load_config, resolve_path
from src.utils.decision_support import build_enriched_predictions, get_category_model_settings
from src.utils.feature_matrix import build_feature_matrix
from src.utils.feedback import attach_feedback_columns
from src.utils.graph_export import build_graph_export_dataframe
from src.utils.model_runtime import load_model_runtime, predict_dataframe


def _series_distribution(series: pd.Series) -> dict[str, int]:
    return {str(key): int(value) for key, value in series.fillna("").astype(str).value_counts().items() if str(key).strip()}


def _non_empty_rate(series: pd.Series) -> float:
    normalized = series.fillna("").astype(str).str.strip()
    return round(float(normalized.ne("").mean()), 4)


def _format_dict_block(data: dict[str, int]) -> str:
    if not data:
        return "- 无"
    return "\n".join(f"- {key}: {value}" for key, value in data.items())


def _format_top10(dataframe: pd.DataFrame) -> str:
    if dataframe.empty:
        return "无高优先级样本。"

    columns = [
        "sample_id",
        "pred_level",
        "pred_category",
        "priority_score",
        "priority_label",
        "need_review",
        "text",
    ]
    available = [column for column in columns if column in dataframe.columns]
    lines = []
    for _, row in dataframe[available].head(10).iterrows():
        lines.append(
            f"- {row.get('sample_id', '')} | 等级={row.get('pred_level', '')} | "
            f"类别={row.get('pred_category', '')} | 优先级={row.get('priority_label', '')} "
            f"({row.get('priority_score', '')}) | 复核={row.get('need_review', '')} | "
            f"问题={str(row.get('text', ''))[:80]}"
        )
    return "\n".join(lines)


def main() -> None:
    config_path = "configs/real_problem_level_v1.yaml"
    config = load_config(config_path)
    input_path = resolve_path(config["data"]["input_path"])
    model_path = resolve_path("artifacts/outputs_real_problem_level_v1/best_model")
    output_dir = resolve_path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_output = output_dir / "decision_extension_sample.csv"
    enriched_output = output_dir / "predictions_enriched.csv"
    feature_output = output_dir / "feature_matrix.csv"
    review_output = output_dir / "predictions_review_human.xlsx"
    graph_output = output_dir / "graph_records.jsonl"
    report_output = output_dir / "decision_extension_report.md"

    source_df = read_table(input_path).head(40).copy()
    write_table(source_df, sample_output)

    runtime = load_model_runtime(config, model_path)
    predicted_df, predictions, _ = predict_dataframe(source_df, runtime=runtime)
    predicted_df["predicted_label_id"] = [item["predicted_label_id"] for item in predictions]
    predicted_df["predicted_label"] = [item["predicted_label"] for item in predictions]
    predicted_df["confidence"] = [item["confidence"] for item in predictions]
    predicted_df = attach_feedback_columns(
        dataframe=predicted_df,
        model_path=str(runtime.model_path),
        config=config,
        include_model_input=True,
    )

    category_config, category_model_path = get_category_model_settings(config)
    enriched_df = build_enriched_predictions(
        dataframe=predicted_df,
        config=config,
        category_config_path=category_config,
        category_model_path=category_model_path,
        device=runtime.device,
    )
    write_table(enriched_df, enriched_output)

    feature_df = build_feature_matrix(
        input_df=source_df,
        level_config=config,
        level_model_path=str(model_path),
        category_config_path=category_config,
        category_model_path=category_model_path,
    )
    write_table(feature_df, feature_output)

    export_review_sheet(str(enriched_output), str(review_output))

    graph_df = build_graph_export_dataframe(enriched_df, config=config, label_role="level")
    write_table(graph_df, graph_output)

    high_priority_df = enriched_df.sort_values(by="priority_score", ascending=False)
    field_non_empty_rate = {}
    for column in [
        "pred_level",
        "pred_category",
        "level_confidence",
        "category_confidence",
        "priority_score",
        "priority_label",
        "priority_reason",
        "suggestion",
        "suggestion_basis",
        "recommended_action",
    ]:
        if column in enriched_df.columns:
            field_non_empty_rate[column] = _non_empty_rate(enriched_df[column])

    report = f"""# 决策增强验证报告

## 1. 输入文件路径
- 原始输入：{input_path}
- 小样本输入：{sample_output}

## 2. 输出文件路径
- enriched predictions：{enriched_output}
- feature matrix：{feature_output}
- review excel：{review_output}
- graph jsonl：{graph_output}
- markdown report：{report_output}

## 3. 样本数量
- 样本总数：{len(enriched_df)}

## 4. 问题等级分布
{_format_dict_block(_series_distribution(enriched_df.get("pred_level", pd.Series(dtype=str))))}

## 5. 问题类别分布
{_format_dict_block(_series_distribution(enriched_df.get("pred_category", pd.Series(dtype=str))))}

## 6. 优先级分布
{_format_dict_block(_series_distribution(enriched_df.get("priority_label", pd.Series(dtype=str))))}

## 7. need_review 数量
- need_review=true：{int(enriched_df.get("need_review", pd.Series(False)).fillna(False).astype(bool).sum())}

## 8. 高优先级样本 Top 10
{_format_top10(high_priority_df)}

## 9. 新增字段非空率
{_format_dict_block({key: int(value * 10000) / 100 for key, value in field_non_empty_rate.items()})}

## 10. 当前模块局限说明
- 当前优先级评估是规则加权版本，尚未引入统计学习或专家动态校准。
- 问题类别依赖外部 `dw1.2.5` 类别模型；若该模型不可用，将退化为仅基于等级与结构化字段的决策增强。
- 建议生成目前为规则模板，适合做初步决策支持，暂不替代人工专业判断。
- 复合特征矩阵当前导出的是单模型 CLS 向量，尚未加入多层特征融合或历史时序特征。
"""
    report_output.write_text(report, encoding="utf-8")
    print(f"Decision extension validation finished. Report saved to: {report_output}")


if __name__ == "__main__":
    main()
