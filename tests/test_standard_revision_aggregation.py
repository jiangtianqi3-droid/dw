from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils.standard_revision_aggregation import (
    aggregate_standard_revision_priority,
    build_standard_revision_report,
    write_json_summary,
)


class StandardRevisionAggregationTest(unittest.TestCase):
    def _sample_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "problem_id": "P1",
                    "问题描述": "GIS 接地线未按标准采用螺栓连接",
                    "设备类型": "组合电器",
                    "监督专业": "电气",
                    "所属单位": "单位A",
                    "地区": "地区1",
                    "predicted_severity": "重大",
                    "category_confidence": 0.92,
                    "severity_confidence": 0.88,
                    "need_review": False,
                    "kg_standard_id": "S1",
                    "kg_standard_name": "组合电器监督标准",
                    "kg_clause_id": "C1",
                    "kg_clause_title": "接地连接要求",
                    "kg_relation_confidence": 0.8,
                    "revision_demand": True,
                    "standard_revision_priority_score": 90,
                },
                {
                    "problem_id": "P2",
                    "问题描述": "接地连接方式与现场执行不一致",
                    "设备类型": "组合电器",
                    "监督专业": "电气",
                    "所属单位": "单位B",
                    "地区": "地区2",
                    "predicted_severity": "较大",
                    "category_confidence": 0.86,
                    "severity_confidence": 0.82,
                    "need_review": True,
                    "kg_standard_id": "S1",
                    "kg_standard_name": "组合电器监督标准",
                    "kg_clause_id": "C1",
                    "kg_clause_title": "接地连接要求",
                    "kg_relation_confidence": 0.6,
                    "revision_demand": "是",
                    "standard_revision_priority_score": 80,
                },
                {
                    "problem_id": "P3",
                    "问题描述": "GIS 接地母线连接记录缺失",
                    "设备类型": "组合电器",
                    "监督专业": "电气",
                    "所属单位": "单位A",
                    "地区": "地区1",
                    "predicted_severity": "重大",
                    "category_confidence": 0.91,
                    "severity_confidence": 0.9,
                    "need_review": False,
                    "kg_standard_id": "S1",
                    "kg_standard_name": "组合电器监督标准",
                    "kg_clause_id": "C1",
                    "kg_clause_title": "接地连接要求",
                    "kg_relation_confidence": 0.9,
                    "revision_demand": False,
                    "standard_revision_priority_score": 75,
                },
                {
                    "problem_id": "P4",
                    "问题描述": "隔离开关铭牌信息不完整",
                    "设备类型": "隔离开关",
                    "监督专业": "电气",
                    "所属单位": "单位C",
                    "地区": "地区3",
                    "predicted_severity": "一般",
                    "category_confidence": 0.75,
                    "severity_confidence": 0.73,
                    "need_review": False,
                    "kg_standard_id": "S2",
                    "kg_standard_name": "隔离开关监督标准",
                    "kg_clause_id": "C2",
                    "kg_clause_title": "铭牌要求",
                    "kg_relation_confidence": 0.35,
                    "revision_demand": False,
                    "standard_revision_priority_score": 20,
                },
                {
                    "problem_id": "P5",
                    "问题描述": "标准未明确特殊工况适用要求",
                    "设备类型": "组合电器",
                    "监督专业": "电气",
                    "所属单位": "单位D",
                    "地区": "地区4",
                    "predicted_severity": "较大",
                    "category_confidence": 0.8,
                    "severity_confidence": 0.81,
                    "need_review": False,
                    "kg_standard_id": "S3",
                    "kg_standard_name": "通用监督标准",
                    "kg_clause_id": "",
                    "kg_clause_title": "",
                    "kg_relation_confidence": 0.7,
                    "revision_demand": "标准缺失",
                    "standard_revision_priority_score": 65,
                },
            ]
        )

    def test_aggregate_counts_and_confidence(self) -> None:
        result = aggregate_standard_revision_priority(self._sample_dataframe())
        row = result.dataframe[result.dataframe["kg_clause_id"] == "C1"].iloc[0]

        self.assertEqual(row["related_problem_count"], 3)
        self.assertEqual(row["high_severity_problem_count"], 2)
        self.assertEqual(row["revision_demand_count"], 2)
        self.assertAlmostEqual(row["avg_relation_confidence"], 0.7667, places=4)
        self.assertAlmostEqual(row["max_relation_confidence"], 0.9, places=4)
        self.assertGreaterEqual(row["aggregated_priority_score"], 0)
        self.assertLessEqual(row["aggregated_priority_score"], 100)

    def test_high_frequency_clause_ranks_above_low_frequency_clause(self) -> None:
        result = aggregate_standard_revision_priority(self._sample_dataframe())
        top = result.dataframe.iloc[0]
        self.assertEqual(top["kg_clause_id"], "C1")
        self.assertGreater(top["aggregated_priority_score"], result.dataframe.iloc[-1]["aggregated_priority_score"])

    def test_missing_clause_id_uses_standard_level_group(self) -> None:
        result = aggregate_standard_revision_priority(self._sample_dataframe())
        row = result.dataframe[result.dataframe["kg_standard_id"] == "S3"].iloc[0]
        self.assertTrue(row["clause_missing"])
        self.assertEqual(row["kg_clause_id"], "")

    def test_missing_optional_fields_do_not_crash(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "problem_id": "P1",
                    "问题描述": "问题描述",
                    "kg_standard_refs": "标准A",
                    "revision_need": True,
                }
            ]
        )
        result = aggregate_standard_revision_priority(dataframe)
        self.assertEqual(len(result.dataframe), 1)
        self.assertIn("kg_relation_confidence", result.missing_fields)
        self.assertEqual(result.dataframe.iloc[0]["related_problem_count"], 1)

    def test_markdown_report_and_json_output_can_be_generated(self) -> None:
        result = aggregate_standard_revision_priority(self._sample_dataframe())
        report = build_standard_revision_report(result)
        self.assertIn("Top 10 标准/条款修订优先级", report)
        self.assertIn("总关联问题数", report)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "summary.json"
            write_json_summary(result, output_path)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIn("summary", payload)
            self.assertIn("items", payload)
            self.assertGreater(len(payload["items"]), 0)


if __name__ == "__main__":
    unittest.main()
