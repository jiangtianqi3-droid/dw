from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.utils.kg_revision import (
    build_standard_priority_report,
    classify_revision_need,
    enrich_dataframe_with_kg,
    load_kg_index,
    match_record_to_kg,
)


def _sample_graph() -> dict:
    return {
        "nodes": [
            {"id": "EQUIPMENT_TYPE_组合电器", "node_type": "设备类型", "label": "组合电器"},
            {"id": "TECHNICAL_STAGE_工程设计", "node_type": "技术阶段", "label": "工程设计"},
            {"id": "组合电器_SUPERVISION_PROJECT_接地检查", "node_type": "监督项目", "label": "接地检查"},
            {
                "id": "point_1",
                "node_type": "监督要点",
                "label": "接地线与GIS接地母线应采用螺栓连接方式。",
                "full_text": "接地线与GIS接地母线应采用螺栓连接方式。",
                "equipment": "组合电器",
                "reference": "《电气装置安装工程接地装置施工及验收规范》（GB 50169-2016）4.3.10。",
                "stage": "工程设计",
            },
            {"id": "SEVERITY_LEVEL_一般", "node_type": "问题分级", "label": "一般"},
            {
                "id": "standard_1",
                "node_type": "标准文件",
                "label": "电气装置安装工程接地装置施工及验收规范",
                "full_reference": "《电气装置安装工程接地装置施工及验收规范》（GB 50169-2016）",
            },
            {
                "id": "requirement_1",
                "node_type": "监督要求",
                "label": "接地线与GIS接地母线应采用螺栓连接方式。",
                "full_text": "接地线与GIS接地母线应采用螺栓连接方式。",
            },
        ],
        "links": [
            {"source": "组合电器_SUPERVISION_PROJECT_接地检查", "target": "TECHNICAL_STAGE_工程设计", "edge_type": "BELONGS_TO_STAGE"},
            {"source": "组合电器_SUPERVISION_PROJECT_接地检查", "target": "point_1", "edge_type": "HAS_POINT"},
            {"source": "point_1", "target": "SEVERITY_LEVEL_一般", "edge_type": "HAS_SEVERITY"},
            {"source": "point_1", "target": "standard_1", "edge_type": "GOVERNED_BY"},
            {"source": "point_1", "target": "requirement_1", "edge_type": "HAS_REQUIREMENT"},
        ],
    }


def _sample_standard_library() -> dict:
    return {
        "standards": [
            {
                "standard_id": "STD-1",
                "standard_name": "组合电器接地标准",
                "standard_no": "Q/SAMPLE-1",
                "standard_status": "待修订",
                "domain": "电气设备",
                "equipment_type": "组合电器",
                "risk_level": "较大",
            }
        ],
        "clauses": [
            {
                "clause_id": "CLAUSE-1",
                "standard_id": "STD-1",
                "clause_no": "4.3.10",
                "clause_text": "GIS接地线应与接地母线采用螺栓连接方式。",
                "keywords": ["GIS", "接地线", "接地母线", "螺栓连接"],
                "equipment_type": "组合电器",
                "problem_category": "施工质量",
            }
        ],
    }


def _load_sample_index():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "kg_graph.json"
        path.write_text(json.dumps(_sample_graph(), ensure_ascii=False), encoding="utf-8")
        return load_kg_index(path)


def _load_sample_standard_index():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "sample_kg_graph.json"
        path.write_text(json.dumps(_sample_standard_library(), ensure_ascii=False), encoding="utf-8")
        return load_kg_index(path)


class KGRevisionTest(unittest.TestCase):
    def test_load_kg_index_extracts_points_and_standards(self) -> None:
        index = _load_sample_index()
        self.assertEqual(index.supported_equipment, {"组合电器"})
        self.assertEqual(len(index.points), 1)
        self.assertIn("GB 50169", " ".join(index.points[0].standards))
        self.assertEqual(index.points[0].severity, "一般")

    def test_known_checkpoint_matches_kg_point(self) -> None:
        index = _load_sample_index()
        record = {
            "sample_id": "case_1",
            "device_type": "组合电器",
            "stage_name": "工程设计",
            "rule_level": "一般",
            "checkpoint_text": "3.6.2 接地线与 GIS接地母线应采用螺栓连接方式。",
            "text": "GIS接地线无法与汇流接地母线用螺栓紧固。",
        }
        status, matches = match_record_to_kg(record, index, top_k=3, min_score=0.18)
        self.assertEqual(status, "matched")
        self.assertEqual(matches[0].point.id, "point_1")
        self.assertGreater(matches[0].score, 0.8)

    def test_unsupported_equipment_is_not_forced(self) -> None:
        index = _load_sample_index()
        status, matches = match_record_to_kg({"device_type": "运行杆塔", "text": "螺栓错位"}, index)
        self.assertEqual(status, "unsupported_equipment")
        self.assertEqual(matches, [])

    def test_revision_need_rules_detect_ambiguity(self) -> None:
        result = classify_revision_need(
            {"text": "该标准表述不明确，现场理解存在偏差。"},
            match_status="low_score",
            best_match=None,
        )
        self.assertEqual(result["revision_need_type"], "需人工判断")

        index = _load_sample_index()
        status, matches = match_record_to_kg(
            {
                "device_type": "组合电器",
                "stage_name": "工程设计",
                "checkpoint_text": "接地线与 GIS接地母线应采用螺栓连接方式。",
                "text": "该标准表述不明确，现场理解存在偏差。",
            },
            index,
        )
        result = classify_revision_need({"text": "该标准表述不明确，现场理解存在偏差。"}, status, matches[0])
        self.assertTrue(result["revision_need"])
        self.assertEqual(result["revision_need_type"], "标准表述歧义")

    def test_standard_priority_report_is_sorted_and_complete(self) -> None:
        index = _load_sample_index()
        df = pd.DataFrame(
            [
                {
                    "sample_id": "case_1",
                    "device_type": "组合电器",
                    "stage_name": "工程设计",
                    "rule_level": "一般",
                    "checkpoint_text": "接地线与 GIS接地母线应采用螺栓连接方式。",
                    "text": "该标准表述不明确，现场理解存在偏差。",
                    "priority_score": 0.7,
                }
            ]
        )
        enriched = enrich_dataframe_with_kg(df, index)
        report = build_standard_priority_report(enriched)
        self.assertEqual(list(report.columns)[0], "standard_ref")
        self.assertEqual(int(report.iloc[0]["linked_issue_count"]), 1)
        self.assertIn(report.iloc[0]["standard_revision_priority_label"], {"高", "中", "低"})

    def test_standard_library_matches_correct_clause_and_outputs_fields(self) -> None:
        index = _load_sample_standard_index()
        dataframe = pd.DataFrame(
            [
                {
                    "problem_id": "P1",
                    "问题描述": "GIS接地线未按要求与接地母线采用螺栓连接。",
                    "设备类型": "组合电器",
                    "predicted_category": "施工质量",
                    "predicted_severity": "较大",
                }
            ]
        )
        enriched = enrich_dataframe_with_kg(dataframe, index, min_score=0.1)
        row = enriched.iloc[0]
        self.assertEqual(row["related_clause_id"], "CLAUSE-1")
        self.assertEqual(row["related_standard_id"], "STD-1")
        self.assertGreater(row["standard_match_confidence"], 0)
        self.assertIn(row["problem_standard_relation_type"], {"standard_execution", "standard_lagging"})
        self.assertIn(row["revision_need_type"], {"执行落实问题", "适用性不足"})
        self.assertIn(row["standard_revision_priority_initial"], {"high", "medium", "low"})
        self.assertEqual(row["graph_relation_type"], "PROBLEM_MATCHES_CLAUSE")

    def test_empty_standard_library_and_missing_problem_fields_do_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "empty.json"
            path.write_text(json.dumps({"standards": [], "clauses": []}, ensure_ascii=False), encoding="utf-8")
            index = load_kg_index(path)
        enriched = enrich_dataframe_with_kg(pd.DataFrame([{"problem_id": "P-empty"}]), index)
        row = enriched.iloc[0]
        self.assertEqual(row["kg_match_status"], "no_kg_data")
        self.assertEqual(row["related_standard_name"], "")
        self.assertEqual(row["problem_standard_relation_type"], "unmatched")


if __name__ == "__main__":
    unittest.main()
