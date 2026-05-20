from __future__ import annotations

import json
import unittest

import pandas as pd

from src.utils.graph_export import build_graph_export_dataframe


class GraphExportTest(unittest.TestCase):
    def test_clause_node_and_clause_edges_are_exported(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "problem_id": "P1",
                    "问题描述": "GIS 接地线连接方式不符合条款要求",
                    "predicted_category": "设备安装",
                    "predicted_severity": "重大",
                    "设备类型": "组合电器",
                    "监督专业": "电气",
                    "kg_standard_id": "S1",
                    "kg_standard_name": "组合电器监督标准",
                    "kg_clause_id": "C1",
                    "kg_clause_title": "接地连接要求",
                    "kg_relation_confidence": 0.86,
                    "revision_reason": "关联条款明确要求接地线连接方式。",
                }
            ]
        )

        exported = build_graph_export_dataframe(dataframe, config={}, label_role="category")
        nodes = json.loads(exported.iloc[0]["graph_nodes"])
        edges = json.loads(exported.iloc[0]["graph_edges"])

        node_types = {node["type"] for node in nodes}
        edge_types = {edge["relation_type"] for edge in edges}

        self.assertIn("clause", node_types)
        self.assertIn("related_to_clause", edge_types)
        self.assertIn("belongs_to_standard", edge_types)
        self.assertIn("related_to_standard", edge_types)

    def test_missing_clause_does_not_export_empty_clause_node(self) -> None:
        dataframe = pd.DataFrame(
            [
                {
                    "problem_id": "P2",
                    "问题描述": "隔离开关铭牌信息不完整",
                    "predicted_category": "资料记录",
                    "predicted_severity": "一般",
                    "设备类型": "隔离开关",
                    "监督专业": "电气",
                    "kg_standard_id": "S2",
                    "kg_standard_name": "隔离开关监督标准",
                    "kg_clause_id": "",
                    "kg_clause_title": "",
                    "kg_relation_confidence": 0.72,
                }
            ]
        )

        exported = build_graph_export_dataframe(dataframe, config={}, label_role="category")
        nodes = json.loads(exported.iloc[0]["graph_nodes"])
        edges = json.loads(exported.iloc[0]["graph_edges"])

        node_types = {node["type"] for node in nodes}
        edge_types = {edge["relation_type"] for edge in edges}

        self.assertNotIn("clause", node_types)
        self.assertNotIn("related_to_clause", edge_types)
        self.assertNotIn("belongs_to_standard", edge_types)
        self.assertIn("related_to_standard", edge_types)


if __name__ == "__main__":
    unittest.main()
