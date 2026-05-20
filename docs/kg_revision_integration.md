# 课题二知识图谱接入与标准修订优先级说明

## 1. 目标

本模块用于把模型侧预测结果与课题二知识图谱连接起来，形成可交付的决策闭环：

```text
问题预测结果
-> 关联课题二监督要点、标准文件、监督要求
-> 识别疑似标准修订需求
-> 聚合生成标准修订优先级排序
-> 导出复核表、图谱 JSONL 和验证报告
```

第一版只消费课题二离线图谱文件，不依赖 Neo4j 或 FastAPI 服务。

## 2. 输入文件

模型侧输入：

- `outputs/predictions_enriched.csv`
- 或任意包含问题文本、设备类型、监督要点、优先级等字段的 CSV/JSONL/XLSX

课题二输入：

- `data/kg/kg_graph.json`

仓库已内置该离线图谱文件，因此克隆本仓库后不需要额外准备 `trustworthy-tech-kg/` 目录。

当前图谱主要覆盖：

- 组合电器
- 隔离开关

非覆盖设备会输出：

```text
kg_match_status=unsupported_equipment
```

## 3. 运行命令

```powershell
.\.venv_local\Scripts\python.exe -m src.enrich_with_kg `
  --config configs/real_problem_level_v1.yaml `
  --input-file outputs/predictions_enriched.csv `
  --kg-graph data\kg\kg_graph.json `
  --output-file outputs/predictions_kg_linked.csv `
  --standard-report outputs/standard_revision_priority.csv `
  --markdown-report outputs/kg_revision_report.md `
  --top-k 3 `
  --min-score 0.18
```

## 4. 匹配逻辑

模型侧每条问题优先使用以下字段进行匹配：

- `checkpoint_text`
- `text`
- `supervision_opinion`
- `actual_fix`
- `rule_name`
- `device_type`
- `stage_name / supervision_stage / problem_stage`

课题二图谱侧使用以下节点和关系：

- `监督要点`
- `监督项目`
- `标准文件`
- `监督要求`
- `问题分级`
- `设备类型`
- `技术阶段`
- `HAS_POINT`
- `GOVERNED_BY`
- `HAS_REQUIREMENT`
- `HAS_SEVERITY`

匹配分数综合考虑监督要点文本、问题文本、整改文本、标准依据、阶段和问题分级。若监督要点文本直接命中课题二监督要点或监督要求，会给出较高分。

## 5. 新增字段

问题级图谱关联字段：

- `kg_match_status`
- `kg_match_score`
- `kg_point_id`
- `kg_point_text`
- `kg_project`
- `kg_stage`
- `kg_severity`
- `kg_standard_refs`
- `kg_requirement_texts`
- `kg_top_matches`

修订需求字段：

- `revision_need`
- `revision_need_type`
- `revision_priority_score`
- `revision_priority_label`
- `revision_reason`

标准聚合报告字段：

- `standard_ref`
- `linked_issue_count`
- `revision_need_count`
- `major_or_above_issue_count`
- `max_issue_priority`
- `avg_issue_priority`
- `avg_kg_match_score`
- `standard_revision_priority_score`
- `standard_revision_priority_label`
- `example_issue_ids`

## 6. 修订需求类型

当前规则固定输出以下类型：

- `执行落实问题`
- `标准缺失`
- `标准表述歧义`
- `标准冲突`
- `适用性不足`
- `需人工判断`

其中 `标准缺失`、`标准表述歧义`、`标准冲突`、`适用性不足` 会被标记为疑似标准修订需求。

## 7. 局限

- 当前是规则匹配和规则识别版本，适合形成演示和报告闭环。
- 课题二图谱覆盖范围有限，未覆盖设备不做低置信度硬匹配。
- 修订需求结论仍需专家复核，不能直接替代正式标准制修订决策。
