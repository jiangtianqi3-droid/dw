# 课题二知识图谱接入与标准修订优先级说明

## 1. 目标

本模块用于把模型侧预测结果与课题二知识图谱连接起来，形成可交付的决策闭环：

```text
问题预测结果
-> 关联课题二监督要点、标准文件、监督要求
-> 识别疑似标准修订需求
-> 单条问题级标准修订优先级识别
-> 标准/条款级聚合生成标准修订优先级排序
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

标准/条款级聚合字段：

- `rank`
- `kg_standard_id`
- `kg_standard_name`
- `kg_clause_id`
- `kg_clause_title`
- `clause_missing`
- `related_problem_count`
- `high_severity_problem_count`
- `medium_high_severity_problem_count`
- `revision_demand_count`
- `manual_review_count`
- `low_confidence_problem_count`
- `avg_relation_confidence`
- `max_relation_confidence`
- `avg_standard_revision_priority_score`
- `device_type_count`
- `major_count`
- `organization_count`
- `region_count`
- `aggregated_priority_score`
- `aggregated_priority_level`
- `aggregated_priority_reason`
- `representative_problem_ids`
- `representative_problem_texts`

## 6. 修订需求类型

当前规则固定输出以下类型：

- `执行落实问题`
- `标准缺失`
- `标准表述歧义`
- `标准冲突`
- `适用性不足`
- `需人工判断`

其中 `标准缺失`、`标准表述歧义`、`标准冲突`、`适用性不足` 会被标记为疑似标准修订需求。

## 7. 标准/条款级修订优先级聚合

标准/条款级聚合用于把多条问题按同一标准和同一条款汇总，形成面向标准制修订工作的排序清单。若输入中缺少 `kg_clause_id`，系统按标准级聚合，并在输出中标记：

```text
clause_missing=true
```

聚合维度优先使用：

- `kg_standard_id`
- `kg_standard_name`
- `kg_clause_id`
- `kg_clause_title`

如果标准 ID 或名称缺失，但存在 `kg_standard_refs`，会从标准引用文本派生稳定 ID，避免因字段缺失导致流程中断。

聚合分数采用规则化加权，不引入机器学习模型：

```text
aggregated_priority_score =
  0.25 * frequency_score
+ 0.25 * severity_score
+ 0.20 * revision_demand_score
+ 0.15 * relation_confidence_score
+ 0.10 * coverage_score
+ 0.05 * existing_record_priority_score
- 0.10 * uncertainty_penalty
```

各子分数限制在 0 到 1 之间，最终分数限制在 0 到 100 之间。等级划分如下：

其中 `frequency_score = min(count / max(3, min(total_count, 10)), 1.0)`，用于让小样本演示数据中的多问题聚集也能被合理体现，同时保持规则简单可解释。

- `score >= 80`：高优先级
- `60 <= score < 80`：中高优先级
- `40 <= score < 60`：中优先级
- `score < 40`：低优先级

运行命令：

```powershell
.\.venv_local\Scripts\python.exe -m src.aggregate_standard_revision_priority `
  --input outputs/predictions_kg_linked.csv `
  --output outputs/standard_revision_priority_summary.csv `
  --json-output outputs/standard_revision_priority_summary.json `
  --report-output outputs/standard_revision_priority_report.md `
  --top-k 50 `
  --min-problem-count 1
```

未显式传入 `--input` 时，CLI 默认读取 `outputs/predictions_kg_linked.csv`。

报告会输出总关联问题数、参与聚合的标准数、参与聚合的条款数、高优先级数量、中高优先级数量、Top 10 标准/条款、缺失字段说明，以及低置信度或需人工复核说明。

## 8. 图谱导出

`src.export_graph_records` 保留默认的一行一个问题 JSONL 导出，同时新增 `graph_nodes` 和 `graph_edges` 字段，用于显式交付图谱节点和关系。

节点类型包括：

- `problem`
- `standard`
- `clause`
- `category`
- `severity`
- `device`
- `major`

关系类型包括：

- `problem -> clause`：`related_to_clause`
- `clause -> standard`：`belongs_to_standard`
- `problem -> standard`：`related_to_standard`
- `problem -> category`：`has_category`
- `problem -> severity`：`has_severity`
- `problem -> device`：`has_device`
- `problem -> major`：`has_major`

边上会尽量保留：

- `confidence`
- `source_field`
- `relation_reason`

如果 `kg_clause_id` 为空，不生成空条款节点，但仍保留 `problem -> standard` 关系。

## 9. 概念区分

- `priority`：问题处理优先级，面向单条问题的现场整改和管理闭环。
- `standard_revision_priority`：单条问题指向的标准修订优先级，说明该问题是否提示某个标准需要修订。
- `aggregated_standard_revision_priority`：标准/条款级聚合后的标准修订优先级，用于说明哪些标准或条款应优先进入修订需求池。

## 10. 局限

- 当前是规则匹配和规则识别版本，适合形成演示和报告闭环。
- 课题二图谱覆盖范围有限，未覆盖设备不做低置信度硬匹配。
- 修订需求结论仍需专家复核，不能直接替代正式标准制修订决策。
