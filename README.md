# 技术监督问题分类与标准修订决策 Baseline（dw2.0）

`dw2.0` 是集成后的模型侧主工程，已经把原 `dw1.1.2` 的问题等级识别、原 `dw1.2.5` 的问题类别识别，以及课题二知识图谱关联能力合并到同一项目目录中。

当前支持：

- 问题等级分类
- 问题类别分类
- 批量预测
- 决策增强
- 人工复核
- 反馈回流
- 图谱导出
- 课题二知识图谱关联
- 标准修订需求识别
- 单条问题级标准修订优先级排序
- 标准/条款级修订优先级聚合

## 环境要求

- Python 3.10+
- 建议安装 `requirements.txt` 中的依赖
- 首次下载 Hugging Face 模型时，如需国内镜像，可通过环境变量设置：

```bash
set HF_ENDPOINT=https://hf-mirror.com
```

## 目录说明

```text
dw2.0/
├─ configs/
├─ data/
│  ├─ mock/
│  ├─ generated/
│  └─ real/
│     ├─ raw/
│     │  └─ pytest_processed_1.xlsx
│     └─ processed/
├─ src/
├─ docs/
├─ outputs/
├─ tests/
├─ artifacts/
├─ README.md
└─ requirements.txt
```

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 真实数据准备

原始 Excel 默认放在 `data/real/raw/pytest_processed_1.xlsx`。

生成“问题等级”训练数据：

```bash
python -m src.data.prepare_real_data
```

输出文件：

- `data/real/processed/pytest_processed_1_problem_level_v1.csv`
- `data/real/processed/pytest_processed_1_problem_level_v1_stats.json`

## 训练

```bash
python -m src.train --config configs/real_problem_level_v1.yaml
```

训练完成后，主要产物在 `artifacts/outputs_real_problem_level_v1/`：

- `best_model/`
- `train_metrics.json`
- `valid_metrics.json`
- `test_metrics.json`
- `valid_classification_report.json`
- `test_classification_report.json`
- `valid_confusion_matrix.csv`
- `test_confusion_matrix.csv`
- `label_mapping.json`
- `resolved_config.yaml`

## 评估

```bash
python -m src.evaluate --config configs/real_problem_level_v1.yaml --model-path artifacts/outputs_real_problem_level_v1/best_model --split test
```

## 预测

单条预测：

```bash
python -m src.predict --config configs/real_problem_level_v1.yaml --model-path artifacts/outputs_real_problem_level_v1/best_model --text "主变压器预防性试验项目未按标准要求全部完成"
```

批量预测：

```bash
python -m src.predict --config configs/real_problem_level_v1.yaml --model-path artifacts/outputs_real_problem_level_v1/best_model --input-file data/real/processed/pytest_processed_1_problem_level_v1.csv --output-file artifacts/predictions_real_problem_level_v1/predictions.csv
```

## 人工复核表

批量预测 CSV 更适合程序读取。如果要交给人工复核，可以导出 Excel 版复核表：

```bash
python -m src.export_review_sheet --input-file artifacts/predictions_real_problem_level_v1/predictions_for_review.csv --output-file artifacts/predictions_real_problem_level_v1/predictions_for_review_human.xlsx
```

Excel 中包含：

- `summary`：复核统计
- `need_review`：只包含低置信度、建议人工复核的样本
- `all_predictions`：全部预测结果

如果输入文件已经经过知识图谱关联增强，复核表还会包含：

- 关联监督要点
- 关联标准依据
- 关联监督要求
- 修订需求类型
- 修订优先级
- 修订需求判定原因

## 决策增强

在已有预测结果上补充问题类别、优先级和优化建议：

```bash
python -m src.enrich_predictions --config configs/real_problem_level_v1.yaml --input-file artifacts/predictions_real_problem_level_v1/predictions.csv --output-file outputs/predictions_enriched.csv
```

该步骤会调用项目内置的问题类别模型配置和模型产物：

- `configs/real_problem_category_v1.yaml`
- `artifacts/outputs_real_problem_category_v1/best_model`

输出：

- `pred_level`
- `pred_category`
- `priority_score`
- `priority_label`
- `priority_reason`
- `suggestion`
- `recommended_action`

## 接入课题二知识图谱

仓库已经内置课题二离线图谱文件，克隆本仓库后无需额外准备 `trustworthy-tech-kg/` 目录即可运行知识图谱关联：

```text
data/kg/kg_graph.json
```

在决策增强结果上关联课题二监督要点、标准文件和监督要求，并生成单条问题级标准修订优先级：

```bash
python -m src.enrich_with_kg --config configs/real_problem_level_v1.yaml --input-file outputs/predictions_enriched.csv --kg-graph data/kg/kg_graph.json --output-file outputs/predictions_kg_linked.csv --standard-report outputs/standard_revision_priority.csv --markdown-report outputs/kg_revision_report.md --top-k 3 --min-score 0.18
```

主要输出：

- `outputs/predictions_kg_linked.csv`：问题级图谱关联与修订需求识别结果
- `outputs/standard_revision_priority.csv`：基于问题级关联结果生成的标准修订优先级排序
- `outputs/kg_revision_report.md`：验证报告，包含匹配率、未匹配原因、Top 标准和典型样本
- `outputs/predictions_kg_linked_human.xlsx`：人工复核 Excel，可由 `src.export_review_sheet` 导出

新增问题级字段包括：

- `kg_match_status`
- `kg_match_score`
- `kg_point_id`
- `kg_point_text`
- `kg_project`
- `kg_stage`
- `kg_severity`
- `kg_standard_refs`
- `kg_requirement_texts`
- `revision_need`
- `revision_need_type`
- `revision_priority_score`
- `revision_priority_label`
- `revision_reason`

当前课题二图谱主要覆盖 `组合电器` 和 `隔离开关`。其他设备会标记为 `kg_match_status=unsupported_equipment`，不会强行低置信度匹配。

详细说明见：

- `docs/kg_revision_integration.md`

## 标准库样例与一键闭环 Demo

当前仓库保留正式课题二图谱入口：

- `data/kg/kg_graph.json`

同时提供一个最小可运行标准库样例：

- `data/kg/sample_kg_graph.json`
- `data/examples/sample_predictions.jsonl`

正式标准库建议放在 `data/kg/` 下。标准库 JSON 至少包含：

- `standards[]`：`standard_id`、`standard_name`、`standard_no`、`standard_status`、`domain`、`equipment_type`、`risk_level`
- `clauses[]`：`clause_id`、`standard_id`、`clause_no`、`clause_text`、`keywords`、`equipment_type`、`problem_category`

一键运行小闭环：

```bash
python scripts/run_kg_revision_smoke_test.py
```

该脚本执行：

```text
问题数据
-> 分类预测结果样例
-> 标准条款关联
-> 标准修订触发判断
-> 标准/条款级修订优先级聚合
-> 图谱节点/关系 CSV 导出
```

输出文件：

- `outputs/review_sheet_with_kg.jsonl`
- `outputs/review_sheet_with_kg.csv`
- `outputs/standard_revision_priority.csv`
- `outputs/graph_nodes.csv`
- `outputs/graph_edges.csv`

也可以单独运行标准条款增强：

```bash
python src/enrich_with_kg.py --input data/examples/sample_predictions.jsonl --kg data/kg/sample_kg_graph.json --output outputs/review_sheet_with_kg.jsonl --min-score 0.12
```

输出质量检查：

```bash
python src/validate_kg_outputs.py --input outputs/review_sheet_with_kg.jsonl
```

## 标准/条款级修订优先级聚合

完整流程如下：

```text
问题记录
-> 分类预测
-> 置信度与人工复核
-> 问题—标准/条款关联
-> 单条问题级标准修订需求识别
-> 标准/条款级聚合排序
-> 输出标准修订优先级汇总表和图谱记录
```

三个优先级概念需要区分：

- `priority`：单条问题处理优先级，面向现场整改和问题处置。
- `standard_revision_priority`：单条问题指向的标准修订优先级，说明该问题是否提示标准需要修订。
- `aggregated_standard_revision_priority`：多条问题按同一标准/条款聚合后的标准修订优先级，用于回答“哪些标准/条款应优先修订”。

聚合命令：

```bash
python -m src.aggregate_standard_revision_priority --input outputs/predictions_kg_linked.csv --output outputs/standard_revision_priority_summary.csv --json-output outputs/standard_revision_priority_summary.json --report-output outputs/standard_revision_priority_report.md --top-k 50 --min-problem-count 1
```

未传入 `--input` 时，默认读取 `outputs/predictions_kg_linked.csv`。

主要输出：

- `outputs/standard_revision_priority_summary.csv`：标准/条款级修订优先级排行榜
- `outputs/standard_revision_priority_summary.json`：结构化聚合结果
- `outputs/standard_revision_priority_report.md`：Markdown 汇总报告，包含 Top 10、缺失字段说明和复核提示

## 反馈回流

```bash
python -m src.prepare_feedback_retrain --config configs/real_problem_level_v1.yaml --feedback-file artifacts/predictions_real_problem_level_v1/predictions.csv --output-file artifacts/feedback_real_problem_level_v1/retrain_dataset.csv
```

## 图谱导出

```bash
python -m src.export_graph_records --config configs/real_problem_level_v1.yaml --input-file data/real/processed/pytest_processed_1_problem_level_v1.csv
```

如果输入文件为 `outputs/predictions_kg_linked.csv`，导出的 JSONL 会同时包含课题二知识图谱关联字段和标准修订需求字段：

```bash
python -m src.export_graph_records --config configs/real_problem_level_v1.yaml --input-file outputs/predictions_kg_linked.csv --output-file outputs/graph_records_kg_linked_from_predictions.jsonl
```

图谱导出保留一行一个问题的默认结构，同时新增 `graph_nodes` 和 `graph_edges` 字段，显式包含：

- 节点：`problem`、`standard`、`clause`、`category`、`severity`、`device`、`major`
- 边：`related_to_clause`、`belongs_to_standard`、`related_to_standard`、`has_category`、`has_severity`、`has_device`、`has_major`

一键 demo 额外导出 Neo4j 友好的 CSV：

- `outputs/graph_nodes.csv`：`node_id`、`node_type`、`name`、`standard_no`、`status`、`clause_no`、`text`
- `outputs/graph_edges.csv`：`edge_id`、`source_id`、`target_id`、`relation_type`、`confidence`、`relation_reason`

节点类型包括 `Problem`、`Standard`、`Clause`、`Equipment`、`ProblemCategory`、`RevisionNeed`。关系类型包括 `PROBLEM_MATCHES_CLAUSE`、`CLAUSE_BELONGS_TO_STANDARD`、`PROBLEM_HAS_CATEGORY`、`PROBLEM_INVOLVES_EQUIPMENT`、`PROBLEM_TRIGGERS_REVISION_NEED`、`STANDARD_HAS_REVISION_PRIORITY`。

端到端示例：

```bash
python -m src.enrich_with_kg --config configs/real_problem_level_v1.yaml --input-file outputs/predictions_enriched.csv --kg-graph data/kg/kg_graph.json --output-file outputs/predictions_kg_linked.csv --standard-report outputs/standard_revision_priority.csv --markdown-report outputs/kg_revision_report.md --top-k 3 --min-score 0.18
python -m src.aggregate_standard_revision_priority --input outputs/predictions_kg_linked.csv --output outputs/standard_revision_priority_summary.csv --json-output outputs/standard_revision_priority_summary.json --report-output outputs/standard_revision_priority_report.md
python -m src.export_graph_records --config configs/real_problem_level_v1.yaml --input-file outputs/predictions_kg_linked.csv --output-file outputs/graph_records_kg_linked_from_predictions.jsonl
```

## 测试

知识图谱关联与标准修订优先级相关单元测试：

```bash
pytest
python -m unittest tests.test_kg_revision -v
python -m unittest tests.test_standard_revision_aggregation -v
python -m unittest discover tests
```

本轮已验证：

- `kg_graph.json` 解析
- 已知监督要点匹配
- 非课题二覆盖设备识别
- 修订需求规则
- 标准/条款级修订优先级聚合

## 当前限制与后续方向

当前条款匹配主要是规则、关键词和轻量文本相似度，不是完整知识图谱推理，也不是深度语义标准理解模型。真实效果依赖标准库质量、字段完整性和人工复核闭环。

后续可扩展：

- 接入正式标准库
- 接入课题二图谱服务
- 使用向量检索增强条款匹配
- 使用人工复核结果优化匹配规则
- 增加标准生命周期数据
- 增加多跳图谱推理

## 可移植性约定

- 所有配置路径默认使用项目内相对路径
- 原始真实数据统一放在 `data/real/raw/`
- 训练产物统一放在 `artifacts/`
- 不要在源码里写死本地磁盘路径
