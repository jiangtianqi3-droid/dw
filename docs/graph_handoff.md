# 图谱对接说明

## 1. 对接目标

当前反馈模型侧负责输出“问题实例的结构化记录”，图谱侧负责将这些记录映射为图谱节点和关系。

建议的对接原则：

- 图谱侧优先消费 `final_confirmed_label`
- `predicted_label` 和 `confidence` 作为辅助信息保留，不作为最终事实
- `label_source` 用来区分标签来源：`dataset` / `model` / `manual`
- 人工复核后的样本优先进入图谱

## 2. 当前可直接交付的导出文件

统一导出命令：

```bash
python -m src.export_graph_records --config <config> --input-file <input-file>
```

示例：

```bash
python -m src.export_graph_records --config configs/real_problem_level_v1.yaml --input-file data/real/processed/pytest_processed_1_problem_level_v1.csv
```

默认导出为 JSONL，一行一条问题记录。

## 3. 导出字段说明

| 字段名 | 含义 | 图谱侧建议用途 |
|---|---|---|
| `issue_id` | 问题唯一标识 | 作为问题节点主键 |
| `issue_text` | 问题原文 | 问题节点属性 |
| `issue_category` | 问题类别标签 | 连接“问题类别”节点 |
| `issue_level` | 问题等级标签 | 连接“问题等级”节点 |
| `device_type` | 设备类型 | 连接“设备类型”节点 |
| `specialty` | 监督专业 | 连接“监督专业”节点 |
| `supervision_stage` | 问题产生环节/监督环节 | 连接“监督环节”节点 |
| `problem_stage` | 问题产生阶段 | 连接“问题阶段”节点 |
| `event_time` | 发现时间 | 问题节点属性 |
| `source_unit` | 来源单位 | 连接“来源单位”节点或保留属性 |
| `data_source` | 数据来源 | 数据治理属性 |
| `is_reviewed` | 是否已复核 | 过滤条件 |
| `review_status` | 复核状态 | 反馈状态属性 |
| `predicted_label` | 模型原始预测标签 | 模型侧辅助属性 |
| `final_confirmed_label` | 最终确认标签 | 图谱事实标签优先来源 |
| `label_source` | 标签来源 | 判断可信度与优先级 |
| `confidence` | 模型置信度 | 辅助分析属性 |
| `model_version` | 模型版本 | 模型治理属性 |
| `prediction_time` | 预测时间 | 时效属性 |
| `problem_reason` | 问题原因 | 可作为关联分析属性 |
| `rule_name` | 细则名称 | 后续关联标准条款的重要入口 |

## 4. 当前任务与字段使用方式

### 当前真实任务：问题等级

当前第一版真实数据训练的是“问题等级”三分类：

- `一般`
- `较大`
- `重大`

因此当前图谱导出中：

- `issue_level` 有值
- `issue_category` 为空

### 后续目标任务：问题类别

后续如果切到“问题类别”任务，则应变为：

- `issue_category` 有值
- `issue_level` 可以来自另一个任务头或另一个模型

也就是说，图谱侧最好同时预留：

- “问题类别”节点
- “问题等级”节点

## 5. 建议的图谱节点

建议最少包含以下节点类型：

- `问题`
- `问题类别`
- `问题等级`
- `设备类型`
- `监督专业`
- `监督环节`
- `问题阶段`
- `标准细则`
- `来源单位`

## 6. 建议的关系

建议最少包含以下关系：

- `问题 -> 属于类别 -> 问题类别`
- `问题 -> 对应等级 -> 问题等级`
- `问题 -> 涉及设备 -> 设备类型`
- `问题 -> 涉及专业 -> 监督专业`
- `问题 -> 发生于环节 -> 监督环节`
- `问题 -> 产生于阶段 -> 问题阶段`
- `问题 -> 关联细则 -> 标准细则`
- `问题 -> 来源于 -> 来源单位`

## 7. 标签优先级规则

图谱侧建议按以下优先级写入标签：

1. `manual`
2. `dataset`
3. `model`

对应字段判断：

- 若 `label_source=manual`，优先采用 `final_confirmed_label`
- 若 `label_source=dataset`，说明是已有标注数据
- 若 `label_source=model`，说明还未人工确认，建议进入待复核区

## 8. 推荐对接流程

1. 模型侧输出预测结果
2. 人工复核并更新 `review_status / reviewed_label / final_confirmed_label`
3. 使用 `src.export_graph_records` 导出统一 JSONL
4. 图谱侧读取 JSONL 并写入节点/关系
5. 图谱侧如有补充标准条款关系，可回传给模型侧用于后续特征增强

## 9. 当前建议

短期内建议图谱同学先接这几个字段：

- `issue_id`
- `issue_text`
- `issue_level`
- `device_type`
- `specialty`
- `supervision_stage`
- `problem_stage`
- `rule_name`
- `final_confirmed_label`
- `label_source`

如果后续切到“问题类别”任务，只需要把 `issue_level` 的主消费切换到 `issue_category`，整体接口不需要重做。
