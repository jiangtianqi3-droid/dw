# 大文件与模型产物说明

本仓库提交的是 `dw2.0` 的代码、配置、文档和测试。

以下内容未提交到 GitHub：

- 虚拟环境目录：`.venv_local/`、`.conda_gpu312/` 等
- 训练产物目录：`artifacts/`
- 运行输出目录：`outputs/`
- 大模型权重：`*.safetensors`、`*.pt`、`*.bin` 等

原因：这些文件体积较大，普通 GitHub 仓库不适合直接保存。

仓库已经包含课题二离线图谱：

- `data/kg/kg_graph.json`

因此克隆仓库后，不需要额外准备 `trustworthy-tech-kg/` 目录即可运行知识图谱关联相关代码。

如需直接使用已经训练好的等级/类别模型进行预测和决策增强，请从本地 `dw2.0` 或后续约定的模型文件交付位置复制模型产物到：

- `artifacts/outputs_real_problem_level_v1/best_model`
- `artifacts/outputs_real_problem_category_v1/best_model`

代码中的路径已按 `dw2.0` 项目内相对路径配置。

如果没有这些模型产物，也可以先安装依赖并按 README 中的训练命令重新训练，再运行预测和决策增强。
