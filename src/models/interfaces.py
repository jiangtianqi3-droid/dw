from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TaskDefinition:
    name: str
    task_type: str
    label_field: str
    labels: list[str] = field(default_factory=list)
    loss_weight: float = 1.0


def load_task_definitions(config: dict) -> list[TaskDefinition]:
    task_config = config["task"]
    explicit_tasks = task_config.get("tasks")

    if explicit_tasks:
        # 多任务模式下，把配置统一展开成标准任务定义列表。
        return [
            TaskDefinition(
                name=item["name"],
                task_type=item["task_type"],
                label_field=item["label_field"],
                labels=item.get("labels", []),
                loss_weight=float(item.get("loss_weight", 1.0)),
            )
            for item in explicit_tasks
        ]

    # 单任务模式也统一包装成列表，方便后续平滑切到多任务头。
    return [
        TaskDefinition(
            name=task_config.get("target_name", "problem_category"),
            task_type=task_config["type"],
            label_field=task_config["label_field"],
            labels=task_config.get("label_list", []),
        )
    ]
