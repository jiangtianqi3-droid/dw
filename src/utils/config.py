from __future__ import annotations

import os
from pathlib import Path

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    expanded = os.path.expanduser(os.path.expandvars(path_str))
    path = Path(expanded)
    if path.is_absolute():
        return path
    return get_project_root() / path


def load_config(config_path: str) -> dict:
    config_file = resolve_path(config_path)
    if config_file is None or not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_file.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["config_path"] = str(config_file)
    return config


def save_yaml(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, allow_unicode=True, sort_keys=False)
