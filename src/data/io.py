from __future__ import annotations

from pathlib import Path

import pandas as pd


def infer_file_type(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    raise ValueError(f"Unsupported file type for: {path}")


def read_table(path: str | Path, file_type: str | None = None) -> pd.DataFrame:
    resolved_type = (file_type or infer_file_type(path)).lower()
    path = Path(path)

    if resolved_type == "csv":
        return pd.read_csv(path)
    if resolved_type == "jsonl":
        return pd.read_json(path, lines=True)
    if resolved_type in {"excel", "xlsx", "xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {resolved_type}")


def write_table(dataframe: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_type = infer_file_type(path)

    if file_type == "csv":
        dataframe.to_csv(path, index=False)
        return
    if file_type == "jsonl":
        dataframe.to_json(path, orient="records", lines=True, force_ascii=False)
        return
    if file_type == "excel":
        dataframe.to_excel(path, index=False)
        return

    raise ValueError(f"Unsupported output type: {file_type}")
