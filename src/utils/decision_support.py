from __future__ import annotations

from typing import Any

import pandas as pd

from src.decision.priority import DEFAULT_PRIORITY_RULES, compute_priority_score
from src.decision.suggestions import DEFAULT_SUGGESTION_RULES, generate_suggestion
from src.utils.config import resolve_path
from src.utils.model_runtime import load_model_runtime, predict_dataframe


DECISION_OUTPUT_COLUMNS = [
    "pred_level",
    "pred_category",
    "level_confidence",
    "category_confidence",
    "need_review",
    "priority_score",
    "priority_label",
    "priority_reason",
    "suggestion",
    "suggestion_basis",
    "recommended_action",
]


def get_decision_config(config: dict) -> dict:
    return config.get("decision", {})


def _merge_nested_dict(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(defaults)
    if not overrides:
        return merged
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null"}:
        return default
    return text


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalize_text(value).lower() in {"true", "1", "yes", "y"}


def infer_primary_label_role(config: dict) -> str:
    task_config = config.get("task", {})
    target_name = _normalize_text(task_config.get("target_name"), default="").lower()
    label_field = _normalize_text(task_config.get("label_field"), default="").lower()
    if "level" in target_name or "level" in label_field:
        return "level"
    return "category"


def attach_prediction_role_columns(
    dataframe: pd.DataFrame,
    role: str,
    predictions: list[dict[str, Any]],
) -> pd.DataFrame:
    dataframe = dataframe.copy()
    label_column = f"pred_{role}"
    confidence_column = f"{role}_confidence"
    id_column = f"{role}_label_id"
    dataframe[id_column] = [item["predicted_label_id"] for item in predictions]
    dataframe[label_column] = [item["predicted_label"] for item in predictions]
    dataframe[confidence_column] = [item["confidence"] for item in predictions]
    return dataframe


def normalize_primary_prediction_columns(dataframe: pd.DataFrame, config: dict) -> pd.DataFrame:
    dataframe = dataframe.copy()
    role = infer_primary_label_role(config)
    label_column = f"pred_{role}"
    confidence_column = f"{role}_confidence"

    if label_column not in dataframe.columns:
        fallback_label = "predicted_label" if "predicted_label" in dataframe.columns else "label"
        if fallback_label in dataframe.columns:
            dataframe[label_column] = dataframe[fallback_label].fillna("").astype(str)
        else:
            dataframe[label_column] = ""

    if confidence_column not in dataframe.columns:
        if "confidence" in dataframe.columns:
            dataframe[confidence_column] = pd.to_numeric(dataframe["confidence"], errors="coerce")
        else:
            dataframe[confidence_column] = pd.NA

    if "pred_level" not in dataframe.columns:
        dataframe["pred_level"] = ""
    if "pred_category" not in dataframe.columns:
        dataframe["pred_category"] = ""
    if "level_confidence" not in dataframe.columns:
        dataframe["level_confidence"] = pd.NA
    if "category_confidence" not in dataframe.columns:
        dataframe["category_confidence"] = pd.NA

    if role == "level":
        dataframe["pred_level"] = dataframe[label_column].where(
            dataframe["pred_level"].fillna("").astype(str).str.strip().eq(""),
            dataframe["pred_level"],
        )
        dataframe["level_confidence"] = pd.to_numeric(
            dataframe["level_confidence"].fillna(dataframe[confidence_column]),
            errors="coerce",
        )
    else:
        dataframe["pred_category"] = dataframe[label_column].where(
            dataframe["pred_category"].fillna("").astype(str).str.strip().eq(""),
            dataframe["pred_category"],
        )
        dataframe["category_confidence"] = pd.to_numeric(
            dataframe["category_confidence"].fillna(dataframe[confidence_column]),
            errors="coerce",
        )

    return dataframe


def get_category_model_settings(config: dict) -> tuple[str | None, str | None]:
    decision_config = get_decision_config(config)
    model_config = decision_config.get("category_model", {})
    return model_config.get("config_path"), model_config.get("model_path")


def maybe_attach_category_predictions(
    dataframe: pd.DataFrame,
    config: dict,
    category_config_path: str | None = None,
    category_model_path: str | None = None,
    device=None,
) -> pd.DataFrame:
    dataframe = normalize_primary_prediction_columns(dataframe, config)
    if dataframe["pred_category"].fillna("").astype(str).str.strip().ne("").all():
        return dataframe

    category_config_path = category_config_path or get_category_model_settings(config)[0]
    category_model_path = category_model_path or get_category_model_settings(config)[1]
    if not category_config_path or not category_model_path:
        return dataframe

    runtime = load_model_runtime(
        config_or_path=category_config_path,
        model_path=category_model_path,
        device=device,
    )
    prepared_df, predictions, _ = predict_dataframe(dataframe, runtime)
    enriched = attach_prediction_role_columns(prepared_df, role="category", predictions=predictions)

    dataframe = dataframe.copy()
    dataframe["pred_category"] = enriched["pred_category"]
    dataframe["category_confidence"] = pd.to_numeric(enriched["category_confidence"], errors="coerce")
    dataframe["category_label_id"] = enriched["category_label_id"]
    if "model_input" not in dataframe.columns and "model_input" in enriched.columns:
        dataframe["model_input"] = enriched["model_input"]
    return dataframe


def apply_decision_outputs(dataframe: pd.DataFrame, config: dict) -> pd.DataFrame:
    dataframe = normalize_primary_prediction_columns(dataframe, config)
    decision_config = get_decision_config(config)
    review_threshold = float(
        decision_config.get(
            "review_threshold",
            config.get("feedback", {}).get("prediction_export", {}).get("review_confidence_threshold", 0.75),
        )
    )

    level_conf = pd.to_numeric(dataframe["level_confidence"], errors="coerce")
    category_conf = pd.to_numeric(dataframe["category_confidence"], errors="coerce")
    existing_need_review = (
        dataframe["need_manual_review"].map(_normalize_bool)
        if "need_manual_review" in dataframe.columns
        else pd.Series(False, index=dataframe.index)
    )
    dataframe["need_review"] = existing_need_review | level_conf.fillna(1.0).lt(review_threshold) | category_conf.fillna(1.0).lt(review_threshold)

    priority_rules = _merge_nested_dict(DEFAULT_PRIORITY_RULES, decision_config.get("priority_rules"))
    suggestion_rules = _merge_nested_dict(DEFAULT_SUGGESTION_RULES, decision_config.get("suggestion_rules"))

    priority_rows = dataframe.to_dict(orient="records")
    priority_df = pd.DataFrame([compute_priority_score(row, rules=priority_rules) for row in priority_rows])
    overwrite_columns = set(priority_df.columns) | {"need_review", "suggestion", "suggestion_basis", "recommended_action"}
    base_df = dataframe.drop(columns=[column for column in overwrite_columns if column in dataframe.columns], errors="ignore")
    merged = pd.concat([base_df.reset_index(drop=True), priority_df], axis=1)
    merged["need_review"] = (
        existing_need_review | level_conf.fillna(1.0).lt(review_threshold) | category_conf.fillna(1.0).lt(review_threshold)
    )
    suggestion_df = pd.DataFrame([generate_suggestion(row, rules=suggestion_rules) for row in merged.to_dict(orient="records")])
    merged = pd.concat([merged, suggestion_df], axis=1)
    return merged


def build_enriched_predictions(
    dataframe: pd.DataFrame,
    config: dict,
    category_config_path: str | None = None,
    category_model_path: str | None = None,
    device=None,
) -> pd.DataFrame:
    enriched = maybe_attach_category_predictions(
        dataframe=dataframe,
        config=config,
        category_config_path=category_config_path,
        category_model_path=category_model_path,
        device=device,
    )
    return apply_decision_outputs(enriched, config=config)


def resolve_enriched_output_path(config: dict, output_file: str | None) -> str:
    if output_file:
        return str(resolve_path(output_file))
    decision_config = get_decision_config(config)
    output_dir = resolve_path(decision_config.get("output_dir", "outputs"))
    return str(output_dir / "predictions_enriched.csv")
