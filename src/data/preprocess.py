from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


PUNCTUATION_MAP = {
    "，": ",",
    "。": ".",
    "；": ";",
    "：": ":",
    "（": "(",
    "）": ")",
    "【": "[",
    "】": "]",
}


@dataclass
class TextPreprocessor:
    # 这里只做轻量归一化，避免把数据清洗逻辑和业务规则耦合得太深。
    strip: bool = True
    lowercase: bool = False
    normalize_punctuation: bool = True
    terminology_map: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: dict) -> "TextPreprocessor":
        return cls(
            strip=config.get("strip", True),
            lowercase=config.get("lowercase", False),
            normalize_punctuation=config.get("normalize_punctuation", True),
            terminology_map=config.get("terminology_map", {}) or {},
        )

    def normalize_text(self, text: str) -> str:
        normalized = "" if text is None else str(text)
        if self.strip:
            normalized = normalized.strip()
        if self.lowercase:
            normalized = normalized.lower()
        if self.normalize_punctuation:
            normalized = "".join(PUNCTUATION_MAP.get(char, char) for char in normalized)
        for source, target in self.terminology_map.items():
            normalized = normalized.replace(source, target)
        return normalized

    def preprocess_record(self, record: dict, text_field: str) -> dict:
        updated = dict(record)
        updated[text_field] = self.normalize_text(updated.get(text_field, ""))
        return updated


@dataclass
class InputBuilder:
    # 把结构化字段和问题文本拼成统一输入，后续方便扩展到更多业务字段。
    enabled: bool = False
    output_field: str = "model_input"
    structured_fields: list[str] = field(default_factory=list)
    field_aliases: dict[str, str] = field(default_factory=dict)
    key_value_sep: str = ": "
    field_sep: str = "；"
    include_text_field_name: bool = True
    skip_empty_fields: bool = True

    @classmethod
    def from_config(cls, config: dict | None) -> "InputBuilder":
        config = config or {}
        return cls(
            enabled=bool(config.get("enabled", False)),
            output_field=str(config.get("output_field", "model_input")),
            structured_fields=list(config.get("structured_fields", []) or []),
            field_aliases=dict(config.get("field_aliases", {}) or {}),
            key_value_sep=str(config.get("key_value_sep", ": ")),
            field_sep=str(config.get("field_sep", "；")),
            include_text_field_name=bool(config.get("include_text_field_name", True)),
            skip_empty_fields=bool(config.get("skip_empty_fields", True)),
        )

    def get_output_field(self, text_field: str) -> str:
        return self.output_field if self.enabled else text_field

    def build_text(self, record: dict, text_field: str, preprocessor: TextPreprocessor) -> str:
        base_text = preprocessor.normalize_text(record.get(text_field, ""))
        if not self.enabled:
            return base_text

        parts: list[str] = []
        for field_name in self.structured_fields:
            field_value = preprocessor.normalize_text(record.get(field_name, ""))
            if self.skip_empty_fields and not field_value:
                continue
            alias = self.field_aliases.get(field_name, field_name)
            parts.append(f"{alias}{self.key_value_sep}{field_value}")

        # 文本字段通常放在最后，便于保留“结构化上下文 + 原问题描述”的顺序。
        if self.include_text_field_name:
            text_alias = self.field_aliases.get(text_field, text_field)
            parts.append(f"{text_alias}{self.key_value_sep}{base_text}")
        else:
            parts.append(base_text)

        return self.field_sep.join(part for part in parts if part)

    def transform_dataframe(
        self,
        dataframe: pd.DataFrame,
        text_field: str,
        preprocessor: TextPreprocessor,
    ) -> pd.DataFrame:
        transformed = dataframe.copy()
        relevant_fields = [text_field, *self.structured_fields]

        for field_name in relevant_fields:
            if field_name in transformed.columns:
                transformed[field_name] = transformed[field_name].fillna("").map(preprocessor.normalize_text)

        output_field = self.get_output_field(text_field)
        transformed[output_field] = transformed.to_dict(orient="records")
        transformed[output_field] = transformed[output_field].map(
            lambda record: self.build_text(record, text_field=text_field, preprocessor=preprocessor)
        )
        return transformed


def get_model_input_field(config: dict, text_field: str) -> str:
    return InputBuilder.from_config(config.get("input_builder", {})).get_output_field(text_field)
