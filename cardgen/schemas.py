"""Модели данных: вход, ответ LLM и итоговый результат."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MISSING_MARK = "данных нет"


@dataclass(frozen=True)
class ProductInput:
    name: str
    manufacturer: str = ""
    # Значение "" означает, что характеристика указана, но не заполнена.
    specs: dict[str, str] = field(default_factory=dict)
    description: str = ""
    ignored_fields: tuple[str, ...] = ()

    @property
    def filled_specs(self) -> dict[str, str]:
        return {key: value for key, value in self.specs.items() if value}

    def source_text(self) -> str:
        """Весь текст, из которого разрешено брать факты."""
        parts = [self.name, self.manufacturer, self.description]
        for key, value in self.specs.items():
            parts += [key, value]
        return "\n".join(parts)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class ExtractedSpec(_Strict):
    name: str
    value: str
    quote: str


class LlmCard(_Strict):
    """Ответ модели. Длины здесь намеренно не проверяются: это делает factcheck,
    чтобы вернуть модели понятный список замечаний."""

    seo_title: str
    seo_description: str
    extracted_specs: list[ExtractedSpec]
    search_queries: list[str]
    target_audience: list[str]
    tasks: list[str]
    data_gaps: list[str]

    @field_validator("search_queries", "target_audience", "tasks", "data_gaps")
    @classmethod
    def _drop_blank(cls, items: list[str]) -> list[str]:
        return [item.strip() for item in items if item.strip()]


def _string_list(description: str) -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": description}


# Схема для строгого режима Structured Outputs: все поля обязательны,
# лишние запрещены. Длины и количество элементов проверяет factcheck.
LLM_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "seo_title",
        "seo_description",
        "extracted_specs",
        "search_queries",
        "target_audience",
        "tasks",
        "data_gaps",
    ],
    "properties": {
        "seo_title": {"type": "string", "description": "SEO-заголовок, не более 70 символов"},
        "seo_description": {"type": "string", "description": "SEO-описание, 800–1200 символов"},
        "extracted_specs": {
            "type": "array",
            "description": "Характеристики из description, которых нет в specs",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "value", "quote"],
                "properties": {
                    "name": {"type": "string"},
                    "value": {"type": "string"},
                    "quote": {"type": "string", "description": "Дословная цитата из description"},
                },
            },
        },
        "search_queries": _string_list("3–5 поисковых запросов"),
        "target_audience": _string_list("Кому подходит прибор"),
        "tasks": _string_list("Какие задачи решает прибор"),
        "data_gaps": _string_list("Что контент-менеджеру нужно уточнить"),
    },
}


class SpecItem(BaseModel):
    name: str
    value: str
    source: Literal["input", "description", "missing"]
    quote: str | None = None


class Commercial(BaseModel):
    target_audience: list[str]
    tasks: list[str]


class CardDraft(BaseModel):
    seo_title: str
    seo_description: str
    key_specs: list[SpecItem]
    search_queries: list[str]
    commercial: Commercial


class Issue(BaseModel):
    severity: Literal["error", "warning"]
    code: str
    field: str
    message: str


class Meta(BaseModel):
    model: str
    prompt_version: str
    attempts: int
    input_tokens: int = 0
    output_tokens: int = 0
    request_ids: list[str] = Field(default_factory=list)


class CardResult(BaseModel):
    # ready — автоматические проверки пройдены; needs_review — есть замечания уровня error.
    status: Literal["ready", "needs_review"]
    product_name: str
    card: CardDraft
    missing_fields: list[str]
    issues: list[Issue]
    meta: Meta
    markdown: str = ""


class Preflight(BaseModel):
    product_name: str
    missing_fields: list[str]
    issues: list[Issue]
