"""Разбор и проверка входных данных до обращения к LLM."""

from __future__ import annotations

import json
import re
from typing import Any

from cardgen.errors import InvalidJsonError, InvalidProductError
from cardgen.schemas import Issue, Preflight, ProductInput

MAX_NAME = 300
MAX_DESCRIPTION = 20_000
MAX_SPECS = 100
MAX_SPEC_TEXT = 500
SHORT_DESCRIPTION = 100

# Значения, которые означают отсутствие данных, а не сами данные.
PLACEHOLDERS = {
    "", "-", "—", "–", "?", "н/д", "нд", "нет данных", "данных нет", "нет", "уточняется",
    "n/a", "na", "none", "null", "tbd", "unknown",
}

# Характеристики, перечисленные в задании как типовые для прибора: канонический
# заголовок → варианты написания. Для простых приборов часть полей может быть
# неприменима, поэтому отсутствие поля — повод уточнить, а не ошибка.
RECOMMENDED_SPECS: dict[str, tuple[str, ...]] = {
    "Диапазон измерений": (
        "диапазон измерений", "диапазон измерения", "диапазон",
        "пределы измерений", "предел измерений",
    ),
    "Класс точности": ("класс точности", "точность", "погрешность", "основная погрешность"),
    "Питание": ("питание", "напряжение питания", "источник питания"),
    "Тип корпуса": ("тип корпуса", "корпус", "исполнение корпуса"),
    "Интерфейсы": ("интерфейсы", "интерфейс", "цифровой интерфейс", "протокол связи"),
}


def normalize(text: str) -> str:
    """Приводит текст к виду для сравнения: регистр, ё/е, пробелы, крайняя пунктуация."""
    text = text.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", text).strip(" \t\n.,:;")


def clean_value(value: str) -> str:
    """Убирает лишние пробелы и превращает заглушки вроде «н/д» в пустую строку."""
    value = value.strip()
    return "" if normalize(value) in PLACEHOLDERS else value


def recommended_title(spec_name: str) -> str | None:
    wanted = normalize(spec_name)
    for title, aliases in RECOMMENDED_SPECS.items():
        if wanted in aliases:
            return title
    return None


def parse_json(text: str | bytes) -> Any:
    if isinstance(text, bytes):
        try:
            text = text.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise InvalidJsonError("Данные должны быть в кодировке UTF-8") from None
    text = text.lstrip("\ufeff")
    if not text.strip():
        raise InvalidJsonError("Передан пустой JSON")

    def reject_constant(name: str) -> None:
        raise ValueError(name)

    try:
        return json.loads(text, parse_constant=reject_constant)
    except json.JSONDecodeError as exc:
        raise InvalidJsonError(
            f"Некорректный JSON: строка {exc.lineno}, позиция {exc.colno} ({exc.msg})"
        ) from None
    except ValueError as exc:
        raise InvalidJsonError(f"Некорректный JSON: недопустимое значение {exc}") from None


def _text_field(raw: dict[str, Any], key: str, limit: int, errors: list[dict[str, str]]) -> str:
    value = raw.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        errors.append({"field": key, "message": "Ожидается строка"})
        return ""
    if len(value) > limit:
        errors.append({"field": key, "message": f"Длина превышает {limit} символов"})
        return ""
    return clean_value(value)


def _spec_value(value: Any) -> str | None:
    """Возвращает строку или None, если тип значения не подходит."""
    if value is None:
        return ""
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    return clean_value(str(value))


def parse_product(raw: Any) -> ProductInput:
    """Проверяет структуру входа. Если писать карточку не из чего, LLM не вызывается."""
    if not isinstance(raw, dict):
        raise InvalidProductError(
            "Ожидается JSON-объект с данными одного товара",
            [{"field": "$", "message": f"Получен тип {type(raw).__name__}"}],
        )

    errors: list[dict[str, str]] = []
    name = _text_field(raw, "name", MAX_NAME, errors)
    manufacturer = _text_field(raw, "manufacturer", MAX_NAME, errors)
    description = _text_field(raw, "description", MAX_DESCRIPTION, errors)

    specs: dict[str, str] = {}
    raw_specs = raw.get("specs")
    if raw_specs is not None and not isinstance(raw_specs, dict):
        errors.append({"field": "specs", "message": "Ожидается объект «название: значение»"})
    elif raw_specs:
        if len(raw_specs) > MAX_SPECS:
            errors.append({"field": "specs", "message": f"Больше {MAX_SPECS} характеристик"})
        else:
            for key, value in raw_specs.items():
                spec_name = key.strip()
                cleaned = _spec_value(value)
                field = f"specs.{spec_name or '(пустое имя)'}"
                if not spec_name:
                    errors.append({"field": field, "message": "Пустое название характеристики"})
                elif cleaned is None:
                    errors.append({"field": field, "message": "Значение должно быть строкой или числом"})
                elif len(cleaned) > MAX_SPEC_TEXT or len(spec_name) > MAX_SPEC_TEXT:
                    errors.append({"field": field, "message": f"Длина превышает {MAX_SPEC_TEXT} символов"})
                else:
                    specs[spec_name] = cleaned

    if not errors:
        if not name:
            errors.append({"field": "name", "message": "Обязательное поле: название товара"})
        if not description and not any(specs.values()):
            errors.append({
                "field": "specs, description",
                "message": "Нужны характеристики или описание: писать карточку не из чего",
            })
    if errors:
        raise InvalidProductError("Данные о товаре нужно исправить", errors)

    known = {"name", "manufacturer", "specs", "description"}
    return ProductInput(
        name=name,
        manufacturer=manufacturer,
        specs=specs,
        description=description,
        ignored_fields=tuple(sorted(str(key) for key in raw if key not in known)),
    )


def missing_recommended(present_names: list[str]) -> list[str]:
    present = {recommended_title(name) for name in present_names}
    return [title for title in RECOMMENDED_SPECS if title not in present]


def preflight(product: ProductInput) -> Preflight:
    """Сообщает, чего не хватает. Работает без LLM и без ключа API."""
    issues: list[Issue] = []

    def warn(code: str, field: str, message: str) -> None:
        issues.append(Issue(severity="warning", code=code, field=field, message=message))

    missing: list[str] = []
    if not product.manufacturer:
        missing.append("Производитель")
        warn("missing_manufacturer", "manufacturer", "Не указан производитель")
    if not product.description:
        warn("missing_description", "description", "Нет описания: текст строится только на характеристиках")
    elif product.description.endswith(("...", "…")):
        warn("truncated_description", "description",
             "Описание обрывается многоточием: проверьте, что передан полный текст")
    elif len(product.description) < SHORT_DESCRIPTION:
        warn("short_description", "description", "Описание очень короткое: текст может получиться общим")

    for spec_name, value in product.specs.items():
        if not value:
            missing.append(spec_name)
            warn("empty_spec", f"specs.{spec_name}", f"Характеристика «{spec_name}» указана без значения")

    for title in missing_recommended(list(product.filled_specs)):
        if title not in missing and recommended_title(title) not in {
            recommended_title(name) for name in missing
        }:
            missing.append(title)

    if product.ignored_fields:
        warn("ignored_fields", "$", "Эти поля не используются: " + ", ".join(product.ignored_fields))

    return Preflight(product_name=product.name, missing_fields=missing, issues=issues)
