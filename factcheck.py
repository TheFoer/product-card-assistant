"""Проверка ответа модели по входным данным. Это код, а не просьба к модели,
поэтому результат не зависит от того, послушалась ли она промпта."""

from __future__ import annotations

import re

from cardgen.analysis import RECOMMENDED_SPECS, normalize, recommended_title
from cardgen.schemas import MISSING_MARK, ExtractedSpec, Issue, LlmCard, ProductInput, SpecItem

TITLE_MAX = 70
DESCRIPTION_MIN = 800
DESCRIPTION_MAX = 1200
QUERIES_MIN = 3
QUERIES_MAX = 5

_NUMBER = re.compile(r"\d+(?:[.,]\d+)*(?:/\d+)?")
_LATIN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-/+][A-Za-z0-9]+)*")
_ABBREVIATION = re.compile(r"\b[А-ЯЁ]{2,}\b")
_ALLOWED_ABBREVIATIONS = {"КИП"}

# Частые выдуманные свойства приборов, которые нельзя поймать по числам.
_RISKY_CLAIMS: dict[str, str] = {
    r"взрыво": "взрывозащита",
    r"нержаве": "нержавеющая сталь",
    r"вибро": "виброустойчивость",
    r"герметич": "герметичность",
    r"пылевлаг|влагозащ": "пылевлагозащита",
    r"реестр": "внесение в реестр",
    r"поверк|поверен|межповероч": "поверка",
    r"сертифи": "сертификация",
    r"гаранти": "гарантия",
    r"морозо": "морозостойкость",
    r"коррози": "коррозионная стойкость",
    r"агрессивн": "агрессивные среды",
    r"глицерин": "заполнение корпуса",
    r"латун|бронз|алюмин": "материал",
    r"надежн|долговечн": "надёжность",
    r"высок\w*\s+точн|высокоточн": "высокая точность",
    r"нефт|газов\w*\s+промышл|атомн|энергетик|химическ\w*\s+(?:промышл|производ)|пищев": "отрасль",
    r"в наличии|на складе|скидк|доставк": "коммерческие условия",
}


def _number_key(token: str) -> str:
    token = token.replace(",", ".")
    if "." in token and "/" not in token:
        token = token.rstrip("0").rstrip(".")
    return token


def numbers_in(text: str) -> set[str]:
    return {_number_key(match) for match in _NUMBER.findall(text)}


def unsupported_facts(text: str, source: str) -> list[str]:
    """Возвращает факты из текста, которых нет в источнике."""
    source_norm = normalize(source)
    source_lower = source.lower()
    found: list[str] = []
    found += sorted(numbers_in(text) - numbers_in(source))
    found += sorted(
        {token for token in _LATIN.findall(text) if token.lower() not in source_lower}
    )
    found += sorted(
        {
            token
            for token in _ABBREVIATION.findall(text)
            if token not in _ALLOWED_ABBREVIATIONS and token.lower() not in source_lower
        }
    )
    text_norm = normalize(text)
    for pattern, label in _RISKY_CLAIMS.items():
        if re.search(pattern, text_norm) and not re.search(pattern, source_norm):
            found.append(label)
    return found


def check_card(card: LlmCard, product: ProductInput) -> list[Issue]:
    """Проверяет объём, состав и фактическую точность. Замечания уходят модели на исправление."""
    issues: list[Issue] = []

    def error(code: str, field: str, message: str) -> None:
        issues.append(Issue(severity="error", code=code, field=field, message=message))

    title_len = len(card.seo_title)
    if not title_len:
        error("title_empty", "seo_title", "Заголовок пуст")
    elif title_len > TITLE_MAX:
        error("title_too_long", "seo_title",
              f"Заголовок {title_len} символов, нужно не более {TITLE_MAX}")

    text_len = len(card.seo_description)
    if not DESCRIPTION_MIN <= text_len <= DESCRIPTION_MAX:
        error("description_length", "seo_description",
              f"Описание {text_len} символов, нужно {DESCRIPTION_MIN}–{DESCRIPTION_MAX}")
    if MISSING_MARK in card.seo_description.lower():
        error("missing_mark_in_text", "seo_description",
              f"В SEO-тексте не должно быть фразы «{MISSING_MARK}»")

    unique_queries = {normalize(query) for query in card.search_queries}
    if not QUERIES_MIN <= len(card.search_queries) <= QUERIES_MAX:
        error("queries_count", "search_queries",
              f"Запросов {len(card.search_queries)}, нужно {QUERIES_MIN}–{QUERIES_MAX}")
    if len(unique_queries) != len(card.search_queries):
        error("queries_duplicate", "search_queries", "Запросы повторяются")

    if not card.target_audience:
        error("audience_empty", "target_audience", "Не указано, кому подходит прибор")
    if not card.tasks:
        error("tasks_empty", "tasks", "Не указано, какие задачи решает прибор")

    source = product.source_text()
    checked = {
        "seo_title": card.seo_title,
        "seo_description": card.seo_description,
        "search_queries": "\n".join(card.search_queries),
        "target_audience": "\n".join(card.target_audience),
        "tasks": "\n".join(card.tasks),
    }
    for field, text in checked.items():
        unsupported = unsupported_facts(text, source)
        if unsupported:
            error("unsupported_fact", field,
                  "Нет во входных данных: " + ", ".join(f"«{item}»" for item in unsupported))
    return issues


def _extraction_is_supported(spec: ExtractedSpec, description: str) -> bool:
    quote = normalize(spec.quote)
    if not spec.name or not spec.value or not quote or quote not in normalize(description):
        return False
    value_numbers = numbers_in(spec.value)
    if value_numbers:
        return value_numbers <= numbers_in(spec.quote)
    # Без чисел сравниваем по основам слов: «нержавеющая сталь» и «из нержавеющей стали».
    return all(word[: max(3, len(word) - 2)] in quote for word in normalize(spec.value).split())


def _capitalize(text: str) -> str:
    return text[:1].upper() + text[1:]


def build_specs(
    product: ProductInput, extracted: list[ExtractedSpec]
) -> tuple[list[SpecItem], list[str], list[Issue]]:
    """Собирает таблицу характеристик. Значения берутся из входа кодом, а не из текста модели."""
    issues: list[Issue] = []
    rows: list[SpecItem] = []

    if product.manufacturer:
        rows.append(SpecItem(name="Производитель", value=product.manufacturer, source="input"))
    else:
        rows.append(SpecItem(name="Производитель", value=MISSING_MARK, source="missing"))

    for name, value in product.specs.items():
        title = _capitalize(name)
        if value:
            rows.append(SpecItem(name=title, value=value, source="input"))
        else:
            rows.append(SpecItem(name=title, value=MISSING_MARK, source="missing"))

    def find(name: str) -> SpecItem | None:
        wanted = normalize(name)
        wanted_title = recommended_title(name)
        for row in rows:
            if normalize(row.name) == wanted:
                return row
            if wanted_title and recommended_title(row.name) == wanted_title:
                return row
        return None

    for spec in extracted:
        if not _extraction_is_supported(spec, product.description):
            issues.append(Issue(
                severity="warning", code="extraction_rejected", field="key_specs",
                message=f"Значение «{spec.name}: {spec.value}» не подтверждено цитатой и не добавлено",
            ))
            continue
        row = find(spec.name)
        if row is None:
            rows.append(SpecItem(name=_capitalize(spec.name), value=spec.value, source="description", quote=spec.quote))
        elif row.source == "missing":
            row.value, row.source, row.quote = spec.value, "description", spec.quote
        elif numbers_in(row.value) != numbers_in(spec.value) or (
            not numbers_in(row.value) and normalize(row.value) != normalize(spec.value)
        ):
            issues.append(Issue(
                severity="warning", code="data_conflict", field="key_specs",
                message=f"Расхождение «{row.name}»: «{row.value}» в характеристиках и "
                        f"«{spec.value}» в описании. В таблице оставлено первое значение",
            ))

    for title in RECOMMENDED_SPECS:
        if find(title) is None:
            rows.append(SpecItem(name=title, value=MISSING_MARK, source="missing"))

    missing = [row.name for row in rows if row.source == "missing"]
    return rows, missing, issues
