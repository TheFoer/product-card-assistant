"""Текстовое представление результата для копирования в CMS."""

from __future__ import annotations

from cardgen.schemas import MISSING_MARK, CardResult


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def to_markdown(result: CardResult) -> str:
    card = result.card
    lines = [
        f"# {result.product_name}",
        "",
        f"**Статус:** {'готово к проверке' if result.status == 'ready' else 'требует правки: см. замечания'}",
        "",
        f"## SEO-заголовок ({len(card.seo_title)} симв.)",
        "",
        card.seo_title,
        "",
        f"## SEO-описание ({len(card.seo_description)} симв.)",
        "",
        card.seo_description,
        "",
        "## Ключевые характеристики",
        "",
        "| Характеристика | Значение |",
        "| --- | --- |",
    ]
    for spec in card.key_specs:
        value = f"*{MISSING_MARK}*" if spec.source == "missing" else _cell(spec.value)
        lines.append(f"| {_cell(spec.name)} | {value} |")

    lines += ["", "## Поисковые запросы", ""]
    lines += [f"- {query}" for query in card.search_queries]
    lines += ["", "## Для менеджера", "", "**Кому подходит:**", ""]
    lines += [f"- {item}" for item in card.commercial.target_audience]
    lines += ["", "**Какие задачи решает:**", ""]
    lines += [f"- {item}" for item in card.commercial.tasks]

    if result.missing_fields or result.issues:
        lines += ["", "## Нужно уточнить", ""]
        if result.missing_fields:
            lines.append("- Нет данных: " + ", ".join(result.missing_fields))
        for issue in result.issues:
            marker = "Ошибка" if issue.severity == "error" else "Внимание"
            lines.append(f"- {marker}: {issue.message}")
    return "\n".join(lines) + "\n"
