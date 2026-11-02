"""Сценарий: проверка входа → генерация → проверка фактов → исправление → результат."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import ValidationError

from cardgen.analysis import preflight
from cardgen.errors import LlmOutputError
from cardgen.factcheck import build_specs, check_card
from cardgen.llm import LlmClient
from cardgen.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_repair_prompt, build_user_prompt
from cardgen.render import to_markdown
from cardgen.schemas import CardDraft, CardResult, Commercial, Issue, LlmCard, Meta, ProductInput

log = logging.getLogger(__name__)


class CardGenerator:
    def __init__(self, llm: LlmClient, max_repairs: int = 2) -> None:
        self._llm = llm
        self._max_repairs = max_repairs

    def generate(self, product: ProductInput) -> CardResult:
        check = preflight(product)
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": build_user_prompt(product, check.missing_fields)}
        ]
        meta = Meta(model=self._llm.model, prompt_version=PROMPT_VERSION, attempts=0)
        best: tuple[LlmCard, list[Issue]] | None = None

        for _ in range(self._max_repairs + 1):
            response = self._llm.complete(SYSTEM_PROMPT, messages)
            meta.attempts += 1
            meta.input_tokens += response.input_tokens
            meta.output_tokens += response.output_tokens
            if response.request_id:
                meta.request_ids.append(response.request_id)

            card, problems = self._evaluate(response.data, product)
            if card is not None and (best is None or len(problems) < len(best[1])):
                best = (card, problems)
            if card is not None and not problems:
                break
            messages += [
                {"role": "assistant", "content": response.raw_text},
                {"role": "user", "content": build_repair_prompt(problems)},
            ]

        if best is None:
            raise LlmOutputError("Модель не вернула карточку в нужном формате")

        log.info("Карточка собрана: попыток=%d, замечаний=%d", meta.attempts, len(best[1]))
        return self._assemble(product, best[0], best[1], check.issues, meta)

    @staticmethod
    def _evaluate(data: dict[str, Any] | None, product: ProductInput) -> tuple[LlmCard | None, list[Issue]]:
        malformed = Issue(
            severity="error", code="bad_format", field="$",
            message="Ответ не соответствует схеме: верни JSON со всеми полями",
        )
        if data is None:
            return None, [malformed]
        try:
            card = LlmCard.model_validate(data)
        except ValidationError:
            return None, [malformed]
        return card, check_card(card, product)

    @staticmethod
    def _assemble(
        product: ProductInput,
        card: LlmCard,
        problems: list[Issue],
        input_issues: list[Issue],
        meta: Meta,
    ) -> CardResult:
        specs, missing, spec_issues = build_specs(product, card.extracted_specs)
        gaps = [
            Issue(severity="warning", code="model_gap", field="data_gaps", message=gap)
            for gap in card.data_gaps
        ]
        result = CardResult(
            status="needs_review" if problems else "ready",
            product_name=product.name,
            card=CardDraft(
                seo_title=card.seo_title,
                seo_description=card.seo_description,
                key_specs=specs,
                search_queries=card.search_queries,
                commercial=Commercial(target_audience=card.target_audience, tasks=card.tasks),
            ),
            missing_fields=missing,
            issues=[*problems, *spec_issues, *input_issues, *gaps],
            meta=meta,
        )
        result.markdown = to_markdown(result)
        return result
