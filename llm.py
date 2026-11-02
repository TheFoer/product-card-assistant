"""Клиент Anthropic Messages API: таймауты, повторы и понятные ошибки."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from cardgen.config import Settings
from cardgen.errors import (
    ConfigError,
    LlmAuthError,
    LlmError,
    LlmOutputError,
    LlmRateLimitError,
    LlmRefusalError,
    LlmRequestError,
    LlmTimeoutError,
    LlmUnavailableError,
)
from cardgen.schemas import LLM_JSON_SCHEMA

log = logging.getLogger(__name__)

API_VERSION = "2023-06-01"
RETRYABLE_STATUSES = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_RETRY_DELAY_SECONDS = 30.0


@dataclass
class LlmResponse:
    # None, если текст ответа не удалось разобрать как JSON.
    data: dict[str, Any] | None
    # Блоки ответа без изменений: при исправлении они возвращаются в диалог.
    raw_content: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    request_id: str = ""


class LlmClient(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict[str, Any]]) -> LlmResponse: ...


class AnthropicClient:
    def __init__(
        self,
        settings: Settings,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not settings.api_key:
            raise ConfigError(
                "Не задан ANTHROPIC_API_KEY. Скопируйте .env.example в .env и укажите ключ"
            )
        self.model = settings.model
        self._settings = settings
        self._session = session or requests.Session()
        self._sleep = sleep

    def complete(self, system: str, messages: list[dict[str, Any]]) -> LlmResponse:
        body = {
            "model": self.model,
            "max_tokens": self._settings.max_tokens,
            "system": system,
            "messages": messages,
            "output_config": {"format": {"type": "json_schema", "schema": LLM_JSON_SCHEMA}},
        }
        attempts = self._settings.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return self._parse(self._send(body))
            except _Retryable as exc:
                if attempt == attempts:
                    raise exc.final from None
                delay = exc.retry_after if exc.retry_after is not None else self._backoff(attempt)
                log.warning("LLM: %s, повтор %d/%d через %.1f с",
                            exc.reason, attempt, attempts - 1, delay)
                self._sleep(min(delay, MAX_RETRY_DELAY_SECONDS))
        raise LlmError("Сервис LLM не ответил")  # недостижимо, нужно для проверки типов

    @staticmethod
    def _backoff(attempt: int) -> float:
        return 2.0 ** (attempt - 1) + random.uniform(0, 0.5)

    def _send(self, body: dict[str, Any]) -> requests.Response:
        try:
            response = self._session.post(
                f"{self._settings.base_url}/v1/messages",
                headers={
                    "x-api-key": self._settings.api_key,
                    "anthropic-version": API_VERSION,
                    "content-type": "application/json",
                },
                json=body,
                timeout=(10, self._settings.timeout_seconds),
            )
        except requests.Timeout:
            raise _Retryable("таймаут", LlmTimeoutError("Сервис LLM не ответил вовремя")) from None
        except requests.RequestException as exc:
            raise _Retryable(
                f"сетевая ошибка {type(exc).__name__}",
                LlmUnavailableError("Не удалось подключиться к сервису LLM"),
            ) from None

        if response.status_code == 200:
            return response
        self._raise_for_status(response)
        raise AssertionError("unreachable")

    @staticmethod
    def _raise_for_status(response: requests.Response) -> None:
        status = response.status_code
        request_id = response.headers.get("request-id", "")
        try:
            error = response.json().get("error", {})
            detail = str(error.get("message", ""))[:300]
        except (ValueError, AttributeError):
            detail = ""
        log.error("LLM HTTP %s, request_id=%s: %s", status, request_id, detail)
        suffix = f" (request_id: {request_id})" if request_id else ""

        if status in (401, 403):
            raise LlmAuthError("Сервис LLM отклонил ключ API: проверьте ANTHROPIC_API_KEY" + suffix)
        if status in RETRYABLE_STATUSES:
            final: LlmError
            if status == 429:
                final = LlmRateLimitError("Превышен лимит запросов к LLM, повторите позже" + suffix)
            elif status in (408, 504):
                final = LlmTimeoutError("Сервис LLM не успел обработать запрос" + suffix)
            else:
                final = LlmUnavailableError("Сервис LLM временно недоступен" + suffix)
            raise _Retryable(f"HTTP {status}", final, _retry_after(response))
        raise LlmRequestError(f"Сервис LLM отклонил запрос (HTTP {status}): {detail}{suffix}")

    @staticmethod
    def _parse(response: requests.Response) -> LlmResponse:
        request_id = response.headers.get("request-id", "")
        try:
            payload = response.json()
            content = payload["content"]
            if not isinstance(content, list):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            raise LlmOutputError("Сервис LLM вернул ответ неожиданного формата") from None

        stop_reason = payload.get("stop_reason")
        if stop_reason == "refusal":
            raise LlmRefusalError("Модель отказалась обрабатывать эти данные")
        if stop_reason == "max_tokens":
            raise LlmOutputError("Ответ модели обрезан: увеличьте LLM_MAX_TOKENS")

        # Перед текстом могут идти блоки thinking, поэтому ищем текст по типу, а не по индексу.
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
        data: dict[str, Any] | None
        try:
            parsed = json.loads(text)
            data = parsed if isinstance(parsed, dict) else None
        except ValueError:
            data = None

        usage = payload.get("usage") or {}
        return LlmResponse(
            data=data,
            raw_content=content,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            request_id=request_id,
        )


class _Retryable(Exception):
    def __init__(self, reason: str, final: LlmError, retry_after: float | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.final = final
        self.retry_after = retry_after


def _retry_after(response: requests.Response) -> float | None:
    try:
        return max(0.0, float(response.headers["retry-after"]))
    except (KeyError, ValueError):
        return None
