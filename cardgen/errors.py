"""Ошибки сервиса. Каждая несёт машинный код, HTTP-статус и понятное пользователю сообщение."""

from __future__ import annotations


class CardGenError(Exception):
    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, details: list[dict[str, str]] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or []


class InvalidJsonError(CardGenError):
    code = "invalid_json"
    http_status = 400


class InvalidProductError(CardGenError):
    code = "invalid_product"
    http_status = 422


class ConfigError(CardGenError):
    code = "not_configured"
    http_status = 503


class LlmError(CardGenError):
    code = "llm_error"
    http_status = 502


class LlmAuthError(LlmError):
    code = "llm_auth_failed"


class LlmRequestError(LlmError):
    code = "llm_request_rejected"


class LlmRateLimitError(LlmError):
    code = "llm_rate_limited"
    http_status = 503


class LlmUnavailableError(LlmError):
    code = "llm_unavailable"
    http_status = 503


class LlmTimeoutError(LlmError):
    code = "llm_timeout"
    http_status = 504


class LlmRefusalError(LlmError):
    code = "llm_refusal"


class LlmOutputError(LlmError):
    code = "llm_bad_output"
