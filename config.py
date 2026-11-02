"""Настройки из переменных окружения. Ключ API никогда не хранится в коде."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field

from cardgen.errors import ConfigError


def _int(env: Mapping[str, str], name: str, default: int, minimum: int) -> int:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name}: ожидается целое число, получено {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{name}: значение должно быть не меньше {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    # repr=False: ключ не попадёт в логи и трассировки при выводе настроек.
    api_key: str = field(default="", repr=False)
    base_url: str = "https://api.anthropic.com"
    model: str = "claude-sonnet-5-5"
    max_tokens: int = 8000
    timeout_seconds: int = 120
    max_retries: int = 3
    max_repairs: int = 2
    log_level: str = "INFO"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        env = os.environ if env is None else env
        return cls(
            api_key=env.get("ANTHROPIC_API_KEY", "").strip(),
            base_url=(env.get("ANTHROPIC_BASE_URL", "").strip() or cls.base_url).rstrip("/"),
            model=env.get("LLM_MODEL", "").strip() or cls.model,
            max_tokens=_int(env, "LLM_MAX_TOKENS", cls.max_tokens, 1000),
            timeout_seconds=_int(env, "LLM_TIMEOUT_SECONDS", cls.timeout_seconds, 1),
            max_retries=_int(env, "LLM_MAX_RETRIES", cls.max_retries, 0),
            max_repairs=_int(env, "CARD_MAX_REPAIRS", cls.max_repairs, 0),
            log_level=env.get("LOG_LEVEL", "").strip().upper() or cls.log_level,
        )
