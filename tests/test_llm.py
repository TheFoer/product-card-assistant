import pytest
from stub_llm import StubLlm, fail, ok

from cardgen.config import Settings
from cardgen.errors import (
    ConfigError,
    LlmAuthError,
    LlmOutputError,
    LlmRateLimitError,
    LlmRefusalError,
    LlmRequestError,
    LlmTimeoutError,
    LlmUnavailableError,
)
from cardgen.llm import OpenAIClient
from cardgen.schemas import LLM_JSON_SCHEMA

MESSAGES = [{"role": "user", "content": "x"}]


def client(stub, delays=None, **overrides):
    settings = Settings(api_key="secret-key", base_url=stub.url, **overrides)
    sleep = delays.append if delays is not None else (lambda _: None)
    return OpenAIClient(settings, sleep=sleep)


def test_missing_key_is_a_config_error():
    with pytest.raises(ConfigError):
        OpenAIClient(Settings(api_key=""))


def test_key_not_in_settings_repr():
    assert "secret-key" not in repr(Settings(api_key="secret-key"))


def test_request_shape():
    with StubLlm([ok({"a": 1})]) as stub:
        response = client(stub, model="m").complete("sys", MESSAGES)
    sent = stub.requests[0]
    assert sent["model"] == "m"
    assert sent["messages"][0] == {"role": "developer", "content": "sys"}
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert sent["response_format"]["json_schema"]["schema"] == LLM_JSON_SCHEMA
    assert "max_completion_tokens" in sent and "max_tokens" not in sent
    assert stub.headers[0]["authorization"] == "Bearer secret-key"
    assert response.data == {"a": 1}
    assert (response.input_tokens, response.output_tokens, response.request_id) == (10, 20, "req_stub")


def test_exhausted_balance_is_not_retried():
    body = {"error": {"code": "insufficient_quota", "message": "quota"}}
    with StubLlm([fail(429, body=body)]) as stub:
        with pytest.raises(LlmRateLimitError, match="баланс"):
            client(stub).complete("s", MESSAGES)
    assert len(stub.requests) == 1


def test_schema_uses_only_supported_keywords():
    banned = {"minLength", "maxLength", "minimum", "maximum", "maxItems", "pattern"}

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for key, value in node.items():
                if key != "properties":
                    assert key not in banned
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(LLM_JSON_SCHEMA)


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_errors_are_retried(status):
    with StubLlm([fail(status), ok({"a": 1})]) as stub:
        response = client(stub, delays=[]).complete("s", MESSAGES)
    assert response.data == {"a": 1}
    assert len(stub.requests) == 2


def test_retry_after_is_honoured_and_capped():
    delays = []
    with StubLlm([fail(429, {"retry-after": "7"}), fail(429, {"retry-after": "999"}), ok({})]) as stub:
        client(stub, delays=delays).complete("s", MESSAGES)
    assert delays == [7.0, 30.0]


@pytest.mark.parametrize("status, error", [
    (429, LlmRateLimitError),
    (503, LlmUnavailableError),
    (504, LlmTimeoutError),
])
def test_retries_are_limited(status, error):
    with StubLlm([fail(status)]) as stub:
        with pytest.raises(error):
            client(stub, max_retries=2).complete("s", MESSAGES)
    assert len(stub.requests) == 3


@pytest.mark.parametrize("status, error", [
    (400, LlmRequestError), (401, LlmAuthError), (403, LlmAuthError), (404, LlmRequestError),
])
def test_permanent_errors_are_not_retried(status, error):
    with StubLlm([fail(status)]) as stub:
        with pytest.raises(error):
            client(stub).complete("s", MESSAGES)
    assert len(stub.requests) == 1


def test_html_error_body_does_not_crash():
    with StubLlm([fail(400, body="<html>bad gateway</html>")]) as stub:
        with pytest.raises(LlmRequestError):
            client(stub).complete("s", MESSAGES)


def test_network_failure_maps_to_unavailable():
    settings = Settings(api_key="k", base_url="http://127.0.0.1:9", max_retries=1)
    with pytest.raises(LlmUnavailableError):
        OpenAIClient(settings, sleep=lambda _: None).complete("s", MESSAGES)


def test_refusal_and_truncation():
    with StubLlm([ok("", refusal="нет")]) as stub:
        with pytest.raises(LlmRefusalError):
            client(stub).complete("s", MESSAGES)
    with StubLlm([ok("{", finish_reason="length")]) as stub:
        with pytest.raises(LlmOutputError, match="LLM_MAX_TOKENS"):
            client(stub).complete("s", MESSAGES)


def test_unexpected_payload_and_bad_json():
    with StubLlm([{"status": 200, "headers": {}, "body": {"foo": 1}}]) as stub:
        with pytest.raises(LlmOutputError):
            client(stub).complete("s", MESSAGES)
    with StubLlm([ok("не json")]) as stub:
        assert client(stub).complete("s", MESSAGES).data is None
