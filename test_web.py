import json

import pytest
from stub_llm import StubLlm, fail, ok

from cardgen.config import Settings
from cardgen.web import create_app


def client_for(url="http://127.0.0.1:9", api_key="k"):
    settings = Settings(api_key=api_key, base_url=url, max_retries=0)
    return create_app(settings).test_client()


def post(client, path, body):
    data = body if isinstance(body, (str, bytes)) else json.dumps(body, ensure_ascii=False)
    return client.post(path, data=data, content_type="application/json")


def test_form_and_health():
    client = client_for()
    assert "Создать карточку" in client.get("/").get_data(as_text=True)
    assert client.get("/healthz").get_json()["llm_configured"] is True


def test_form_warns_without_key():
    assert "ANTHROPIC_API_KEY" in client_for(api_key="").get("/").get_data(as_text=True)


def test_generate(dm02_input, dm02_response):
    with StubLlm([ok(dm02_response)]) as stub:
        response = post(client_for(stub.url), "/api/v1/cards", dm02_input)
    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "ready"
    assert body["markdown"].startswith("# Манометр")
    assert "X-Request-Id" in response.headers


def test_check_needs_no_key(dm02_input):
    response = post(client_for(api_key=""), "/api/v1/check", dm02_input)
    assert response.status_code == 200
    assert "Питание" in response.get_json()["missing_fields"]


@pytest.mark.parametrize("body, status, code", [
    ("{broken", 400, "invalid_json"),
    ("", 400, "invalid_json"),
    ("[]", 422, "invalid_product"),
    ({"name": "Только имя"}, 422, "invalid_product"),
])
def test_bad_input_never_reaches_llm(body, status, code):
    with StubLlm([ok({})]) as stub:
        response = post(client_for(stub.url), "/api/v1/cards", body)
    assert response.status_code == status
    assert response.get_json()["error"]["code"] == code
    assert stub.requests == []


def test_wrong_content_type():
    response = client_for().post("/api/v1/cards", data="name=x", content_type="application/x-www-form-urlencoded")
    assert response.status_code == 415


def test_body_too_large():
    response = post(client_for(), "/api/v1/cards", "x" * (1024 * 1024 + 1))
    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "http_413"


def test_missing_key_on_generate(dm02_input):
    response = post(client_for(api_key=""), "/api/v1/cards", dm02_input)
    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "not_configured"


@pytest.mark.parametrize("llm_status, status, code", [
    (401, 502, "llm_auth_failed"),
    (429, 503, "llm_rate_limited"),
    (529, 503, "llm_unavailable"),
    (400, 502, "llm_request_rejected"),
])
def test_llm_errors_are_mapped(dm02_input, llm_status, status, code):
    with StubLlm([fail(llm_status)]) as stub:
        response = post(client_for(stub.url), "/api/v1/cards", dm02_input)
    assert response.status_code == status
    assert response.get_json()["error"]["code"] == code


def test_unexpected_error_does_not_leak(dm02_input):
    class Boom:
        def generate(self, product):
            raise RuntimeError("секрет")

    client = create_app(Settings(api_key="k"), generator_factory=Boom).test_client()
    response = post(client, "/api/v1/cards", dm02_input)
    assert response.status_code == 500
    assert "секрет" not in response.get_data(as_text=True)


def test_unknown_route_is_json():
    response = client_for().get("/nope")
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "http_404"
