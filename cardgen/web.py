"""HTTP API и веб-форма."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable

from flask import Flask, Response, g, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from cardgen.analysis import parse_json, parse_product, preflight
from cardgen.config import Settings
from cardgen.errors import CardGenError
from cardgen.llm import OpenAIClient
from cardgen.service import CardGenerator

log = logging.getLogger(__name__)

MAX_BODY_BYTES = 1024 * 1024


def _error(code: str, message: str, status: int, details: list[dict[str, str]] | None = None):
    return jsonify(error={
        "code": code,
        "message": message,
        "details": details or [],
        "request_id": g.get("request_id", ""),
    }), status


def create_app(
    settings: Settings | None = None,
    generator_factory: Callable[[], CardGenerator] | None = None,
) -> Flask:
    settings = settings or Settings.from_env()

    def default_factory() -> CardGenerator:
        return CardGenerator(OpenAIClient(settings), settings.max_repairs)

    factory = generator_factory or default_factory

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_BODY_BYTES
    app.json.ensure_ascii = False

    @app.before_request
    def _start() -> None:
        g.request_id = uuid.uuid4().hex[:12]
        g.started = time.monotonic()

    @app.after_request
    def _finish(response: Response) -> Response:
        # В лог идут только технические данные: содержимое товара и ключ не пишутся.
        log.info("%s %s -> %s за %.0f мс [%s]", request.method, request.path, response.status_code,
                 (time.monotonic() - g.started) * 1000, g.request_id)
        response.headers["X-Request-Id"] = g.request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.errorhandler(CardGenError)
    def _domain_error(exc: CardGenError):
        return _error(exc.code, exc.message, exc.http_status, exc.details)

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        status = exc.code or 500
        messages = {
            404: "Адрес не найден",
            405: "Метод не поддерживается",
            413: "Слишком большой запрос: допускается не более 1 МБ",
        }
        return _error(f"http_{status}", messages.get(status, exc.description or "Ошибка запроса"), status)

    @app.errorhandler(Exception)
    def _unexpected(exc: Exception):
        log.exception("Необработанная ошибка [%s]", g.get("request_id", ""))
        return _error("internal_error", "Внутренняя ошибка сервиса", 500)

    def _read_product():
        # Требование JSON заодно закрывает простые межсайтовые POST-запросы к локальному сервису.
        if request.mimetype != "application/json":
            return None
        body = request.get_data(cache=False)
        return parse_product(parse_json(body))

    @app.get("/")
    def index():
        return render_template("index.html", llm_configured=bool(settings.api_key), model=settings.model)

    @app.get("/healthz")
    def health():
        return jsonify(status="ok", llm_configured=bool(settings.api_key), model=settings.model)

    @app.post("/api/v1/check")
    def check():
        product = _read_product()
        if product is None:
            return _error("unsupported_media_type", "Нужен заголовок Content-Type: application/json", 415)
        return jsonify(preflight(product).model_dump())

    @app.post("/api/v1/cards")
    def cards():
        product = _read_product()
        if product is None:
            return _error("unsupported_media_type", "Нужен заголовок Content-Type: application/json", 415)
        return jsonify(factory().generate(product).model_dump())

    return app

