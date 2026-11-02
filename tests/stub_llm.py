"""Локальный HTTP-сервер вместо OpenAI API. Клиент проверяется по настоящему HTTP."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def ok(data: Any, *, finish_reason: str = "stop", refusal: str | None = None) -> dict[str, Any]:
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return {
        "status": 200,
        "headers": {"x-request-id": "req_stub"},
        "body": {
            "choices": [{
                "message": {"role": "assistant", "content": text, "refusal": refusal},
                "finish_reason": finish_reason,
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        },
    }


def fail(status: int, headers: dict[str, str] | None = None, body: Any = None) -> dict[str, Any]:
    return {
        "status": status,
        "headers": headers or {},
        "body": body if body is not None else {
            "error": {"type": "stub", "code": "stub", "message": f"stub {status}"},
        },
    }


class StubLlm:
    def __init__(self, script: list[dict[str, Any]]) -> None:
        self.script = list(script)
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("content-length", "0"))
                stub.requests.append(json.loads(self.rfile.read(length)))
                stub.headers.append({key.lower(): value for key, value in self.headers.items()})
                reply = stub.script.pop(0) if len(stub.script) > 1 else stub.script[0]
                raw = reply["body"]
                payload = raw.encode() if isinstance(raw, str) else json.dumps(raw).encode()
                self.send_response(reply["status"])
                for key, value in reply["headers"].items():
                    self.send_header(key, value)
                self.send_header("content-length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "StubLlm":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()
