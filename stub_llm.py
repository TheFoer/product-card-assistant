"""Локальный HTTP-сервер вместо Anthropic API. Клиент проверяется по настоящему HTTP."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


def ok(data: Any, *, thinking: bool = False, stop_reason: str = "end_turn") -> dict[str, Any]:
    content: list[dict[str, Any]] = []
    if thinking:
        content.append({"type": "thinking", "thinking": "...", "signature": "sig"})
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    content.append({"type": "text", "text": text})
    return {
        "status": 200,
        "headers": {"request-id": "req_stub"},
        "body": {
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 10, "output_tokens": 20},
        },
    }


def fail(status: int, headers: dict[str, str] | None = None, body: Any = None) -> dict[str, Any]:
    return {
        "status": status,
        "headers": headers or {},
        "body": body if body is not None else {
            "type": "error", "error": {"type": "stub", "message": f"stub {status}"},
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
