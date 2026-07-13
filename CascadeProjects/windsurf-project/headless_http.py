#!/usr/bin/env python3
"""Low-overhead HTTP boundary for Weaver headless v2 routes."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Awaitable, Callable

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

from headless_schemas import PublicHTTPError, PublicHTTPErrorEnvelope


MAX_V2_BODY_BYTES = 32_768
MAX_V2_QUERY_BYTES = 2_048
MAX_V2_HEADER_BYTES = 16_384
MAX_V2_HEADERS = 64
CORRELATION_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
V2_HTTP_ALLOWLIST = frozenset({
    ("POST", "/headless/v2/session"),
    ("POST", "/headless/v2/session/renew"),
    ("DELETE", "/headless/v2/session"),
    ("GET", "/headless/v2/state"),
    ("GET", "/headless/v2/memory"),
    ("POST", "/headless/v2/chat/stream"),
    ("POST", "/headless/v2/voice/synth"),
})


def correlation_id(value: str = "") -> str:
    candidate = str(value or "").strip()
    return candidate if CORRELATION_PATTERN.fullmatch(candidate) else f"req-{uuid.uuid4().hex[:24]}"


def request_correlation_id(request: Request) -> str:
    return correlation_id(getattr(request.state, "correlation_id", ""))


class HeadlessHTTPError(HTTPException):
    """Stable public failure with no internal diagnostic content."""

    def __init__(self, status_code: int, code: str, *, retryable: bool = False) -> None:
        super().__init__(status_code=status_code, detail=code)
        self.code = code
        self.retryable = retryable


async def headless_http_error_handler(
    request: Request,
    exc: HeadlessHTTPError,
) -> JSONResponse:
    request_id = request_correlation_id(request)
    envelope = PublicHTTPErrorEnvelope(
        error=PublicHTTPError(
            code=exc.code,
            retryable=exc.retryable,
            correlation_id=request_id,
        )
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=envelope.model_dump(mode="json"),
        headers={
            "Cache-Control": "no-store",
            "X-Correlation-ID": request_id,
        },
    )


class HeadlessBoundaryMiddleware:
    """Enforce the small v2 HTTP surface without slowing legacy routes."""

    def __init__(self, app: Callable[..., Awaitable[Any]]) -> None:
        self.app = app

    @staticmethod
    async def _reject(send: Callable[..., Awaitable[Any]], status: int, code: str, request_id: str) -> None:
        payload = PublicHTTPErrorEnvelope(
            error=PublicHTTPError(
                code=code,
                retryable=False,
                correlation_id=request_id,
            )
        ).model_dump(mode="json")
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        await send({
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
                (b"cache-control", b"no-store"),
                (b"x-correlation-id", request_id.encode("ascii")),
                (b"x-content-type-options", b"nosniff"),
            ],
        })
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = str(scope.get("path") or "")
        if scope.get("type") != "http" or not path.startswith("/headless/v2/"):
            await self.app(scope, receive, send)
            return

        raw_headers = list(scope.get("headers") or [])
        headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in raw_headers
        }
        request_id = correlation_id(
            headers.get("x-correlation-id", "")
            or str(scope.get("state", {}).get("correlation_id", ""))
        )
        scope.setdefault("state", {})["correlation_id"] = request_id
        method = str(scope.get("method") or "").upper()
        dynamic_memory_delete = (
            method == "DELETE"
            and re.fullmatch(r"/headless/v2/memory/mem-[0-9a-f]{24}", path)
        )
        dynamic_chat_cancel = (
            method == "DELETE"
            and re.fullmatch(r"/headless/v2/chat/turn-[0-9a-f]{24}", path)
        )
        if (
            (method, path) not in V2_HTTP_ALLOWLIST
            and not dynamic_memory_delete
            and not dynamic_chat_cancel
        ):
            await self._reject(send, 404, "invalid-request", request_id)
            return
        if len(raw_headers) > MAX_V2_HEADERS or sum(
            len(key) + len(value) for key, value in raw_headers
        ) > MAX_V2_HEADER_BYTES:
            await self._reject(send, 400, "invalid-request", request_id)
            return
        if len(scope.get("query_string") or b"") > MAX_V2_QUERY_BYTES or scope.get("query_string"):
            await self._reject(send, 400, "invalid-request", request_id)
            return
        content_length = headers.get("content-length", "0").strip()
        try:
            body_bytes = int(content_length or "0")
        except ValueError:
            await self._reject(send, 400, "invalid-request", request_id)
            return
        if body_bytes > MAX_V2_BODY_BYTES:
            await self._reject(send, 413, "request-too-large", request_id)
            return
        body_route = method == "POST" and path in {
            "/headless/v2/chat/stream",
            "/headless/v2/voice/synth",
        }
        downstream_receive = receive
        if body_route:
            content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            if content_type != "application/json" or headers.get("content-encoding", "identity") not in {
                "",
                "identity",
            }:
                await self._reject(send, 400, "invalid-request", request_id)
                return
            parts: list[bytes] = []
            received = 0
            more_body = True
            while more_body:
                message = await receive()
                if message.get("type") != "http.request":
                    await self._reject(send, 400, "invalid-request", request_id)
                    return
                chunk = bytes(message.get("body") or b"")
                received += len(chunk)
                if received > MAX_V2_BODY_BYTES:
                    await self._reject(send, 413, "request-too-large", request_id)
                    return
                parts.append(chunk)
                more_body = bool(message.get("more_body", False))
            body = b"".join(parts)
            replayed = False

            async def _replay_receive() -> dict[str, Any]:
                nonlocal replayed
                if not replayed:
                    replayed = True
                    return {"type": "http.request", "body": body, "more_body": False}
                return await receive()

            downstream_receive = _replay_receive
        # All other current v2 HTTP routes are bodyless. State frames carry
        # their own strict 32 KiB limit in the WebSocket transport.
        elif body_bytes != 0 or "transfer-encoding" in headers:
            await self._reject(send, 400, "invalid-request", request_id)
            return

        async def _send_with_headers(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                response_headers = list(message.get("headers") or [])
                lower_names = {key.lower() for key, _ in response_headers}
                for key, value in (
                    (b"x-correlation-id", request_id.encode("ascii")),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                ):
                    if key not in lower_names:
                        response_headers.append((key, value))
                message["headers"] = response_headers
            await send(message)

        await self.app(scope, downstream_receive, _send_with_headers)
