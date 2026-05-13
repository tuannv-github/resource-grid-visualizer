"""Log REST request and response *content* (body or query), not full HTTP client metadata."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .log_config import get_logger

_log = get_logger("resgrid.http")

# Set to 0/false/off to disable this middleware entirely.
_ENABLED = os.environ.get("RESGRID_HTTP_ACCESS_LOG", "1").strip().lower() not in (
    "0",
    "false",
    "off",
    "no",
)

# Max characters logged for JSON/text bodies (full payload may be larger).
_MAX_BODY_CHARS = max(200, int(os.environ.get("RESGRID_REQUEST_BODY_LOG_MAX", "4000")))
_MAX_RESPONSE_BODY_CHARS = max(
    200, int(os.environ.get("RESGRID_RESPONSE_BODY_LOG_MAX", str(_MAX_BODY_CHARS)))
)
_LOG_RESPONSE_BODY = os.environ.get(
    "RESGRID_HTTP_LOG_RESPONSE_BODY", "1"
).strip().lower() not in (
    "0",
    "false",
    "off",
    "no",
)


def _path_with_qs(request: Request) -> str:
    p = request.url.path
    q = request.url.query
    return f"{p}?{q}" if q else p


def _format_request_content(request: Request, body: bytes) -> str:
    """Human-readable description of what the client sent (not headers)."""
    method = request.method
    if method in ("GET", "HEAD", "OPTIONS"):
        q = request.url.query
        if not q:
            return "body: {}"
        return f"query={q}"

    if not body:
        return "body=<empty>"

    ct = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if "json" in ct:
        try:
            text = body.decode("utf-8")
            obj = json.loads(text)
            out = json.dumps(
                obj, ensure_ascii=False, separators=(",", ":"), default=str
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            out = body.decode("utf-8", errors="replace")
    elif ct.startswith("text/") or ct in ("application/x-www-form-urlencoded", ""):
        out = body.decode("utf-8", errors="replace")
    else:
        n = len(body)
        preview = body[: min(64, n)]
        return f"body=<{ct or 'unknown'} {n} bytes> {preview!r}" + (
            "..." if n > 64 else ""
        )

    if len(out) > _MAX_BODY_CHARS:
        return out[:_MAX_BODY_CHARS] + f"... (truncated, {len(body)} bytes total)"
    return out


def _format_response_content(response: Response, body: bytes) -> str:
    """Short summary of response entity-body for logs (status line is logged separately)."""
    if not _LOG_RESPONSE_BODY:
        return "body=<omitted RESGRID_HTTP_LOG_RESPONSE_BODY=0>"
    if not body:
        return "body: {}"
    ct = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
    if "json" in ct:
        try:
            text = body.decode("utf-8")
            obj = json.loads(text)
            out = json.dumps(
                obj, ensure_ascii=False, separators=(",", ":"), default=str
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            out = body.decode("utf-8", errors="replace")
    elif ct.startswith("text/"):
        out = body.decode("utf-8", errors="replace")
    else:
        n = len(body)
        preview = body[: min(64, n)]
        return f"body=<{ct or 'unknown'} {n} bytes> {preview!r}" + (
            "..." if n > 64 else ""
        )

    if len(out) > _MAX_RESPONSE_BODY_CHARS:
        return (
            out[:_MAX_RESPONSE_BODY_CHARS] + f"... (truncated, {len(body)} bytes total)"
        )
    return out


async def _response_with_body(response: Response) -> tuple[Response, bytes]:
    """Materialize streaming bodies so we can log and still return the full payload to the client."""
    if getattr(response, "body", None) is not None:
        return response, response.body or b""
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    raw = b"".join(chunks)
    return (
        Response(
            content=raw,
            status_code=response.status_code,
            headers=response.headers,
            media_type=response.media_type,
            background=response.background,
        ),
        raw,
    )


class HttpAccessLogMiddleware(BaseHTTPMiddleware):
    """Log ``[request]`` before the handler runs, then ``[response]`` after (truncated bodies); status poll at DEBUG."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if not _ENABLED:
            return await call_next(request)

        method = request.method
        path_qs = _path_with_qs(request)
        body = b""
        req = request

        if method in ("POST", "PUT", "PATCH", "DELETE"):
            body = await request.body()

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            req = Request(request.scope, receive=receive)

        content = _format_request_content(request, body)

        req_msg = "[request] %s %s %s" % (method, path_qs, content)
        status_poll = method == "GET" and request.url.path in (
            "/api/plotter/status",
            "/api/plotter/playback",
            "/api/generator/status",
        )
        if status_poll:
            _log.debug(req_msg)
        else:
            _log.info(req_msg)

        try:
            response = await call_next(req)
        except BaseException:
            _log.exception(
                "Handler failed after [request]: %s %s",
                method,
                path_qs,
            )
            raise

        response, resp_body = await _response_with_body(response)
        resp_summary = _format_response_content(response, resp_body)
        res_msg = "[response] %s %s %s %s" % (
            method,
            path_qs,
            response.status_code,
            resp_summary,
        )
        if status_poll:
            _log.debug(res_msg)
        else:
            _log.info(res_msg)
        return response
