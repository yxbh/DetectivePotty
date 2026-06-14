"""Consistent JSON error responses for the local web API."""

from __future__ import annotations

import re
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


def error_payload(detail: Any) -> dict[str, dict[str, str]]:
    """Return the canonical API error body for an HTTP exception detail."""

    if isinstance(detail, dict):
        code = detail.get("code")
        message = detail.get("message")
        if isinstance(code, str) and isinstance(message, str):
            return {"error": {"code": code, "message": message}}

    message = detail if isinstance(detail, str) else str(detail)
    return {"error": {"code": _error_code(message), "message": message}}


async def http_exception_handler(
    _request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Render explicit route errors as ``{"error": {"code", "message"}}``."""

    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload(exc.detail),
        headers=exc.headers,
    )


def _error_code(message: str) -> str:
    base = message.split(":", 1)[0]
    code = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
    return code or "http_error"
