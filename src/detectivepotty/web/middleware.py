"""ASGI middleware used by the local review web app."""

from __future__ import annotations

from starlette.datastructures import MutableHeaders


class ApiNoStoreMiddleware:
    """Default ``Cache-Control: no-store`` for /api/ responses.

    Implemented as pure ASGI (not ``BaseHTTPMiddleware``) because the latter
    buffers the whole response body, which breaks streaming endpoints. This only
    rewrites the response *start* headers, so streamed bodies flush incrementally.
    Endpoints that set their own Cache-Control are left untouched.
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith("/api/"):
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(raw=message["headers"])
                if "cache-control" not in headers:
                    headers["cache-control"] = "no-store"
            await send(message)

        await self.app(scope, receive, send_wrapper)
