"""FastAPI app for local DetectivePotty event review."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from detectivepotty.config import CONFIG_ENV_VAR, Config, load_config
from detectivepotty.web.dataset_index import DatasetIndex
from detectivepotty.web.event_routes import (
    _event_stream as _event_stream,
    register_event_routes,
)
from detectivepotty.web.label_routes import register_label_routes
from detectivepotty.web.media import no_store_file_response
from detectivepotty.web.middleware import ApiNoStoreMiddleware
from detectivepotty.web.state import init_app_state
from detectivepotty.web.tune_routes import (
    TUNE_POSE_MAX_CROPS as TUNE_POSE_MAX_CROPS,
    register_tune_routes,
)
from detectivepotty.web.tune_tracking import (
    ultralytics_tracking_available as _ultralytics_tracking_available,
)


FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

_BUILD_MISSING_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>DetectivePotty Review</title>
  </head>
  <body style="font-family: system-ui, sans-serif; margin: 3rem; color: #ddd;
               background: #10141c;">
    <h1>DetectivePotty Review</h1>
    <p>The review portal frontend has not been built yet.</p>
    <p>Run the following, then reload:</p>
    <pre>cd src/detectivepotty/web/frontend
npm install
npm run build</pre>
  </body>
</html>
"""


def create_app(
    config: Config,
    *,
    tune_detector: object | None = None,
    tune_pose_estimator: object | None = None,
) -> FastAPI:
    dataset_index = DatasetIndex(config.resolve_path(config.global_settings.dataset_dir))
    dogs = list(config.global_settings.dogs)
    app = FastAPI(title="DetectivePotty", version="0.1.0")
    init_app_state(
        app,
        config,
        dataset_index=dataset_index,
        dogs=dogs,
        tune_detector=tune_detector,
        tune_pose_estimator=tune_pose_estimator,
    )
    app.add_middleware(ApiNoStoreMiddleware)

    # The built Svelte app references hashed files under /assets. Mount it only
    # when the build exists so app creation still succeeds in CI / fresh
    # checkouts where dist/ (gitignored) has not been built yet.
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.is_dir():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        built = FRONTEND_DIST / "index.html"
        if built.is_file():
            return no_store_file_response(built)
        return HTMLResponse(_BUILD_MISSING_HTML)

    register_event_routes(app, dataset_index=dataset_index, dogs=dogs)

    register_tune_routes(
        app,
        config,
        ultralytics_tracking_available=_ultralytics_tracking_available,
    )

    register_label_routes(app)

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_fallback(full_path: str) -> Response:
        """Serve the SPA shell for client-side routes (e.g. /tune, /live).

        Real ``/api/*`` routes and the ``/assets`` mount are registered earlier
        so they win; only unknown non-API paths fall through here so a browser
        refresh on a client route returns index.html instead of 404. API/asset
        paths that reach here are genuinely unknown -> 404 (never HTML).
        """

        if full_path in {"api", "assets"} or full_path.startswith(("api/", "assets/")):
            raise HTTPException(status_code=404, detail="not found")
        built = FRONTEND_DIST / "index.html"
        if built.is_file():
            return no_store_file_response(built)
        return HTMLResponse(_BUILD_MISSING_HTML)

    return app


def create_app_from_env() -> FastAPI:
    """Build the app from ``$DETECTIVEPOTTY_CONFIG`` (default ``config.yaml``).

    uvicorn's ``--reload`` re-imports the app in a worker subprocess, so it needs
    an import string pointing at a zero-arg factory rather than a prebuilt app
    object. The config path is read from the environment because that subprocess
    never re-runs the CLI; it falls back to ``config.yaml`` relative to the
    process working directory.
    """

    return create_app(load_config())


def run_server(
    config: Config,
    host: str = "127.0.0.1",
    port: int = 8000,
    *,
    reload: bool = False,
    config_path: str | Path | None = None,
) -> None:
    """Serve the review app.

    ``reload=True`` (dev only) re-launches via an import string + app factory so
    uvicorn can hot-reload on source changes; the prebuilt ``config`` is ignored
    in that mode because each reloaded worker rebuilds it from ``config_path``
    (passed through ``$DETECTIVEPOTTY_CONFIG``). Without reload it serves the
    already-built app object, which is the production path.
    """

    if reload:
        if config_path is not None:
            os.environ[CONFIG_ENV_VAR] = str(config_path)
        # Watch the whole package so edits to app.py or anything it imports
        # (config, dataset_index, events, ...) trigger a reload.
        reload_dir = str(Path(__file__).resolve().parents[1])
        uvicorn.run(
            "detectivepotty.web.app:create_app_from_env",
            factory=True,
            host=host,
            port=port,
            reload=True,
            reload_dirs=[reload_dir],
        )
        return
    uvicorn.run(create_app(config), host=host, port=port)
