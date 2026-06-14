"""Application state bootstrap for the local review web app."""

from __future__ import annotations

from collections import OrderedDict
import threading
from typing import Any

from fastapi import FastAPI

from detectivepotty.config import Config
from detectivepotty.web.tune import (
    collect_tune_models,
    collect_tune_roots,
    default_tune_model,
)


def init_app_state(
    app: FastAPI,
    config: Config,
    *,
    dataset_index: Any,
    dogs: list[str],
    tune_detector: object | None = None,
    tune_pose_estimator: object | None = None,
) -> None:
    """Attach shared app state used by the API route closures."""

    app.state.dataset_index = dataset_index
    app.state.dogs = dogs
    app.state.config = config
    app.state.tune_roots = collect_tune_roots(config)
    # Root the range-labeling API discovers harvested clips under. Kept separate
    # from the tuner's browse roots (which include it) so /api/label only ever
    # exposes harvested clip dirs, never the wider dataset/data tree.
    app.state.harvest_root = config.resolve_path(config.global_settings.harvest_dir)
    app.state.tune_default_model = default_tune_model(config)
    # Per-model detector cache (model string -> DogDetector), built lazily under
    # ``tune_detector_lock``. An injected detector (tests) seeds the cache for the
    # default model and pins the allow-list to just that model, so no scanning or
    # real model build happens offline.
    if tune_detector is not None:
        app.state.tune_detectors = OrderedDict(
            [(app.state.tune_default_model, tune_detector)]
        )
        app.state.tune_models = [app.state.tune_default_model]
    else:
        app.state.tune_detectors = OrderedDict()
        app.state.tune_models = collect_tune_models(config)
    app.state.tune_detector_cache_size = 2
    app.state.tune_detector_lock = threading.Lock()
    app.state.tune_infer_lock = threading.Lock()
    app.state.tune_pose_lock = threading.Lock()
    app.state.event_label_lock = threading.Lock()
    app.state.clip_label_lock = threading.Lock()
    # Serializes CoreML exports (heavy, macOS-only) triggered from the tuner UI so
    # only one runs at a time.
    app.state.tune_export_lock = threading.Lock()
    # Resolved as (estimator | None, available). Seeded when a fake is injected.
    app.state.tune_pose_resolved = (
        (tune_pose_estimator, True) if tune_pose_estimator is not None else None
    )
