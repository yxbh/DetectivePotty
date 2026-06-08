from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from detectivepotty.config import CameraInputConfig


def test_rtsp_input_requires_url_env() -> None:
    with pytest.raises(ValidationError):
        CameraInputConfig(kind="rtsp")


def test_rtsp_input_rejects_path() -> None:
    with pytest.raises(ValidationError):
        CameraInputConfig(kind="rtsp", url_env="POOL_RTSP_URL", path=Path("/tmp/x.mp4"))


def test_url_env_only_valid_for_rtsp_kind() -> None:
    with pytest.raises(ValidationError):
        CameraInputConfig(kind="file", url_env="POOL_RTSP_URL")
    with pytest.raises(ValidationError):
        CameraInputConfig(kind="protect", url_env="POOL_RTSP_URL")


def test_url_env_must_be_environment_variable_name() -> None:
    with pytest.raises(ValidationError):
        CameraInputConfig(kind="rtsp", url_env="bad-name!")


def test_rtsp_input_valid_minimal() -> None:
    cfg = CameraInputConfig(kind="rtsp", url_env="POOL_RTSP_URL")
    assert cfg.kind == "rtsp"
    assert cfg.url_env == "POOL_RTSP_URL"


def test_resolve_url_reads_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POOL_RTSP_URL", "rtsp://user:pass@192.168.1.37:554/cam")
    cfg = CameraInputConfig(kind="rtsp", url_env="POOL_RTSP_URL")
    assert cfg.resolve_url() == "rtsp://user:pass@192.168.1.37:554/cam"


def test_resolve_url_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("POOL_RTSP_URL", raising=False)
    cfg = CameraInputConfig(kind="rtsp", url_env="POOL_RTSP_URL")
    assert cfg.resolve_url() is None
    assert CameraInputConfig(kind="file").resolve_url() is None
