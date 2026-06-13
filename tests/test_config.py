from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from detectivepotty.config import (
    CONFIG_ENV_VAR,
    DEFAULT_CONFIG_PATH,
    DEFAULT_DOG_ALIAS_CLASSES,
    CameraInputConfig,
    Config,
    GlobalSettings,
    load_config,
    resolve_config_path,
)


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


def test_global_settings_dog_alias_defaults() -> None:
    g = GlobalSettings()
    assert g.dog_alias_classes == list(DEFAULT_DOG_ALIAS_CLASSES)
    assert g.dog_alias_nms_iou == 0.65


def test_global_settings_dog_alias_validator_normalizes() -> None:
    g = GlobalSettings(dog_alias_classes=["Sheep", "sheep", " COW ", "dog", ""])
    assert g.dog_alias_classes == ["sheep", "cow"]


def test_global_settings_dog_alias_empty_disables() -> None:
    g = GlobalSettings(dog_alias_classes=[])
    assert g.dog_alias_classes == []


def test_resolve_config_path_defaults_to_env_then_config_yaml(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CONFIG_ENV_VAR, raising=False)
    assert resolve_config_path() == DEFAULT_CONFIG_PATH

    monkeypatch.setenv(CONFIG_ENV_VAR, "custom.yaml")
    assert resolve_config_path() == Path("custom.yaml")
    assert resolve_config_path("explicit.yaml") == Path("explicit.yaml")


def test_load_config_uses_default_config_path_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "env-config.yaml"
    config_path.write_text("global:\n  dataset_dir: env-dataset\n", encoding="utf-8")
    monkeypatch.setenv(CONFIG_ENV_VAR, str(config_path))

    cfg = load_config()

    assert cfg.global_settings.dataset_dir == Path("env-dataset")


def test_config_reports_protect_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Config.model_validate(
        {
            "protect": {
                "nvr_host": "https://nvr.local",
                "username_env": "DP_USER",
                "password_env": "DP_PASS",
            }
        }
    )
    assert not cfg.protect_configured()

    monkeypatch.setenv("DP_USER", "user")
    monkeypatch.setenv("DP_PASS", "pass")
    assert cfg.protect_configured()
