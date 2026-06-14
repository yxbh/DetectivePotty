from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest
from uiprotect.exceptions import NotAuthorized

from detectivepotty.config import Config, ProtectConfig
from detectivepotty.protect.client import ProtectClient


class FakeApi:
    def __init__(self) -> None:
        high = SimpleNamespace(
            id=0,
            name="High",
            enabled=True,
            is_rtsp_enabled=True,
            width=2688,
            height=1512,
            fps=30,
            rtsps_url="rtsps://user:pass@nvr.local:7441/high?token=secret&quality=high",
        )
        medium = SimpleNamespace(
            id=1,
            name="Medium",
            enabled=True,
            is_rtsp_enabled=True,
            width=1280,
            height=720,
            fps=15,
            rtsps_url="rtsps://nvr.local:7441/medium?access_token=secret&quality=medium",
        )
        low = SimpleNamespace(
            id=2,
            name="Low",
            enabled=True,
            is_rtsp_enabled=True,
            width=640,
            height=360,
            fps=10,
            rtsps_url="rtsps://nvr.local:7441/low?token=secret",
        )
        camera = SimpleNamespace(
            id="cam-1",
            name="Backyard",
            is_connected=True,
            feature_flags=SimpleNamespace(smart_detect_types=["animal", "person"]),
            channels=[high, medium, low],
        )
        self.bootstrap = SimpleNamespace(cameras={"cam-1": camera})
        self.closed = False

    async def update(self) -> None:
        return None

    async def async_disconnect_ws(self) -> None:
        self.closed = True

    async def close_session(self) -> None:
        self.closed = True


def test_list_cameras_returns_capabilities_and_sanitized_urls() -> None:
    async def run() -> None:
        client = ProtectClient(
            Config(
                protect=ProtectConfig(
                    nvr_host="nvr.local",
                    api_key_env=None,
                    username_env=None,
                    password_env=None,
                )
            ),
            api_client=FakeApi(),
        )

        cameras = await client.list_cameras()

        assert len(cameras) == 1
        camera = cameras[0]
        assert camera.id == "cam-1"
        assert camera.name == "Backyard"
        assert camera.is_connected is True
        assert camera.animal_smart_detect_supported is True
        assert camera.channels[0].sanitized_rtsps_url == "rtsps://nvr.local:7441/high?quality=high"
        assert camera.channels[1].sanitized_rtsps_url == "rtsps://nvr.local:7441/medium?quality=medium"
        assert "secret" not in repr(camera)
        assert "user:pass" not in repr(camera)

    asyncio.run(run())


def test_rtsps_substream_mapping_uses_high_middle_low_order() -> None:
    async def run() -> ProtectClient:
        client = ProtectClient(
            Config(
                protect=ProtectConfig(
                    nvr_host="nvr.local",
                    api_key_env=None,
                    username_env=None,
                    password_env=None,
                )
            ),
            api_client=FakeApi(),
        )
        await client.connect()
        return client

    client = asyncio.run(run())

    assert client.rtsps_url("cam-1", "high").endswith("/high?token=secret&quality=high")
    assert client.rtsps_url("cam-1", "medium").endswith("/medium?access_token=secret&quality=medium")
    assert client.rtsps_url("cam-1", "low").endswith("/low?token=secret")


class ConnectFakeApi:
    """Fake uiprotect client that records private/public bootstrap calls."""

    def __init__(
        self,
        *,
        update_error: Exception | None = None,
        update_error_once: bool = False,
    ) -> None:
        self.update_calls = 0
        self.update_public_calls = 0
        self.authenticate_calls = 0
        self._update_error = update_error
        self._update_error_once = update_error_once

        camera = SimpleNamespace(
            id="cam-1",
            name="Backyard",
            is_connected=True,
            feature_flags=SimpleNamespace(smart_detect_types=["animal"]),
            channels=[],
        )
        self.bootstrap = SimpleNamespace(cameras={"cam-1": camera})
        self.public_bootstrap = SimpleNamespace(cameras={"cam-1": camera}, rtsps_streams={})

    async def update(self) -> None:
        self.update_calls += 1
        if self._update_error is not None and not (
            self._update_error_once and self.update_calls > 1
        ):
            raise self._update_error

    async def authenticate(self) -> None:
        self.authenticate_calls += 1

    async def update_public(self) -> None:
        self.update_public_calls += 1

    async def get_camera_rtsps_streams(self, _camera_id: str) -> None:
        return None

    async def async_disconnect_ws(self) -> None:
        return None

    async def close_session(self) -> None:
        return None

    async def close_public_api_session(self) -> None:
        return None


def _mixed_secret_config() -> Config:
    return Config(protect=ProtectConfig(nvr_host="nvr.local"))


def _set_all_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DETECTIVEPOTTY_NVR_API_KEY", "key-123")
    monkeypatch.setenv("DETECTIVEPOTTY_NVR_USERNAME", "user")
    monkeypatch.setenv("DETECTIVEPOTTY_NVR_PASSWORD", "pass")


def test_connect_private_success_skips_public_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_all_secrets(monkeypatch)
    api = ConnectFakeApi()
    client = ProtectClient(_mixed_secret_config(), api_client=api)

    asyncio.run(client.connect())

    assert api.update_calls == 1
    assert api.update_public_calls == 0
    assert client._private_enabled is True  # noqa: SLF001
    assert client._public_enabled is False  # noqa: SLF001


def test_connect_private_retries_once_on_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_all_secrets(monkeypatch)
    api = ConnectFakeApi(update_error=NotAuthorized("401"), update_error_once=True)
    client = ProtectClient(_mixed_secret_config(), api_client=api)

    asyncio.run(client.connect())

    assert api.authenticate_calls == 1
    assert api.update_calls == 2
    assert api.update_public_calls == 0
    assert client._private_enabled is True  # noqa: SLF001
    assert client._public_enabled is False  # noqa: SLF001


def test_connect_falls_back_to_public_when_private_auth_fails(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _set_all_secrets(monkeypatch)
    api = ConnectFakeApi(update_error=NotAuthorized("401"))
    client = ProtectClient(_mixed_secret_config(), api_client=api)

    with caplog.at_level(logging.WARNING, logger="detectivepotty.protect.client"):
        asyncio.run(client.connect())

    # Private was attempted (login + retry) and then fell back to public.
    assert api.authenticate_calls == 1
    assert api.update_calls == 2
    assert api.update_public_calls == 1
    assert client._private_enabled is False  # noqa: SLF001
    assert client._public_enabled is True  # noqa: SLF001
    assert "private auth failed" in caplog.text

    client._public_rtsps_streams["cam-1"] = object()  # noqa: SLF001
    asyncio.run(client.close())

    assert client._connected is False  # noqa: SLF001
    assert client._private_enabled is False  # noqa: SLF001
    assert client._public_enabled is False  # noqa: SLF001
    assert client._public_rtsps_streams == {}  # noqa: SLF001


def test_connect_raises_when_private_fails_and_no_public_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DETECTIVEPOTTY_NVR_API_KEY", raising=False)
    monkeypatch.setenv("DETECTIVEPOTTY_NVR_USERNAME", "user")
    monkeypatch.setenv("DETECTIVEPOTTY_NVR_PASSWORD", "pass")
    api = ConnectFakeApi(update_error=NotAuthorized("401"))
    client = ProtectClient(_mixed_secret_config(), api_client=api)

    with pytest.raises(NotAuthorized):
        asyncio.run(client.connect())

    assert api.update_public_calls == 0
    assert client._private_enabled is False  # noqa: SLF001
    assert client._public_enabled is False  # noqa: SLF001
