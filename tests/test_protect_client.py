from __future__ import annotations

import asyncio
from types import SimpleNamespace

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
