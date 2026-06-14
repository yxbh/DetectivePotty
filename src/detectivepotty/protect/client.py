"""UniFi Protect client wrapper."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
import threading
from typing import Any, Literal
from urllib.parse import urlsplit

import aiohttp
from uiprotect import ProtectApiClient
from uiprotect.exceptions import BadRequest, ClientError, NotAuthorized, NvrError

from detectivepotty.config import Config
from detectivepotty.sources.base import sanitize_source_id

LOGGER = logging.getLogger(__name__)

Substream = Literal["low", "medium", "high"]

# Auth/transport failures that should trigger a fall back to the public API
# (when an API key is configured) rather than crash. Deliberately scoped: this
# excludes programming/validation errors so genuine bugs still surface.
_PRIVATE_CONNECT_ERRORS: tuple[type[BaseException], ...] = (
    NotAuthorized,
    NvrError,
    BadRequest,
    ClientError,
    aiohttp.ClientError,
    OSError,
    asyncio.TimeoutError,
)

# The public Integration API bootstrap is heavy (it fans out to every
# integration endpoint and then caches RTSPS streams for every camera). The
# pipeline runs one ProtectClient per camera thread, so when several cameras
# fall back to the public API at once they stampede UniFi's per-endpoint rate
# limit (HTTP 429). Serialize the public bootstrap across threads so only one
# runs at a time. Each ProtectClient runs in its own event loop (via
# asyncio.run in a worker thread), so a threading.Lock is the correct
# cross-thread primitive and is only held briefly during startup.
_PUBLIC_BOOTSTRAP_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class ProtectCameraChannel:
    """Secret-free channel summary. ``sanitized_rtsps_url`` is safe to persist."""

    index: int
    id: int | str | None
    name: str
    enabled: bool
    is_rtsp_enabled: bool
    width: int | None = None
    height: int | None = None
    fps: int | None = None
    sanitized_rtsps_url: str | None = None


@dataclass(frozen=True, slots=True)
class ProtectCameraInfo:
    id: str
    name: str
    is_connected: bool
    animal_smart_detect_supported: bool
    channels: tuple[ProtectCameraChannel, ...]


class ProtectClient:
    """Small async wrapper around :class:`uiprotect.ProtectApiClient`."""

    def __init__(
        self,
        config: Config,
        *,
        api_client: Any | None = None,
        ws_timeout: int = 30,
    ) -> None:
        self.config = config
        self._api: Any | None = api_client
        self._ws_timeout = ws_timeout
        self._connected = False
        self._private_enabled = False
        self._public_enabled = False
        self._public_rtsps_streams: dict[str, Any] = {}

    @property
    def api(self) -> Any:
        if self._api is None:
            self._api = self._build_api_client()
        return self._api

    async def __aenter__(self) -> "ProtectClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        await self.close()

    async def connect(self) -> None:
        """Populate uiprotect bootstrap data while honoring TLS verification.

        Private auth is preferred. When it succeeds the public Integration API
        bootstrap is skipped entirely: ``rtsps_url``/``snapshot``/
        ``download_recording`` all use the private bootstrap, so re-running the
        heavy public bootstrap per camera is pure waste and trips UniFi's rate
        limit. The public API is only used as a fallback when private auth is
        unavailable or fails (e.g. a transient ``401`` or a revoked password).
        """

        if self._connected:
            return

        api = self.api
        api_key = self.config.resolve_secret("api_key")
        username = self.config.resolve_secret("username")
        password = self.config.resolve_secret("password")
        want_private = bool(username and password) or self._is_injected_client()
        want_public = bool(api_key)

        private_error: Exception | None = None
        if want_private and hasattr(api, "update"):
            try:
                await self._connect_private(api)
                self._private_enabled = True
            except _PRIVATE_CONNECT_ERRORS as exc:
                private_error = exc

        # Only bootstrap the public API when private auth is not available or
        # failed. Serialize it so concurrent fallbacks don't re-stampede.
        if want_public and not self._private_enabled and hasattr(api, "update_public"):
            try:
                with _PUBLIC_BOOTSTRAP_LOCK:
                    await api.update_public()
                    # ``_cache_public_rtsps_streams`` reads the public bootstrap,
                    # which is gated on ``_public_enabled``; flip it first.
                    self._public_enabled = True
                    await self._cache_public_rtsps_streams()
            except _PRIVATE_CONNECT_ERRORS as exc:
                self._public_enabled = False
                if private_error is not None:
                    raise RuntimeError(
                        "Protect connect failed: private auth error "
                        f"({private_error!r}) and public API error ({exc!r})"
                    ) from exc
                raise

        if private_error is not None and self._public_enabled:
            LOGGER.warning(
                "Protect private auth failed (%s); continuing with the public "
                "Integration API. Recording downloads and private websocket "
                "triggers are unavailable.",
                private_error,
            )
        elif private_error is not None and not self._public_enabled:
            raise private_error

        self._connected = True

    async def _connect_private(self, api: Any) -> None:
        """Run the private bootstrap, refreshing the session once on a 401.

        A persisted ``rememberMe`` session token can be accepted locally yet
        rejected by the NVR (stale/rotated), and uiprotect does not refresh on a
        ``401``. So on :class:`NotAuthorized` we force exactly one fresh login
        and retry the bootstrap once.
        """

        try:
            await api.update()
            return
        except NotAuthorized:
            if not hasattr(api, "authenticate"):
                raise
            LOGGER.warning(
                "Protect private session rejected (401); forcing a fresh login and retrying once.",
            )
        await api.authenticate()
        await api.update()

    async def close(self) -> None:
        """Close websocket and HTTP sessions owned by uiprotect."""

        api = self._api
        try:
            if api is None:
                return
            if hasattr(api, "async_disconnect_ws"):
                await api.async_disconnect_ws()
            if hasattr(api, "close_session"):
                await api.close_session()
            if hasattr(api, "close_public_api_session"):
                await api.close_public_api_session()
        finally:
            self._connected = False
            self._private_enabled = False
            self._public_enabled = False
            self._public_rtsps_streams.clear()

    async def list_cameras(self) -> list[ProtectCameraInfo]:
        """Return connected/capability/channel metadata without raw RTSP secrets."""

        await self.connect()
        cameras = self._bootstrap_cameras(private=True)
        if cameras:
            return [self._camera_info(camera) for camera in cameras]

        public_cameras = self._bootstrap_cameras(private=False)
        if not public_cameras and hasattr(self.api, "get_cameras_public"):
            public_cameras = await self.api.get_cameras_public()
        return [await self._public_camera_info(camera) for camera in public_cameras]

    def rtsps_url(self, camera_id: str, substream: Substream) -> str | None:
        """Return a raw RTSPS URL for a channel, or ``None`` if RTSP is disabled.

        Protect private camera channels are ordered highest-to-lowest quality.
        Mapping: ``high`` -> first channel, ``low`` -> last channel, and
        ``medium`` -> the middle channel (lower channel when only two exist).
        The returned URL can contain a Protect token; sanitize it before storing
        or logging.
        """

        camera = self._get_bootstrap_camera(camera_id)
        if camera is not None:
            channel = _select_channel(getattr(camera, "channels", ()), substream)
            if channel is None or not bool(getattr(channel, "is_rtsp_enabled", False)):
                return None
            return getattr(channel, "rtsps_url", None)

        streams = self._get_public_rtsps_streams_sync(camera_id)
        if streams is None:
            return None
        url = _get_stream_quality(streams, substream)
        return url if isinstance(url, str) and url else None

    async def snapshot(self, camera_id: str) -> bytes:
        """Return a Protect snapshot. Protect API snapshots are usually ~640x360."""

        await self.connect()
        if self._private_enabled and hasattr(self.api, "get_camera_snapshot"):
            data = await self.api.get_camera_snapshot(camera_id)
        elif hasattr(self.api, "get_public_api_camera_snapshot"):
            data = await self.api.get_public_api_camera_snapshot(camera_id)
        else:
            data = None
        if data is None:
            raise RuntimeError(f"Protect snapshot unavailable for camera {camera_id!r}")
        return data

    async def download_recording(
        self,
        camera_id: str,
        start: datetime,
        end: datetime,
        dest: Path,
    ) -> Path | None:
        """Download a continuous-recording MP4 window via ``get_camera_video``.

        ``uiprotect.ProtectApiClient.get_camera_video(..., channel_index=0,
        output_file=dest)`` exports Protect's recording for the requested time
        range. Channel 0 is Protect's high-quality recording channel.
        """

        await self.connect()
        if not self._private_enabled or not hasattr(self.api, "get_camera_video"):
            return None

        dest.parent.mkdir(parents=True, exist_ok=True)
        await self.api.get_camera_video(
            camera_id,
            start,
            end,
            channel_index=0,
            output_file=dest,
        )
        return dest if dest.exists() else None

    def subscribe_websocket(self, callback: Callable[[Any], None]) -> Callable[[], None]:
        if self._private_enabled and hasattr(self.api, "subscribe_websocket"):
            return self.api.subscribe_websocket(callback)
        if hasattr(self.api, "subscribe_events_websocket"):
            return self.api.subscribe_events_websocket(callback)
        raise RuntimeError("uiprotect websocket subscription is unavailable")

    def subscribe_websocket_state(self, callback: Callable[[Any], None]) -> Callable[[], None]:
        if self._private_enabled and hasattr(self.api, "subscribe_websocket_state"):
            return self.api.subscribe_websocket_state(callback)
        if hasattr(self.api, "subscribe_events_websocket_state"):
            return self.api.subscribe_events_websocket_state(callback)
        return _noop

    def _build_api_client(self) -> ProtectApiClient:
        host, port = _parse_host_port(self.config.protect.nvr_host)
        api_key = self.config.resolve_secret("api_key")
        username = self.config.resolve_secret("username")
        password = self.config.resolve_secret("password")
        if not api_key and not (username and password):
            raise ValueError("Protect credentials must be supplied via configured env vars")
        return ProtectApiClient(
            host=host,
            port=port,
            username=username,
            password=password,
            api_key=api_key,
            verify_ssl=self.config.protect.verify_tls,
            ws_timeout=self._ws_timeout,
        )

    def _is_injected_client(self) -> bool:
        return self._api is not None and not (
            self.config.resolve_secret("api_key")
            or self.config.resolve_secret("username")
            or self.config.resolve_secret("password")
        )

    def _bootstrap_cameras(self, *, private: bool) -> list[Any]:
        # uiprotect's private ``bootstrap`` / ``public_bootstrap`` properties raise
        # when their respective ``update``/``update_public`` was never called, so only
        # touch the bootstrap that was actually initialized in ``connect``.
        if private and not self._private_enabled:
            return []
        if not private and not self._public_enabled:
            return []
        bootstrap_name = "bootstrap" if private else "public_bootstrap"
        bootstrap = getattr(self.api, bootstrap_name, None)
        cameras = getattr(bootstrap, "cameras", None)
        if cameras is None:
            return []
        if isinstance(cameras, dict):
            return list(cameras.values())
        if hasattr(cameras, "values"):
            return list(cameras.values())
        return list(cameras)

    def _get_bootstrap_camera(self, camera_id: str) -> Any | None:
        if not self._private_enabled:
            return None
        for camera in self._bootstrap_cameras(private=True):
            if str(getattr(camera, "id", "")) == camera_id:
                return camera
        return None

    def _camera_info(self, camera: Any) -> ProtectCameraInfo:
        channels = tuple(
            _channel_info(index, channel)
            for index, channel in enumerate(getattr(camera, "channels", ()) or ())
        )
        return ProtectCameraInfo(
            id=str(getattr(camera, "id")),
            name=str(getattr(camera, "name", None) or getattr(camera, "id")),
            is_connected=bool(getattr(camera, "is_connected", False)),
            animal_smart_detect_supported=_has_animal_smart_detect(camera),
            channels=channels,
        )

    async def _public_camera_info(self, camera: Any) -> ProtectCameraInfo:
        camera_id = str(getattr(camera, "id"))
        channels: tuple[ProtectCameraChannel, ...] = ()
        streams = await self._get_public_rtsps_streams(camera_id)
        if streams is not None:
            channels = tuple(_public_channel_info(i, key, streams) for i, key in enumerate(_stream_keys(streams)))
        return ProtectCameraInfo(
            id=camera_id,
            name=str(getattr(camera, "name", None) or camera_id),
            is_connected=_is_public_camera_connected(camera),
            animal_smart_detect_supported=_has_animal_smart_detect(camera),
            channels=channels,
        )

    async def _cache_public_rtsps_streams(self) -> None:
        if not hasattr(self.api, "get_camera_rtsps_streams"):
            return
        for camera in self._bootstrap_cameras(private=False):
            camera_id = str(getattr(camera, "id"))
            try:
                streams = await self.api.get_camera_rtsps_streams(camera_id)
            except Exception:
                continue
            if streams is not None:
                self._public_rtsps_streams[camera_id] = streams

    async def _get_public_rtsps_streams(self, camera_id: str) -> Any | None:
        if camera_id in self._public_rtsps_streams:
            return self._public_rtsps_streams[camera_id]
        if not hasattr(self.api, "get_camera_rtsps_streams"):
            return None
        streams = await self.api.get_camera_rtsps_streams(camera_id)
        if streams is not None:
            self._public_rtsps_streams[camera_id] = streams
        return streams

    def _get_public_rtsps_streams_sync(self, camera_id: str) -> Any | None:
        if not self._public_enabled:
            return None
        if camera_id in self._public_rtsps_streams:
            return self._public_rtsps_streams[camera_id]
        bootstrap = getattr(self.api, "public_bootstrap", None)
        streams_by_camera = getattr(bootstrap, "rtsps_streams", None)
        if isinstance(streams_by_camera, dict):
            return streams_by_camera.get(camera_id)
        return None


def _noop() -> None:
    return None


def _parse_host_port(value: str | None) -> tuple[str, int]:
    if not value:
        raise ValueError("protect.nvr_host is required")
    parsed = urlsplit(value if "://" in value else f"https://{value}")
    if not parsed.hostname:
        raise ValueError("protect.nvr_host must include a hostname")
    default_port = 80 if parsed.scheme == "http" else 443
    return parsed.hostname, parsed.port or default_port


def _select_channel(channels: Any, substream: Substream) -> Any | None:
    ordered = list(channels or ())
    if not ordered:
        return None
    if substream == "high":
        return ordered[0]
    if substream == "low":
        return ordered[-1]
    return ordered[len(ordered) // 2]


def _channel_info(index: int, channel: Any) -> ProtectCameraChannel:
    raw_url = getattr(channel, "rtsps_url", None)
    return ProtectCameraChannel(
        index=index,
        id=getattr(channel, "id", None),
        name=str(getattr(channel, "name", None) or index),
        enabled=bool(getattr(channel, "enabled", True)),
        is_rtsp_enabled=bool(getattr(channel, "is_rtsp_enabled", False)),
        width=getattr(channel, "width", None),
        height=getattr(channel, "height", None),
        fps=getattr(channel, "fps", None),
        sanitized_rtsps_url=sanitize_source_id(raw_url) if isinstance(raw_url, str) else None,
    )


def _public_channel_info(index: int, quality: str, streams: Any) -> ProtectCameraChannel:
    raw_url = _get_stream_quality(streams, quality)
    return ProtectCameraChannel(
        index=index,
        id=quality,
        name=quality,
        enabled=True,
        is_rtsp_enabled=isinstance(raw_url, str) and bool(raw_url),
        sanitized_rtsps_url=sanitize_source_id(raw_url) if isinstance(raw_url, str) else None,
    )


def _has_animal_smart_detect(camera: Any) -> bool:
    feature_flags = getattr(camera, "feature_flags", None)
    raw_types = getattr(feature_flags, "smart_detect_types", ()) or ()
    return "animal" in {_string_value(item).lower() for item in raw_types}


def _is_public_camera_connected(camera: Any) -> bool:
    state = getattr(camera, "state", None)
    value = _string_value(state).upper()
    return value == "CONNECTED"


def _stream_keys(streams: Any) -> list[str]:
    if hasattr(streams, "get_available_stream_qualities"):
        return list(streams.get_available_stream_qualities())
    extra = getattr(streams, "__pydantic_extra__", None)
    if isinstance(extra, dict):
        return list(extra.keys())
    if isinstance(streams, dict):
        return list(streams.keys())
    return [key for key in ("high", "medium", "low") if getattr(streams, key, None)]


def _get_stream_quality(streams: Any, quality: str) -> Any | None:
    if hasattr(streams, "get_stream_url"):
        return streams.get_stream_url(quality)
    if isinstance(streams, dict):
        return streams.get(quality)
    return getattr(streams, quality, None)


def _string_value(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return str(value)
