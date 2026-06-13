"""curl-based UniFi Protect recording downloader (LAN-access fallback).

Why this exists
---------------
The normal download path uses ``uiprotect.ProtectApiClient`` (aiohttp) in-process.
On macOS, **Local Network Privacy** denies the uv-managed Python interpreter access
to *peer* LAN devices (the NVR), so every aiohttp call fails with
``[Errno 65] No route to host`` — while the Apple-signed ``curl`` binary and all WAN
traffic are unaffected. This module shells out to ``curl`` to perform exactly what
uiprotect does over the private API:

1. ``POST /api/auth/login`` with ``{username, password}`` → a session cookie
   (``TOKEN``/``UOS_TOKEN``) plus an ``x-csrf-token`` response header.
2. ``GET /proxy/protect/api/bootstrap`` → the camera list (for name→id resolution).
3. ``GET /proxy/protect/api/video/export?camera=&start=&end=&channel=0`` → streams
   the recording MP4 for the requested absolute-time window.

Only the *network* hop is delegated to curl; detection and clip-cutting stay
in-process and fully offline. The single curl invocation is injectable (``runner``)
so the orchestration is unit-testable without a real binary or network.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode, urlsplit

PRIVATE_API_PATH = "/proxy/protect/api/"
LOGIN_PATH = "/api/auth/login"

CurlRunner = Callable[[Sequence[str], bytes | None], subprocess.CompletedProcess]


def curl_available(curl_bin: str = "curl") -> bool:
    """Return True if the ``curl`` binary is on PATH."""

    return shutil.which(curl_bin) is not None


def js_time(value: datetime) -> int:
    """Convert a datetime to UniFi's JavaScript epoch-milliseconds timestamp."""

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp() * 1000)


def normalize_base_url(nvr_host: str) -> str:
    """Return ``scheme://host[:port]`` with no trailing slash or path."""

    parsed = urlsplit(nvr_host if "://" in nvr_host else f"https://{nvr_host}")
    if not parsed.hostname:
        raise ValueError("nvr_host must include a hostname")
    scheme = parsed.scheme or "https"
    netloc = parsed.hostname
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return f"{scheme}://{netloc}"


def build_export_url(
    base_url: str,
    camera_id: str,
    start: datetime,
    end: datetime,
    channel_index: int = 0,
) -> str:
    """Build the private-API ``video/export`` URL for an absolute-time window."""

    params = {
        "camera": camera_id,
        "start": js_time(start),
        "end": js_time(end),
    }
    if channel_index == 3:
        params["lens"] = 2
    else:
        params["channel"] = channel_index
    return f"{base_url}{PRIVATE_API_PATH}video/export?{urlencode(params)}"


def parse_csrf_header(header_text: str) -> str | None:
    """Extract the ``x-csrf-token`` value from raw HTTP response headers."""

    token: str | None = None
    for line in header_text.splitlines():
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        if name.strip().lower() == "x-csrf-token":
            token = value.strip()
    return token


def match_camera_id(cameras: Sequence[dict], camera: str) -> str | None:
    """Resolve ``camera`` against camera ids, then names (case-insensitive)."""

    for cam in cameras:
        if cam.get("id") == camera:
            return camera
    lowered = camera.strip().lower()
    for cam in cameras:
        if str(cam.get("name", "")).strip().lower() == lowered:
            return cam.get("id")
    return None


@dataclass
class _CurlResult:
    returncode: int
    http_code: int
    stdout: bytes
    stderr: str


class CurlProtectError(RuntimeError):
    """Raised when a curl-backed Protect request fails."""


class CurlProtectDownloader:
    """Download Protect recordings via the ``curl`` binary.

    Logs in once, caches the session cookie + CSRF token in a private temp dir, and
    reuses them across chunk downloads (re-authenticating once on a 401). Use as a
    context manager so the temp dir is cleaned up.
    """

    def __init__(
        self,
        nvr_host: str,
        username: str,
        password: str,
        *,
        verify_tls: bool = True,
        curl_bin: str = "curl",
        connect_timeout: float = 10.0,
        max_time: float = 600.0,
        runner: CurlRunner | None = None,
    ) -> None:
        self.base_url = normalize_base_url(nvr_host)
        self._username = username
        self._password = password
        self._verify_tls = verify_tls
        self._curl_bin = curl_bin
        self._connect_timeout = connect_timeout
        self._max_time = max_time
        self._runner = runner or self._default_runner
        self._workdir = Path(tempfile.mkdtemp(prefix="dp-curl-protect-"))
        self._workdir.chmod(0o700)
        self._cookie_jar = self._workdir / "cookies.txt"
        self._csrf: str | None = None

    # -- context manager -------------------------------------------------
    def __enter__(self) -> CurlProtectDownloader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        shutil.rmtree(self._workdir, ignore_errors=True)

    # -- public API ------------------------------------------------------
    def login(self) -> None:
        """Authenticate and cache the session cookie + CSRF token."""

        header_file = self._workdir / "login_headers.txt"
        payload = json.dumps(
            {
                "username": self._username,
                "password": self._password,
                "rememberMe": False,
            }
        ).encode()
        args = [
            *self._base_args(),
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "--data",
            "@-",
            "-c",
            str(self._cookie_jar),
            "-D",
            str(header_file),
            "-o",
            "/dev/null",
            f"{self.base_url}{LOGIN_PATH}",
        ]
        result = self._invoke(args, input_bytes=payload)
        if result.http_code != 200:
            raise CurlProtectError(
                f"Protect login failed (HTTP {result.http_code}). Check credentials."
            )
        self._csrf = parse_csrf_header(_read_text(header_file))

    def list_cameras(self) -> list[dict]:
        """Return the bootstrap camera list (``id``/``name``/``state``/...)."""

        if self._csrf is None:
            self.login()
        target = self._workdir / "bootstrap.json"
        result = self._authed_get(f"{self.base_url}{PRIVATE_API_PATH}bootstrap", target)
        if result.http_code == 401:
            self.login()
            result = self._authed_get(
                f"{self.base_url}{PRIVATE_API_PATH}bootstrap", target
            )
        if result.http_code != 200:
            raise CurlProtectError(
                f"Protect bootstrap failed (HTTP {result.http_code})."
            )
        data = json.loads(_read_text(target) or "{}")
        return list(data.get("cameras", []))

    def resolve_camera_id(self, camera: str) -> str:
        """Resolve a camera id or name to an id, raising if not found."""

        cameras = self.list_cameras()
        camera_id = match_camera_id(cameras, camera)
        if camera_id is None:
            raise CurlProtectError(
                f"camera not found: {camera!r}. Run 'list-cameras' to see ids/names."
            )
        return camera_id

    def download(
        self,
        camera_id: str,
        start: datetime,
        end: datetime,
        dest: Path,
        channel_index: int = 0,
    ) -> Path | None:
        """Export ``[start, end)`` for ``camera_id`` to ``dest``; None if empty."""

        if self._csrf is None:
            self.login()
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = build_export_url(self.base_url, camera_id, start, end, channel_index)
        part = dest.with_name(f"{dest.name}.part")
        part.unlink(missing_ok=True)
        try:
            result = self._authed_get(url, part)
            if result.http_code == 401:
                part.unlink(missing_ok=True)
                self.login()
                result = self._authed_get(url, part)
            if result.http_code != 200:
                part.unlink(missing_ok=True)
                dest.unlink(missing_ok=True)
                return None
            if not part.exists() or part.stat().st_size == 0:
                part.unlink(missing_ok=True)
                dest.unlink(missing_ok=True)
                return None
            part.replace(dest)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        return dest

    def as_download_fn(self) -> Callable[[str, datetime, datetime, Path], Path | None]:
        """Return a ``download_fn`` matching the harvest orchestrator seam."""

        def _download_fn(
            camera_id: str, start: datetime, end: datetime, dest: Path
        ) -> Path | None:
            return self.download(camera_id, start, end, dest)

        return _download_fn

    # -- internals -------------------------------------------------------
    def _base_args(self) -> list[str]:
        args = [
            self._curl_bin,
            "-sS",
            "--connect-timeout",
            str(self._connect_timeout),
            "--max-time",
            str(self._max_time),
        ]
        if not self._verify_tls:
            args.append("-k")
        return args

    def _authed_get(self, url: str, dest: Path) -> _CurlResult:
        args = [
            *self._base_args(),
            "-b",
            str(self._cookie_jar),
            "-H",
            f"x-csrf-token: {self._csrf or ''}",
            "-o",
            str(dest),
            url,
        ]
        return self._invoke(args)

    def _invoke(self, args: list[str], input_bytes: bytes | None = None) -> _CurlResult:
        # Append an http-code sentinel so we can read the status regardless of -o.
        args = [*args[:1], "-w", "%{http_code}", *args[1:]]
        completed = self._runner(args, input_bytes)
        stdout = completed.stdout or b""
        http_code = _trailing_http_code(stdout)
        if completed.returncode != 0:
            stderr = _decode(getattr(completed, "stderr", b""))
            suffix = f" (HTTP {http_code})" if http_code else ""
            raise CurlProtectError(
                f"curl failed (exit {completed.returncode}){suffix}: {stderr.strip()}"
            )
        return _CurlResult(
            returncode=completed.returncode,
            http_code=http_code,
            stdout=stdout,
            stderr=_decode(getattr(completed, "stderr", b"")),
        )

    def _default_runner(
        self, args: Sequence[str], input_bytes: bytes | None
    ) -> subprocess.CompletedProcess:
        return subprocess.run(  # noqa: S603 - args are constructed internally
            list(args),
            input=input_bytes,
            capture_output=True,
            check=False,
        )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except FileNotFoundError:
        return ""


def _decode(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value or "")


def _trailing_http_code(stdout: bytes) -> int:
    text = stdout.decode(errors="replace").strip()
    if not text:
        return 0
    tail = text[-3:]
    if tail.isdigit():
        return int(tail)
    return 0
