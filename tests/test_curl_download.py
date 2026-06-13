"""Offline tests for the curl-based Protect downloader (``protect.curl_download``).

No curl binary, NVR, or network: the single curl invocation is replaced by an
injected ``FakeCurlRunner`` that writes the files curl would have written (cookie
jar, ``-D`` header dump, ``-o`` body) and returns a ``CompletedProcess`` whose
stdout ends in the HTTP status sentinel (``-w %{http_code}``). This mirrors the
``download_fn`` seam used by ``test_harvest_unvr``.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pytest

from detectivepotty.protect.curl_download import (
    LOGIN_PATH,
    PRIVATE_API_PATH,
    CurlProtectDownloader,
    CurlProtectError,
    build_export_url,
    curl_available,
    js_time,
    match_camera_id,
    normalize_base_url,
    parse_csrf_header,
)

UTC = timezone.utc


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #


def test_curl_available_false_for_missing_binary() -> None:
    assert curl_available("definitely-not-a-real-binary-xyz") is False


def test_js_time_naive_is_utc_and_milliseconds() -> None:
    aware = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    naive = datetime(2026, 6, 6, 0, 0)
    assert js_time(aware) == int(aware.timestamp() * 1000)
    # A naive datetime is interpreted as UTC, so it matches the aware value.
    assert js_time(naive) == js_time(aware)
    # Milliseconds, not seconds.
    assert js_time(aware) % 1000 == 0
    assert js_time(aware) == aware.timestamp() * 1000


def test_normalize_base_url_variants() -> None:
    assert normalize_base_url("nvr.bennet.lan") == "https://nvr.bennet.lan"
    assert normalize_base_url("https://nvr.bennet.lan/") == "https://nvr.bennet.lan"
    assert normalize_base_url("https://nvr.bennet.lan/proxy/x") == "https://nvr.bennet.lan"
    assert normalize_base_url("http://10.0.0.5:7443") == "http://10.0.0.5:7443"


def test_normalize_base_url_requires_hostname() -> None:
    with pytest.raises(ValueError):
        normalize_base_url("https://")


def test_build_export_url_default_channel() -> None:
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = datetime(2026, 6, 6, 0, 2, tzinfo=UTC)
    url = build_export_url("https://nvr.lan", "cam-1", start, end)
    assert url.startswith(f"https://nvr.lan{PRIVATE_API_PATH}video/export?")
    assert "camera=cam-1" in url
    assert f"start={js_time(start)}" in url
    assert f"end={js_time(end)}" in url
    assert "channel=0" in url
    assert "lens=" not in url


def test_build_export_url_package_channel_uses_lens() -> None:
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = datetime(2026, 6, 6, 0, 2, tzinfo=UTC)
    url = build_export_url("https://nvr.lan", "cam-1", start, end, channel_index=3)
    assert "lens=2" in url
    assert "channel=" not in url


def test_parse_csrf_header_case_insensitive_last_wins() -> None:
    headers = (
        "HTTP/2 200\r\n"
        "X-CSRF-Token: first-token\r\n"
        "content-type: application/json\r\n"
        "x-csrf-token: final-token\r\n"
    )
    assert parse_csrf_header(headers) == "final-token"


def test_parse_csrf_header_absent_returns_none() -> None:
    assert parse_csrf_header("HTTP/2 200\r\ncontent-type: application/json\r\n") is None


def test_match_camera_id_by_id_then_name() -> None:
    cameras = [
        {"id": "abc123", "name": "Backyard Grass"},
        {"id": "def456", "name": "Front Yard"},
    ]
    assert match_camera_id(cameras, "def456") == "def456"  # exact id
    assert match_camera_id(cameras, "backyard grass") == "abc123"  # case-insensitive name
    assert match_camera_id(cameras, "  Front Yard  ") == "def456"  # trimmed name
    assert match_camera_id(cameras, "nope") is None


# --------------------------------------------------------------------------- #
# Fake curl runner
# --------------------------------------------------------------------------- #


class FakeCurlRunner:
    """Stand-in for the curl subprocess; writes files + returns an HTTP code.

    curl is invoked as ``curl -w %{http_code} ... <url>`` with the body going to a
    ``-o`` file (or ``/dev/null``) and login headers to a ``-D`` file. The runner
    reproduces those side effects so the downloader's parsing exercises real I/O.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bytes | None]] = []
        self.csrf = "csrf-token-xyz"
        self.cameras = [
            {"id": "cam-1", "name": "Backyard Grass", "state": "CONNECTED"},
            {"id": "cam-2", "name": "Front Yard", "state": "CONNECTED"},
        ]
        self.export_body = b"\x00\x00\x00\x18ftypmp42" + b"fake-mp4-payload"
        self.login_code = 200
        self.bootstrap_code = 200
        self.export_code = 200
        self.expire_session_once = False  # first authed GET returns 401, then ok
        self.proc_returncode = 0
        self.export_returncode = 0
        self.login_count = 0
        self._expired = False

    @staticmethod
    def _opt(args: Sequence[str], flag: str) -> str | None:
        args = list(args)
        if flag in args:
            return args[args.index(flag) + 1]
        return None

    def __call__(
        self, args: Sequence[str], input_bytes: bytes | None
    ) -> subprocess.CompletedProcess:
        args = list(args)
        self.calls.append((args, input_bytes))
        url = args[-1]
        out = self._opt(args, "-o")

        if self.proc_returncode != 0:
            return subprocess.CompletedProcess(args, self.proc_returncode, b"", b"curl boom")

        if url.endswith(LOGIN_PATH):
            self.login_count += 1
            if self.login_code == 200:
                hdr = self._opt(args, "-D")
                if hdr and hdr != "/dev/null":
                    Path(hdr).write_text(
                        f"HTTP/2 200\r\nx-csrf-token: {self.csrf}\r\n\r\n"
                    )
                jar = self._opt(args, "-c")
                if jar:
                    Path(jar).write_text("# Netscape HTTP Cookie File\n")
            return self._done(args, self.login_code)

        # Authed GETs (bootstrap / video/export) honour an optional one-shot 401.
        if self.expire_session_once and not self._expired:
            self._expired = True
            return self._done(args, 401)

        if url.endswith("bootstrap"):
            if self.bootstrap_code == 200 and out and out != "/dev/null":
                Path(out).write_text(json.dumps({"cameras": self.cameras}))
            return self._done(args, self.bootstrap_code)

        if "video/export" in url:
            if self.export_code == 200 and out and out != "/dev/null":
                Path(out).write_bytes(self.export_body)
            if self.export_returncode != 0:
                return subprocess.CompletedProcess(
                    args,
                    self.export_returncode,
                    str(self.export_code).encode(),
                    b"curl boom",
                )
            return self._done(args, self.export_code)

        return self._done(args, 404)

    @staticmethod
    def _done(args: list[str], code: int) -> subprocess.CompletedProcess:
        # Body went to the -o file; stdout carries only the -w http-code sentinel.
        return subprocess.CompletedProcess(args, 0, str(code).encode(), b"")


def _downloader(runner: FakeCurlRunner, **kw) -> CurlProtectDownloader:
    return CurlProtectDownloader(
        "https://nvr.bennet.lan",
        "admin",
        "s3cr3t-pass",
        verify_tls=False,
        runner=runner,
        **kw,
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def test_login_caches_csrf_and_keeps_password_off_argv() -> None:
    runner = FakeCurlRunner()
    with _downloader(runner) as dl:
        dl.login()
        assert dl._csrf == runner.csrf
    # Password is delivered on stdin (curl --data @-), never as a command argument.
    login_args, login_input = runner.calls[0]
    assert b"s3cr3t-pass" in (login_input or b"")
    assert not any("s3cr3t-pass" in a for a in login_args)


def test_login_failure_raises() -> None:
    runner = FakeCurlRunner()
    runner.login_code = 401
    with _downloader(runner) as dl:
        with pytest.raises(CurlProtectError, match="login failed"):
            dl.login()


def test_list_cameras_returns_bootstrap_entries() -> None:
    runner = FakeCurlRunner()
    with _downloader(runner) as dl:
        cameras = dl.list_cameras()
    assert [c["id"] for c in cameras] == ["cam-1", "cam-2"]


def test_list_cameras_bootstrap_error_raises() -> None:
    runner = FakeCurlRunner()
    runner.bootstrap_code = 500
    with _downloader(runner) as dl:
        with pytest.raises(CurlProtectError, match="bootstrap failed"):
            dl.list_cameras()


def test_resolve_camera_id_by_name_and_missing() -> None:
    runner = FakeCurlRunner()
    with _downloader(runner) as dl:
        assert dl.resolve_camera_id("Backyard Grass") == "cam-1"
        with pytest.raises(CurlProtectError, match="camera not found"):
            dl.resolve_camera_id("Garage")


def test_download_writes_dest(tmp_path: Path) -> None:
    runner = FakeCurlRunner()
    dest = tmp_path / "nested" / "clip.mp4"
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = datetime(2026, 6, 6, 0, 2, tzinfo=UTC)
    with _downloader(runner) as dl:
        out = dl.download("cam-1", start, end, dest)
    assert out == dest
    assert dest.read_bytes() == runner.export_body
    export_args = [args for args, _input in runner.calls if "video/export" in args[-1]][0]
    assert export_args[export_args.index("-o") + 1] == str(dest.with_name("clip.mp4.part"))
    assert not dest.with_name("clip.mp4.part").exists()


def test_download_reauths_on_401(tmp_path: Path) -> None:
    runner = FakeCurlRunner()
    runner.expire_session_once = True
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = datetime(2026, 6, 6, 0, 2, tzinfo=UTC)
    with _downloader(runner) as dl:
        out = dl.download("cam-1", start, end, tmp_path / "clip.mp4")
    assert out is not None
    # Logged in twice: the initial login + the re-login after the 401.
    assert runner.login_count == 2


def test_download_non_200_returns_none_and_removes_dest(tmp_path: Path) -> None:
    runner = FakeCurlRunner()
    runner.export_code = 404
    dest = tmp_path / "clip.mp4"
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = datetime(2026, 6, 6, 0, 2, tzinfo=UTC)
    with _downloader(runner) as dl:
        out = dl.download("cam-1", start, end, dest)
    assert out is None
    assert not dest.exists()


def test_download_empty_body_returns_none(tmp_path: Path) -> None:
    runner = FakeCurlRunner()
    runner.export_body = b""
    dest = tmp_path / "clip.mp4"
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = datetime(2026, 6, 6, 0, 2, tzinfo=UTC)
    with _downloader(runner) as dl:
        out = dl.download("cam-1", start, end, dest)
    assert out is None
    assert not dest.exists()


def test_download_truncated_export_raises_and_removes_part(tmp_path: Path) -> None:
    runner = FakeCurlRunner()
    runner.export_returncode = 18
    dest = tmp_path / "clip.mp4"
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = datetime(2026, 6, 6, 0, 2, tzinfo=UTC)

    with _downloader(runner) as dl:
        with pytest.raises(CurlProtectError, match=r"curl failed \(exit 18\).*HTTP 200"):
            dl.download("cam-1", start, end, dest)

    assert not dest.exists()
    assert not dest.with_name("clip.mp4.part").exists()


def test_as_download_fn_matches_orchestrator_seam(tmp_path: Path) -> None:
    runner = FakeCurlRunner()
    start = datetime(2026, 6, 6, 0, 0, tzinfo=UTC)
    end = datetime(2026, 6, 6, 0, 2, tzinfo=UTC)
    with _downloader(runner) as dl:
        fn = dl.as_download_fn()
        out = fn("cam-1", start, end, tmp_path / "clip.mp4")
    assert out is not None and out.exists()


def test_curl_process_failure_raises(tmp_path: Path) -> None:
    runner = FakeCurlRunner()
    runner.proc_returncode = 7  # curl: couldn't connect
    with _downloader(runner) as dl:
        with pytest.raises(CurlProtectError, match="curl failed"):
            dl.login()


def test_context_manager_cleans_workdir() -> None:
    runner = FakeCurlRunner()
    dl = _downloader(runner)
    workdir = dl._workdir
    assert workdir.exists()
    with dl:
        dl.login()
    assert not workdir.exists()
