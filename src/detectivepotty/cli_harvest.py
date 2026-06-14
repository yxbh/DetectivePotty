"""Harvest and acquisition CLI commands."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Optional

import typer

from detectivepotty.cli_common import (
    ConfigPathOption,
    DogAliasOption,
    load_cli_config,
    protect_download_result,
    resolve_dog_aliases,
)

_protect_download_result: Callable[[object], object] = protect_download_result


def register_harvest_commands(
    app: typer.Typer,
    protect_download_result_fn: Callable[[object], object] = protect_download_result,
) -> None:
    global _protect_download_result
    _protect_download_result = protect_download_result_fn
    app.command("harvest")(harvest_command)
    app.command("harvest-camera")(harvest_camera_command)
    app.command("experiment-acquire")(experiment_acquire_command)


def harvest_command(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Long recording to scan for dog-present spans.",
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option("--out", help="Directory to write harvested clip dirs into."),
    ] = Path("dataset/harvest"),
    model: Annotated[
        str,
        typer.Option("--model", help="YOLO model name/path."),
    ] = "models/yolo11m.pt",
    long_edge: Annotated[
        int,
        typer.Option("--long-edge", min=1, help="YOLO inference long edge (imgsz)."),
    ] = 640,
    conf: Annotated[
        float,
        typer.Option("--conf", min=0.0, max=1.0, help="Detection confidence threshold."),
    ] = 0.25,
    camera_name: Annotated[
        Optional[str],
        typer.Option(
            "--camera-name",
            help="Friendly camera name recorded in metadata for the labeling UI.",
        ),
    ] = None,
    sample_every: Annotated[
        int,
        typer.Option("--sample-every", min=1, help="Run detection every N frames."),
    ] = 5,
    detect_batch_size: Annotated[
        int,
        typer.Option(
            "--detect-batch-size",
            min=1,
            help="Detect this many sampled frames per batched forward (faster on "
            "accelerated backends; CoreML/MPS true-batch here). 1 = single-frame.",
        ),
    ] = 32,
    max_age_frames: Annotated[
        int,
        typer.Option(
            "--max-age-frames",
            min=0,
            help="Keep a track alive across this many missed source frames "
            "(survives brief detector misses; ~3 missed samples at the default "
            "stride). Higher = fewer fragmented spans.",
        ),
    ] = 15,
    center_dist_gate: Annotated[
        float,
        typer.Option(
            "--center-dist-gate",
            min=0.0,
            help="Re-associate a detection to a track when its box center is within "
            "this many box-diagonals, even if IoU is low (0 disables; handles a dog "
            "that moved between sparse samples). Higher = stickier tracks.",
        ),
    ] = 1.5,
    merge_gap_s: Annotated[
        float,
        typer.Option("--merge-gap", min=0.0, help="Merge same-track gaps up to N seconds."),
    ] = 2.0,
    pad_s: Annotated[
        float,
        typer.Option("--pad", min=0.0, help="Seconds of padding added to each span."),
    ] = 1.0,
    min_len_s: Annotated[
        float,
        typer.Option("--min-len", min=0.0, help="Drop spans shorter than N seconds."),
    ] = 0.5,
    max_len_s: Annotated[
        float,
        typer.Option("--max-len", min=0.0, help="Split spans longer than N seconds."),
    ] = 60.0,
    dog_alias_classes: DogAliasOption = None,
) -> None:
    """Cut dog-present spans out of a long recording into reviewable clip dirs.

    Each span becomes ``<out>/<span_id>/`` with an immutable ``clip.mp4`` plus a
    ``metadata.json`` (fps, frame count, checksum, source UTC range, per-sampled
    detection boxes). Re-running on an unchanged clip is idempotent. Hand-write a
    ``labels.json`` next to a clip, then ``export-dataset`` to build crops.
    """

    from detectivepotty.detect.yolo import DogDetector
    from detectivepotty.harvest import harvest_clips

    alias_classes, alias_nms_iou = resolve_dog_aliases(dog_alias_classes)
    detector = DogDetector(
        model_name=model,
        long_edge=long_edge,
        conf_threshold=conf,
        device="auto",
        alias_classes=alias_classes,
        alias_nms_iou=alias_nms_iou,
    )
    results = harvest_clips(
        input_path,
        out_dir,
        detector=detector,
        sample_every=sample_every,
        merge_gap_s=merge_gap_s,
        pad_s=pad_s,
        min_len_s=min_len_s,
        max_len_s=max_len_s,
        max_age_frames=max_age_frames,
        center_dist_gate=center_dist_gate,
        detect_batch_size=detect_batch_size,
        camera_name=camera_name,
        detect_conf=conf,
    )
    if not results:
        typer.echo("No dog spans found.")
        return
    typer.echo(f"Harvested {len(results)} span(s) into {out_dir}:")
    for result in results:
        span = result.span
        typer.echo(
            f"  {result.span_id}  frames {span.start_frame}-{span.end_frame} "
            f"({span.start_s:.1f}-{span.end_s:.1f}s, track {span.track_id})"
        )


def harvest_camera_command(
    camera: Annotated[
        str,
        typer.Option("--camera", help="Protect camera id or name (see list-cameras)."),
    ],
    config_path: ConfigPathOption = None,
    date: Annotated[
        str | None,
        typer.Option(
            "--date",
            help="Day to harvest as YYYY-MM-DD (a 24h window at --utc-offset).",
        ),
    ] = None,
    start: Annotated[
        str | None,
        typer.Option("--start", help="ISO-8601 start (overrides --date)."),
    ] = None,
    end: Annotated[
        str | None,
        typer.Option("--end", help="ISO-8601 end (overrides --date)."),
    ] = None,
    utc_offset: Annotated[
        float,
        typer.Option(
            "--utc-offset",
            help="Hours offset from UTC for --date (e.g. 10 for AEST). Default 0.",
        ),
    ] = 0.0,
    out_dir: Annotated[
        Path,
        typer.Option("--out", help="Directory to write harvested clip dirs into."),
    ] = Path("dataset/harvest"),
    model: Annotated[
        str,
        typer.Option("--model", help="YOLO model name/path."),
    ] = "models/yolo11m.pt",
    long_edge: Annotated[
        int,
        typer.Option("--long-edge", min=1, help="YOLO inference long edge (imgsz)."),
    ] = 640,
    conf: Annotated[
        float,
        typer.Option("--conf", min=0.0, max=1.0, help="Detection confidence threshold."),
    ] = 0.25,
    chunk_s: Annotated[
        float,
        typer.Option("--chunk", min=1.0, help="Download/process chunk length (seconds)."),
    ] = 3600.0,
    overlap_s: Annotated[
        float,
        typer.Option("--overlap", min=0.0, help="Overlap between chunks (seconds)."),
    ] = 5.0,
    sample_every: Annotated[
        int,
        typer.Option("--sample-every", min=1, help="Run detection every N frames."),
    ] = 5,
    detect_batch_size: Annotated[
        int,
        typer.Option(
            "--detect-batch-size",
            min=1,
            help="Detect this many sampled frames per batched forward (faster on "
            "accelerated backends; CoreML/MPS true-batch here). 1 = single-frame.",
        ),
    ] = 32,
    merge_gap_s: Annotated[
        float,
        typer.Option("--merge-gap", min=0.0, help="Merge same-track gaps up to N seconds."),
    ] = 2.0,
    pad_s: Annotated[
        float,
        typer.Option("--pad", min=0.0, help="Seconds of padding added to each span."),
    ] = 1.0,
    min_len_s: Annotated[
        float,
        typer.Option("--min-len", min=0.0, help="Drop spans shorter than N seconds."),
    ] = 0.5,
    max_len_s: Annotated[
        float,
        typer.Option("--max-len", min=0.0, help="Split spans longer than N seconds."),
    ] = 60.0,
    keep_chunks: Annotated[
        bool,
        typer.Option("--keep-chunks", help="Keep downloaded chunk MP4s (debug)."),
    ] = False,
    downloader: Annotated[
        str,
        typer.Option(
            "--downloader",
            help=(
                "Recording transport: 'auto' (probe LAN, fall back to curl), "
                "'uiprotect' (in-process aiohttp), or 'curl' (shell out to the "
                "curl binary — needed when macOS Local Network Privacy blocks the "
                "Python interpreter from the LAN). Default: auto."
            ),
        ),
    ] = "auto",
    dog_alias_classes: DogAliasOption = None,
) -> None:
    """Pull historical UNVR footage for a camera/day in chunks and harvest spans.

    Downloads ``[start, end)`` (or the ``--date`` day) off UniFi Protect in
    ``--chunk``-second windows, runs dog detection on each, and cuts dog-present
    spans into ``<out>/<span_id>/`` clip dirs (immutable H.264 ``clip.mp4`` +
    ``metadata.json``) — ready to label via ``serve``'s Label tab and then
    ``export-dataset``. Failed/empty chunks are skipped; re-runs are idempotent.
    """

    from detectivepotty.detect.yolo import DogDetector

    config = load_cli_config(config_path)
    if not config.protect_configured():
        typer.echo("Protect is not configured; set nvr_host and credentials env vars.")
        raise typer.Exit(1)

    start_utc, end_utc = _resolve_harvest_window(date, start, end, utc_offset)

    alias_classes, alias_nms_iou = resolve_dog_aliases(dog_alias_classes, config)
    detector = DogDetector(
        model_name=model,
        long_edge=long_edge,
        conf_threshold=conf,
        device="auto",
        alias_classes=alias_classes,
        alias_nms_iou=alias_nms_iou,
    )

    harvest_kwargs = dict(
        chunk_s=chunk_s,
        overlap_s=overlap_s,
        sample_every=sample_every,
        detect_batch_size=detect_batch_size,
        merge_gap_s=merge_gap_s,
        pad_s=pad_s,
        min_len_s=min_len_s,
        max_len_s=max_len_s,
        keep_chunks=keep_chunks,
        detect_conf=conf,
    )

    mode = _select_downloader(downloader, config)

    try:
        if mode == "curl":
            results = _harvest_via_curl(
                config, camera, start_utc, end_utc, out_dir, detector, harvest_kwargs
            )
        else:
            results = asyncio.run(
                _harvest_via_uiprotect(
                    config,
                    camera,
                    start_utc,
                    end_utc,
                    out_dir,
                    detector,
                    harvest_kwargs,
                )
            )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        typer.echo(f"Camera harvest failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not results:
        typer.echo(
            "No dog spans harvested (no recording in range, or no dogs present)."
        )
        return
    typer.echo(
        f"Harvested {len(results)} span(s) for {camera} "
        f"[{start_utc.isoformat()} - {end_utc.isoformat()}] into {out_dir}:"
    )
    for result in results:
        span = result.span
        typer.echo(
            f"  {result.span_id}  ({span.start_s:.1f}-{span.end_s:.1f}s, "
            f"track {span.track_id})"
        )


def experiment_acquire_command(
    camera: Annotated[
        str,
        typer.Option("--camera", help="Protect camera id or name (see list-cameras)."),
    ],
    config_path: ConfigPathOption = None,
    date: Annotated[
        Optional[str],
        typer.Option("--date", help="Day to acquire as YYYY-MM-DD (24h at --utc-offset)."),
    ] = None,
    start: Annotated[
        Optional[str],
        typer.Option("--start", help="ISO-8601 start (overrides --date)."),
    ] = None,
    end: Annotated[
        Optional[str],
        typer.Option("--end", help="ISO-8601 end (overrides --date)."),
    ] = None,
    utc_offset: Annotated[
        float,
        typer.Option("--utc-offset", help="Hours offset from UTC for --date. Default 0."),
    ] = 0.0,
    out_dir: Annotated[
        Path,
        typer.Option("--out", help="Directory to write raw chunk MP4s into (gitignored)."),
    ] = Path("data/experiment"),
    chunk_s: Annotated[
        float,
        typer.Option("--chunk", min=1.0, help="Chunk length in seconds. Default 3600 (1h)."),
    ] = 3600.0,
    downloader: Annotated[
        str,
        typer.Option(
            "--downloader",
            help="Recording transport: 'auto' (probe LAN, fall back to curl), "
            "'uiprotect' (in-process), or 'curl' (shell out).",
        ),
    ] = "auto",
) -> None:
    """Download a raw camera window in chunks for the bake-off (NO dog detection).

    Unlike ``harvest-camera`` (which cuts dog spans), this pulls the *whole* window
    unmodified into ``--out`` as contiguous, non-overlapping chunk MP4s named by UTC
    start (so they sort chronologically). Feed the directory to
    ``experiment-bakeoff --input-dir`` to measure how much of it a motion pre-filter
    can skip. Acquisition is bandwidth/disk-heavy — start with a 2–4h window.
    """

    config = load_cli_config(config_path)
    if not config.protect_configured():
        typer.echo("Protect is not configured; set nvr_host and credentials env vars.")
        raise typer.Exit(1)

    start_utc, end_utc = _resolve_harvest_window(date, start, end, utc_offset)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = _select_downloader(downloader, config)

    typer.echo(
        f"Acquiring {camera} [{start_utc.isoformat()} - {end_utc.isoformat()}] "
        f"in {chunk_s:.0f}s chunks via {mode} -> {out_dir}"
    )
    try:
        if mode == "curl":
            paths = _acquire_via_curl(
                config, camera, start_utc, end_utc, out_dir, chunk_s
            )
        else:
            paths = asyncio.run(
                _acquire_via_uiprotect(
                    config, camera, start_utc, end_utc, out_dir, chunk_s
                )
            )
    except Exception as exc:  # noqa: BLE001 - surface a clean CLI error
        typer.echo(f"Acquire failed: {exc}", err=True)
        raise typer.Exit(1) from exc

    if not paths:
        typer.echo("No footage acquired (no recording in range).")
        return
    total_bytes = sum(p.stat().st_size for p in paths if p.exists())
    typer.echo(
        f"Acquired {len(paths)} chunk(s), {total_bytes / 1e9:.2f} GB into {out_dir}. "
        f"Run: detectivepotty experiment-bakeoff --input-dir {out_dir}"
    )


def _select_downloader(mode: str, config) -> str:
    """Resolve the harvest downloader mode; 'auto' probes LAN reachability.

    macOS Local Network Privacy denies the uv-managed Python interpreter access to
    peer LAN devices (the NVR) while leaving the Apple-signed ``curl`` binary
    unaffected. In ``auto`` mode we attempt a direct socket connection from this
    process and fall back to the curl transport when it is blocked.
    """

    mode = (mode or "auto").lower()
    if mode in {"curl", "uiprotect"}:
        return mode
    if mode != "auto":
        raise typer.BadParameter("--downloader must be auto, uiprotect, or curl")

    import socket

    from detectivepotty.protect.client import _parse_host_port

    host, port = _parse_host_port(config.protect.nvr_host)
    try:
        socket.create_connection((host, port), timeout=4.0).close()
        return "uiprotect"
    except OSError:
        from detectivepotty.protect.curl_download import curl_available

        if curl_available():
            typer.echo(
                f"NVR {host}:{port} is unreachable from Python (likely macOS Local "
                "Network Privacy); falling back to the curl downloader.",
                err=True,
            )
            return "curl"
        return "uiprotect"


def _harvest_via_curl(
    config, camera, start_utc, end_utc, out_dir, detector, harvest_kwargs
) -> list:
    """Harvest a camera window using the curl-based Protect downloader."""

    from detectivepotty.harvest_unvr import harvest_camera_window
    from detectivepotty.protect.curl_download import (
        CurlProtectDownloader,
        curl_available,
    )

    if not curl_available():
        raise RuntimeError("curl binary not found on PATH for the curl downloader")
    username = config.resolve_secret("username")
    password = config.resolve_secret("password")
    if not (username and password):
        raise RuntimeError(
            "curl downloader requires DETECTIVEPOTTY_NVR_USERNAME and "
            "DETECTIVEPOTTY_NVR_PASSWORD env vars"
        )
    with CurlProtectDownloader(
        config.protect.nvr_host,
        username,
        password,
        verify_tls=config.protect.verify_tls,
    ) as downloader:
        cameras = downloader.list_cameras()
        camera_id = downloader.resolve_camera_id(camera)
        camera_name = _camera_name_for(
            [(str(c.get("id", "")), str(c.get("name", ""))) for c in cameras],
            camera_id,
        )
        return harvest_camera_window(
            camera_id,
            start_utc,
            end_utc,
            out_dir,
            detector=detector,
            download_fn=downloader.as_download_fn(),
            camera_name=camera_name,
            **harvest_kwargs,
        )


async def _harvest_via_uiprotect(
    config, camera, start_utc, end_utc, out_dir, detector, harvest_kwargs
) -> list:
    """Harvest a camera window using the in-process uiprotect client."""

    from detectivepotty.harvest_unvr import harvest_camera_window
    from detectivepotty.protect.client import ProtectClient

    async with ProtectClient(config) as client:
        cameras = await client.list_cameras()
        camera_id = await _resolve_camera_id(client, camera)
        camera_name = _camera_name_for(
            [(str(getattr(c, "id", "")), str(getattr(c, "name", "") or "")) for c in cameras],
            camera_id,
        )
        loop = asyncio.get_running_loop()

        def download_fn(cam_id, c_start, c_end, dest):
            future = asyncio.run_coroutine_threadsafe(
                client.download_recording(cam_id, c_start, c_end, dest), loop
            )
            return _protect_download_result(future)

        return await asyncio.to_thread(
            harvest_camera_window,
            camera_id,
            start_utc,
            end_utc,
            out_dir,
            detector=detector,
            download_fn=download_fn,
            camera_name=camera_name,
            **harvest_kwargs,
        )


def _camera_name_for(pairs: list[tuple[str, str]], camera_id: str) -> str | None:
    """Pick the friendly name for ``camera_id`` from ``(id, name)`` pairs."""

    for cam_id, name in pairs:
        if cam_id == camera_id and name:
            return name
    return None


def _resolve_harvest_window(
    date: str | None,
    start: str | None,
    end: str | None,
    utc_offset: float,
) -> tuple[datetime, datetime]:
    """Resolve the harvest window from --start/--end or a --date day."""

    if (start is None) != (end is None):
        raise typer.BadParameter("provide both --start and --end, or neither")
    if start is not None and end is not None:
        start_dt = _parse_iso(start)
        end_dt = _parse_iso(end)
    elif date is not None:
        tz = timezone(timedelta(hours=utc_offset))
        try:
            day = datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise typer.BadParameter(f"--date must be YYYY-MM-DD: {date}") from exc
        start_dt = datetime(day.year, day.month, day.day, tzinfo=tz)
        end_dt = start_dt + timedelta(days=1)
    else:
        raise typer.BadParameter("provide --date, or both --start and --end")
    if end_dt <= start_dt:
        raise typer.BadParameter("end must be after start")
    return start_dt.astimezone(timezone.utc), end_dt.astimezone(timezone.utc)


def _parse_iso(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise typer.BadParameter(f"invalid ISO-8601 datetime: {value}") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


async def _resolve_camera_id(client, camera: str) -> str:
    """Match ``camera`` against Protect camera ids, then names (case-insensitive)."""

    cameras = await client.list_cameras()
    for cam in cameras:
        if getattr(cam, "id", None) == camera:
            return camera
    lowered = camera.strip().lower()
    for cam in cameras:
        if (getattr(cam, "name", "") or "").strip().lower() == lowered:
            return cam.id
    raise typer.BadParameter(
        f"camera not found: {camera!r}. Run 'list-cameras' to see ids/names."
    )


def _download_chunks(chunks, camera_id, download_fn, out_dir: Path) -> list[Path]:
    """Download each planned chunk to ``out_dir``; skip empty/failed ones."""

    from detectivepotty.harvest_unvr import _safe

    written: list[Path] = []
    for index, (chunk_start, chunk_end) in enumerate(chunks):
        dest = out_dir / f"{_safe(camera_id)}_{chunk_start:%Y%m%dT%H%M%SZ}.mp4"
        typer.echo(
            f"  chunk {index + 1}/{len(chunks)} "
            f"{chunk_start.isoformat()}..{chunk_end.isoformat()} -> {dest.name}"
        )
        try:
            path = download_fn(camera_id, chunk_start, chunk_end, dest)
        except Exception as exc:  # noqa: BLE001 - one bad chunk must not abort the run
            typer.echo(f"    download failed: {exc}", err=True)
            continue
        if path is None or not Path(path).exists() or Path(path).stat().st_size == 0:
            typer.echo("    no recording / empty chunk")
            continue
        written.append(Path(path))
    return written


def _acquire_via_curl(config, camera, start_utc, end_utc, out_dir, chunk_s) -> list[Path]:
    from detectivepotty.harvest_unvr import plan_chunks
    from detectivepotty.protect.curl_download import (
        CurlProtectDownloader,
        curl_available,
    )

    if not curl_available():
        raise RuntimeError("curl binary not found on PATH for the curl downloader")
    username = config.resolve_secret("username")
    password = config.resolve_secret("password")
    if not (username and password):
        raise RuntimeError(
            "curl downloader requires DETECTIVEPOTTY_NVR_USERNAME and "
            "DETECTIVEPOTTY_NVR_PASSWORD env vars"
        )
    with CurlProtectDownloader(
        config.protect.nvr_host,
        username,
        password,
        verify_tls=config.protect.verify_tls,
    ) as dl:
        camera_id = dl.resolve_camera_id(camera)
        chunks = plan_chunks(start_utc, end_utc, chunk_s=chunk_s, overlap_s=0.0)
        return _download_chunks(chunks, camera_id, dl.as_download_fn(), out_dir)


async def _acquire_via_uiprotect(
    config, camera, start_utc, end_utc, out_dir, chunk_s
) -> list[Path]:
    from detectivepotty.harvest_unvr import plan_chunks
    from detectivepotty.protect.client import ProtectClient

    async with ProtectClient(config) as client:
        camera_id = await _resolve_camera_id(client, camera)
        loop = asyncio.get_running_loop()

        def download_fn(cam_id, c_start, c_end, dest):
            future = asyncio.run_coroutine_threadsafe(
                client.download_recording(cam_id, c_start, c_end, dest), loop
            )
            return _protect_download_result(future)

        chunks = plan_chunks(start_utc, end_utc, chunk_s=chunk_s, overlap_s=0.0)
        return await asyncio.to_thread(
            _download_chunks, chunks, camera_id, download_fn, out_dir
        )
