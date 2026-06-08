"""Command-line interface for offline detection experiments."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Annotated

import cv2
import numpy as np
import typer

from detectivepotty.config import Config, load_config
from detectivepotty.detect.yolo import DogDetector
from detectivepotty.geometry import crop_from_frame

app = typer.Typer(help="DetectivePotty offline and live utilities.")


@app.callback()
def main() -> None:
    """DetectivePotty offline and live utilities."""


@app.command("run")
def run_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to DetectivePotty YAML config.",
        ),
    ],
    camera_ids: Annotated[
        list[str] | None,
        typer.Option("--camera", "-C", help="Camera id to run; repeat for multiple."),
    ] = None,
    max_workers: Annotated[
        int | None,
        typer.Option(
            "--max-workers",
            "-w",
            min=1,
            help="Max cameras to process concurrently (default: one thread per "
            "camera). Live cameras always keep a dedicated thread; a value below "
            "the live-camera count is raised back up. Use 1 to force sequential "
            "processing of file cameras.",
        ),
    ] = None,
) -> None:
    """Run the end-to-end pipeline for enabled or selected cameras.

    Multiple cameras run concurrently. Live (Protect) cameras stream until you
    interrupt with Ctrl-C; file cameras finish when the clip ends.
    """

    from detectivepotty.pipeline import run_pipeline

    config = load_config(config_path)
    event_dirs = run_pipeline(config, camera_ids=camera_ids, max_workers=max_workers)
    if not event_dirs:
        typer.echo("No events recorded.")
        return

    typer.echo(f"Recorded {len(event_dirs)} event(s):")
    for event_dir in event_dirs:
        typer.echo(f"  {event_dir}")


@app.command("dedupe-events")
def dedupe_events_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to DetectivePotty YAML config.",
        ),
    ],
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Report the plan without deleting anything."),
    ] = False,
) -> None:
    """Collapse existing duplicate events left behind by earlier reruns.

    Groups events by camera + source, keeps the newest-media copy of each
    duplicate cluster, carries human labels forward, and deletes the rest.
    Clusters whose human labels disagree are left untouched.
    """

    from detectivepotty.recording.reconcile import dedupe_dataset

    config = load_config(config_path)
    actions = dedupe_dataset(
        config.global_settings.dataset_dir,
        tolerance_s=config.global_settings.rerun_match_tolerance_s,
        dry_run=dry_run,
    )

    removed = 0
    conflicts = 0
    for action in actions:
        if action.conflict:
            conflicts += 1
            typer.echo(f"CONFLICT (kept all {len(action.cluster)}):")
            for path in action.cluster:
                typer.echo(f"    {path}")
            continue
        removed += len(action.removed)
        verb = "would keep" if dry_run else "kept"
        typer.echo(f"{verb} {action.keeper}")
        for path in action.removed:
            verb = "would remove" if dry_run else "removed"
            typer.echo(f"    {verb} {path}")

    prefix = "Dry run: " if dry_run else ""
    typer.echo(
        f"{prefix}{removed} duplicate event(s) "
        f"{'to remove' if dry_run else 'removed'}, {conflicts} conflict(s)."
    )


@app.command("cleanup-legacy")
def cleanup_legacy_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to DetectivePotty YAML config.",
        ),
    ],
    apply: Annotated[
        bool,
        typer.Option(
            "--apply",
            help="Actually quarantine removable events. Omit for a dry run.",
        ),
    ] = False,
) -> None:
    """Quarantine legacy duplicate events left by the pre-determinism timeline.

    Conservatively removes only unlabeled, pre-determinism events whose source
    video still exists (so a clean re-run can regenerate them), moving them into
    ``<dataset>/.trash/`` so the operation is reversible. Every reviewed event and
    every deterministic-era event is preserved. Runs as a dry run unless
    ``--apply`` is given.
    """

    from detectivepotty.recording.cleanup import cleanup_legacy_events

    config = load_config(config_path)
    report = cleanup_legacy_events(
        config.global_settings.dataset_dir,
        dry_run=not apply,
    )

    for item in report.removable:
        if apply and item.moved_to is not None:
            typer.echo(f"quarantined {item.event_dir} -> {item.moved_to}")
        elif apply:
            typer.echo(f"FAILED to quarantine {item.event_dir}")
        else:
            typer.echo(f"would quarantine {item.event_dir}")
    for item in report.skipped_source_missing:
        typer.echo(f"skipped (source missing) {item.event_dir}")

    prefix = "" if apply else "Dry run: "
    verb = "quarantined" if apply else "to quarantine"
    typer.echo(
        f"{prefix}{len(report.removable)} legacy duplicate(s) {verb}; "
        f"kept {len(report.kept_labeled)} labeled, "
        f"{len(report.kept_deterministic)} deterministic, "
        f"{len(report.skipped_source_missing)} with missing source."
    )
    if apply and report.trash_dir is not None and report.removable:
        typer.echo(f"Quarantine dir: {report.trash_dir}")


@app.command("serve")
def serve_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to DetectivePotty YAML config.",
        ),
    ],
    host: Annotated[str, typer.Option("--host", help="Bind host.")] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", min=1, max=65535, help="Bind port.")] = 8000,
    reload: Annotated[
        bool,
        typer.Option(
            "--reload",
            help="Auto-reload the server when source files change (development only).",
        ),
    ] = False,
) -> None:
    """Launch the local review web app."""

    config = load_config(config_path)
    try:
        from detectivepotty.web import run_server
    except Exception as exc:
        typer.echo(f"Web app is unavailable: {exc}", err=True)
        raise typer.Exit(1) from exc
    run_server(config, host=host, port=port, reload=reload, config_path=config_path)


@app.command("list-cameras")
def list_cameras_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to DetectivePotty YAML config.",
        ),
    ],
) -> None:
    """Best-effort UniFi Protect camera discovery."""

    config = load_config(config_path)
    if not _protect_configured(config):
        typer.echo("Protect is not configured; set nvr_host and credentials env vars.")
        return

    async def _list() -> None:
        try:
            from detectivepotty.protect.client import ProtectClient
        except Exception as exc:
            typer.echo(f"Protect support is unavailable: {exc}", err=True)
            raise typer.Exit(1) from exc

        try:
            async with ProtectClient(config) as client:
                cameras = await client.list_cameras()
        except Exception as exc:
            typer.echo(f"Could not list Protect cameras: {exc}", err=True)
            raise typer.Exit(1) from exc

        if not cameras:
            typer.echo("No Protect cameras found.")
            return
        for camera in cameras:
            typer.echo(
                f"{camera.id}\t{camera.name}\t"
                f"connected={camera.is_connected}\t"
                f"animal={camera.animal_smart_detect_supported}",
            )

    asyncio.run(_list())


@app.command("detect-file")
def detect_file(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Input video file.",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Annotated MP4 output path."),
    ] = Path("outputs/annotated.mp4"),
    save_crops: Annotated[
        Path | None,
        typer.Option("--save-crops", help="Directory for high-res dog crops."),
    ] = None,
    long_edge: Annotated[
        int,
        typer.Option(
            "--long-edge",
            min=1,
            help="YOLO inference long edge (ultralytics imgsz). Default 640 (model-native).",
        ),
    ] = 640,
    every_n: Annotated[
        int,
        typer.Option("--every-n", min=1, help="Run detection every N frames."),
    ] = 1,
    model: Annotated[
        str,
        typer.Option("--model", help="YOLO model name/path."),
    ] = "models/yolo11m.pt",
) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise typer.BadParameter(f"Could not open input video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise typer.BadParameter("Input video did not report a valid resolution")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise typer.BadParameter(f"Could not open output video writer: {output}")

    if save_crops is not None:
        save_crops.mkdir(parents=True, exist_ok=True)

    detector = DogDetector(model_name=model, long_edge=long_edge, device="auto")
    frame_idx = 0
    detection_frames = 0
    dogs_detected = 0
    frames_with_dog = 0
    crops_saved = 0
    inference_latencies_ms: list[float] = []
    inference_resolution: tuple[int, int] | None = None
    started = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            detections = []
            if frame_idx % every_n == 0:
                detection_frames += 1
                detections = detector.detect(
                    frame,
                    frame_idx=frame_idx,
                    mono_ts=time.monotonic(),
                    wall_ts=datetime.now(timezone.utc),
                )
                if detector.last_inference is not None:
                    inference_latencies_ms.append(detector.last_inference.latency_ms)
                    inference_resolution = detector.last_inference.inference_wh
                dogs_detected += len(detections)
                if detections:
                    frames_with_dog += 1
                    if save_crops is not None:
                        crops_saved += _save_best_crop(
                            frame,
                            detections,
                            save_crops,
                            frame_idx,
                        )

            _draw_detections(frame, detections)
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    elapsed_s = max(time.perf_counter() - started, 1e-9)
    _print_summary(
        frames_read=frame_idx,
        detection_frames=detection_frames,
        dogs_detected=dogs_detected,
        frames_with_dog=frames_with_dog,
        crops_saved=crops_saved,
        output=output,
        save_crops=save_crops,
        original_resolution=(width, height),
        inference_resolution=inference_resolution,
        latencies_ms=inference_latencies_ms,
        elapsed_s=elapsed_s,
        device=detector.device,
        model_name=detector.model_name,
    )


@app.command("tune-detect")
def tune_detect(
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            help="Local video file to tune against. Mutually exclusive with --config/--camera.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Config supplying a camera to stream (use with --camera).",
        ),
    ] = None,
    camera_id: Annotated[
        str | None,
        typer.Option("--camera", "-C", help="Camera id within --config (file, rtsp, or protect)."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="YOLO model. Defaults to the config global model or models/yolo11m.pt."),
    ] = None,
    long_edge: Annotated[
        int,
        typer.Option("--long-edge", min=1, help="YOLO inference long edge (imgsz). Default 640."),
    ] = 640,
    floor: Annotated[
        float,
        typer.Option(
            "--floor",
            min=0.0,
            max=1.0,
            help="Detection floor: boxes below this are never returned. Keep low (e.g. 0.05) so "
            "borderline boxes stay visible for the slider.",
        ),
    ] = 0.05,
    conf: Annotated[
        float | None,
        typer.Option(
            "--conf",
            min=0.0,
            max=1.0,
            help="Initial slider threshold. Defaults to the camera's detection_conf_threshold or 0.25.",
        ),
    ] = None,
    every_n: Annotated[
        int,
        typer.Option(
            "--every-n",
            min=1,
            help="Run detection every N played frames (reuse boxes in between for smoother playback).",
        ),
    ] = 1,
) -> None:
    """Interactively tune ``detection_conf_threshold`` with live bounding boxes.

    Opens an OpenCV window playing the input with every detected dog drawn. A confidence
    slider colors boxes green (kept) vs dim red (dropped) at that threshold, so you can
    watch playback and find the cutoff that keeps the dog but drops noise. Detection runs
    at a low ``--floor`` so borderline boxes stay visible; the slider only changes coloring,
    not inference. Accepts a local file (``--input``) or a config camera
    (``--config``/``--camera``; file, rtsp, or protect). Prints the chosen threshold on exit.
    """

    from detectivepotty.preview import (
        FileFrameProvider,
        FrameProvider,
        LiveFrameProvider,
        run_interactive_preview,
    )

    provider: FrameProvider
    if input_path is not None:
        if config_path is not None or camera_id is not None:
            raise typer.BadParameter("Use either --input or --config/--camera, not both.")
        if not input_path.is_file():
            raise typer.BadParameter(f"Input video not found: {input_path}")
        provider = FileFrameProvider(input_path)
        model_name = model or "models/yolo11m.pt"
        initial_conf = conf if conf is not None else 0.25
    else:
        if config_path is None or camera_id is None:
            raise typer.BadParameter("Provide --input, or both --config and --camera.")
        config = load_config(config_path)
        camera = next((cam for cam in config.cameras if cam.id == camera_id), None)
        if camera is None:
            raise typer.BadParameter(f"Camera '{camera_id}' not found in {config_path}.")
        model_name = model or config.global_settings.model_name
        initial_conf = conf if conf is not None else camera.detection_conf_threshold
        provider = _build_camera_provider(config, camera, FileFrameProvider, LiveFrameProvider)

    detector = DogDetector(
        model_name=model_name,
        long_edge=long_edge,
        conf_threshold=floor,
        device="auto",
    )
    source_desc = "live stream" if provider.is_live else "file"
    typer.echo(
        f"Tuning {model_name} on {source_desc} (device={detector.device}); "
        f"floor={floor:.2f}, start threshold={initial_conf:.2f}"
    )
    typer.echo(
        "Keys: space play/pause, n/p step, r restart (file only), q/Esc quit. "
        "Drag the 'conf x100' slider to set the threshold."
    )
    chosen = run_interactive_preview(
        provider,
        detector,
        initial_conf=initial_conf,
        every_n=every_n,
    )
    typer.echo(f"Chosen detection_conf_threshold: {chosen:.2f}")


def _build_camera_provider(config, camera, file_provider_cls, live_provider_cls):
    kind = camera.input.kind
    if kind == "file":
        if camera.input.path is None:
            raise typer.BadParameter(f"Camera '{camera.id}' has no input.path.")
        if not camera.input.path.is_file():
            raise typer.BadParameter(f"Camera '{camera.id}' input file not found: {camera.input.path}")
        return file_provider_cls(camera.input.path)
    url = _resolve_camera_url(config, camera)
    try:
        from detectivepotty.sources.rtsp import RTSPSource
    except Exception as exc:  # pragma: no cover - environment dependent.
        raise typer.BadParameter(f"RTSP support unavailable: {exc}") from exc
    return live_provider_cls(RTSPSource(url))


def _resolve_camera_url(config: Config, camera) -> str:
    kind = camera.input.kind
    if kind == "rtsp":
        url = camera.input.resolve_url()
        if not url:
            raise typer.BadParameter(
                f"rtsp camera '{camera.id}': env var {camera.input.url_env} is unset or empty."
            )
        return url
    if kind == "protect":
        return _resolve_protect_url(config, camera)
    raise typer.BadParameter(f"Unsupported camera kind '{kind}' for tune-detect.")


def _resolve_protect_url(config: Config, camera) -> str:
    async def _resolve() -> str | None:
        from detectivepotty.protect.client import ProtectClient

        client = ProtectClient(config)
        try:
            await client.connect()
            return client.rtsps_url(camera.id, camera.substream_choice)
        finally:
            await client.close()

    url = asyncio.run(_resolve())
    if not url:
        raise typer.BadParameter(
            f"No RTSPS URL for protect camera '{camera.id}' substream {camera.substream_choice}."
        )
    return url


def _protect_configured(config: Config) -> bool:
    has_host = bool(config.protect.nvr_host)
    has_api_key = bool(config.resolve_secret("api_key"))
    has_userpass = bool(config.resolve_secret("username") and config.resolve_secret("password"))
    return has_host and (has_api_key or has_userpass)



def _draw_detections(frame: np.ndarray, detections: list[object]) -> None:
    height, width = frame.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox.clip_to(width, height).to_int_tuple()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 220, 30), 3)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (30, 220, 30),
            2,
            cv2.LINE_AA,
        )


def _save_best_crop(
    frame: np.ndarray,
    detections: list[object],
    save_crops: Path,
    frame_idx: int,
) -> int:
    best = max(detections, key=lambda detection: detection.confidence)
    crop = crop_from_frame(frame, best.bbox, margin_frac=0.5)
    if crop.size == 0:
        return 0
    crop_path = save_crops / f"frame_{frame_idx:06d}_dog_{best.confidence:.2f}.jpg"
    if not cv2.imwrite(str(crop_path), crop):
        raise typer.BadParameter(f"Failed to write crop: {crop_path}")
    return 1


def _print_summary(
    *,
    frames_read: int,
    detection_frames: int,
    dogs_detected: int,
    frames_with_dog: int,
    crops_saved: int,
    output: Path,
    save_crops: Path | None,
    original_resolution: tuple[int, int],
    inference_resolution: tuple[int, int] | None,
    latencies_ms: list[float],
    elapsed_s: float,
    device: str,
    model_name: str,
) -> None:
    if latencies_ms:
        mean_latency = float(np.mean(latencies_ms))
        p95_latency = float(np.percentile(latencies_ms, 95))
        inference_fps = detection_frames / (sum(latencies_ms) / 1000.0)
    else:
        mean_latency = 0.0
        p95_latency = 0.0
        inference_fps = 0.0
    end_to_end_fps = frames_read / elapsed_s

    typer.echo("Detection summary")
    typer.echo(f"  model: {model_name}")
    typer.echo(f"  device: {device}")
    typer.echo(f"  frames read: {frames_read}")
    typer.echo(f"  detection frames: {detection_frames}")
    typer.echo(f"  dog detections: {dogs_detected}")
    typer.echo(f"  frames with dog: {frames_with_dog}")
    typer.echo(f"  crops saved: {crops_saved}")
    typer.echo(f"  annotated video: {output}")
    if save_crops is not None:
        typer.echo(f"  crop directory: {save_crops}")
    typer.echo(f"  original resolution: {original_resolution[0]}x{original_resolution[1]}")
    if inference_resolution is not None:
        typer.echo(f"  inference resolution: {inference_resolution[0]}x{inference_resolution[1]}")
    typer.echo(f"  mean inference FPS: {inference_fps:.2f}")
    typer.echo(f"  end-to-end FPS: {end_to_end_fps:.2f}")
    typer.echo(f"  mean inference latency ms: {mean_latency:.1f}")
    typer.echo(f"  p95 inference latency ms: {p95_latency:.1f}")
