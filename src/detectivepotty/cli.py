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
) -> None:
    """Launch the local review web app."""

    config = load_config(config_path)
    try:
        from detectivepotty.web import run_server
    except Exception as exc:
        typer.echo(f"Web app is unavailable: {exc}", err=True)
        raise typer.Exit(1) from exc
    run_server(config, host=host, port=port)


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
    ] = "yolo11m.pt",
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
