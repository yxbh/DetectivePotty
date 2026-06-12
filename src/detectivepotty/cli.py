"""Command-line interface for offline detection experiments."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import time
from typing import Annotated, Optional

import cv2
import numpy as np
import typer

from detectivepotty.config import Config, DEFAULT_DOG_ALIAS_CLASSES, load_config
from detectivepotty.detect.yolo import DogDetector
from detectivepotty.geometry import crop_from_frame

app = typer.Typer(help="DetectivePotty offline and live utilities.")


def _resolve_dog_aliases(
    override: str | None, config: Config | None = None
) -> tuple[list[str], float]:
    """Resolve the accepted dog-alias classes + NMS IoU for a CLI detector.

    Precedence: an explicit ``--dog-alias-classes`` override (comma list; empty
    string disables) wins; otherwise the loaded config's values; otherwise the
    built-in safe-set default. Keeps aliases default-ON for every detection command.
    """

    iou = config.global_settings.dog_alias_nms_iou if config is not None else 0.65
    if override is not None:
        classes = [c.strip().lower() for c in override.split(",") if c.strip()]
        return classes, iou
    if config is not None:
        return list(config.global_settings.dog_alias_classes), iou
    return list(DEFAULT_DOG_ALIAS_CLASSES), iou


# Shared CLI option for overriding the accepted dog-alias classes on detection
# commands. ``None`` (the default) means "use config / the built-in safe set".
DogAliasOption = Annotated[
    Optional[str],
    typer.Option(
        "--dog-alias-classes",
        help=(
            "Comma-separated YOLO classes to also accept as dogs (e.g. "
            "'sheep,cow'). Empty string disables. Defaults to config / the "
            "built-in safe set."
        ),
    ),
]


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

    async def _list() -> list:
        from detectivepotty.protect.client import ProtectClient

        async with ProtectClient(config) as client:
            cameras = await client.list_cameras()
            return [
                (
                    cam.id,
                    cam.name,
                    cam.is_connected,
                    cam.animal_smart_detect_supported,
                )
                for cam in cameras
            ]

    try:
        rows = asyncio.run(_list())
    except Exception as exc:  # noqa: BLE001 - fall back to the curl transport
        rows = _list_cameras_via_curl(config)
        if rows is None:
            typer.echo(f"Could not list Protect cameras: {exc}", err=True)
            raise typer.Exit(1) from exc

    if not rows:
        typer.echo("No Protect cameras found.")
        return
    for cam_id, name, connected, animal in rows:
        typer.echo(f"{cam_id}\t{name}\tconnected={connected}\tanimal={animal}")


def _list_cameras_via_curl(config) -> list | None:
    """List cameras via the curl transport; None if curl/creds are unavailable."""

    from detectivepotty.protect.curl_download import (
        CurlProtectDownloader,
        curl_available,
    )

    username = config.resolve_secret("username")
    password = config.resolve_secret("password")
    if not curl_available() or not (username and password):
        return None
    try:
        with CurlProtectDownloader(
            config.protect.nvr_host,
            username,
            password,
            verify_tls=config.protect.verify_tls,
        ) as downloader:
            cameras = downloader.list_cameras()
    except Exception:  # noqa: BLE001 - caller reports the original error
        return None
    return [
        (cam.get("id"), cam.get("name"), cam.get("state") == "CONNECTED", "?")
        for cam in cameras
    ]


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
    dog_alias_classes: DogAliasOption = None,
) -> None:
    from detectivepotty.sources.pyav_capture import open_capture

    cap = open_capture(str(input_path))
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

    alias_classes, alias_nms_iou = _resolve_dog_aliases(dog_alias_classes)
    detector = DogDetector(
        model_name=model,
        long_edge=long_edge,
        device="auto",
        alias_classes=alias_classes,
        alias_nms_iou=alias_nms_iou,
    )
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
    dog_alias_classes: DogAliasOption = None,
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
    config: Config | None = None
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

    alias_classes, alias_nms_iou = _resolve_dog_aliases(dog_alias_classes, config)
    detector = DogDetector(
        model_name=model_name,
        long_edge=long_edge,
        conf_threshold=floor,
        device="auto",
        alias_classes=alias_classes,
        alias_nms_iou=alias_nms_iou,
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


@app.command("export-coreml")
def export_coreml_command(
    weights: Annotated[
        list[Path],
        typer.Argument(help="YOLO .pt weights to export (one or more, e.g. models/yolo11m.pt)."),
    ],
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Directory to write <stem>.mlpackage into (e.g. models/coreml for the "
            "committable set). Default: next to each .pt.",
        ),
    ] = None,
    imgsz: Annotated[
        int,
        typer.Option("--imgsz", min=32, help="Inference long edge baked into the model. Default 640."),
    ] = 640,
    half: Annotated[
        bool,
        typer.Option("--half/--no-half", help="Export FP16 (default, GPU-fast) vs FP32."),
    ] = True,
    batch: Annotated[
        int,
        typer.Option(
            "--batch",
            min=1,
            help="Max batch the package accepts. 1 (default) = fixed single-image "
            "package; >1 = dynamic flexible-shape package that batches 1..N in one "
            "GPU forward (the fast ground-truth backend — match the caller's batch).",
        ),
    ] = 1,
) -> None:
    """Export YOLO11 ``.pt`` weights to a GPU-safe CoreML ``.mlpackage``.

    Rewrites YOLO11's C2PSA attention with scaled-dot-product-attention so the
    exported ``mlprogram`` runs on Apple's GPU (~2x faster than the ``.pt`` MPS
    path) instead of crashing the MPSGraph compiler — numerically identical to the
    original weights. macOS-only. The result is auto-discovered by the ``/tune``
    model picker; pass ``--out models/coreml`` to produce the curated, committable
    set. Use ``--batch 32`` for a batched package (fastest dense-detection backend).
    """

    from detectivepotty.detect.coreml_export import export_coreml

    for pt in weights:
        if not pt.is_file():
            raise typer.BadParameter(f"Weights not found: {pt}")
        out_path = (out_dir / f"{pt.stem}.mlpackage") if out_dir is not None else None
        typer.echo(
            f"Exporting {pt} -> CoreML (imgsz={imgsz}, half={half}, batch={batch}) ..."
        )
        result = export_coreml(pt, out_path=out_path, imgsz=imgsz, half=half, batch=batch)
        typer.echo(f"  saved: {result}")


@app.command("harvest")
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

    from detectivepotty.harvest import harvest_clips

    alias_classes, alias_nms_iou = _resolve_dog_aliases(dog_alias_classes)
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


@app.command("harvest-camera")
def harvest_camera_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to DetectivePotty YAML config (for Protect host/creds).",
        ),
    ],
    camera: Annotated[
        str,
        typer.Option("--camera", help="Protect camera id or name (see list-cameras)."),
    ],
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

    config = load_config(config_path)
    if not _protect_configured(config):
        typer.echo("Protect is not configured; set nvr_host and credentials env vars.")
        raise typer.Exit(1)

    start_utc, end_utc = _resolve_harvest_window(date, start, end, utc_offset)

    alias_classes, alias_nms_iou = _resolve_dog_aliases(dog_alias_classes, config)
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
            return future.result()

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


@app.command("export-dataset")
def export_dataset_command(
    clips_root: Annotated[
        Path,
        typer.Option(
            "--clips",
            exists=True,
            file_okay=False,
            help="Root of harvested clip dirs (each with clip.mp4 + labels.json).",
        ),
    ] = Path("dataset/harvest"),
    out_dir: Annotated[
        Path,
        typer.Option("--out", help="Directory to write the classifier dataset into."),
    ] = Path("dataset/export"),
    model: Annotated[
        str,
        typer.Option("--model", help="YOLO model name/path for dense re-detection."),
    ] = "models/yolo11m.pt",
    long_edge: Annotated[
        int,
        typer.Option("--long-edge", min=1, help="YOLO inference long edge (imgsz)."),
    ] = 640,
    conf: Annotated[
        float,
        typer.Option("--conf", min=0.0, max=1.0, help="Detection confidence threshold."),
    ] = 0.25,
    stride_s: Annotated[
        float,
        typer.Option("--stride", min=0.0, help="Within-range sample stride (seconds)."),
    ] = 0.3,
    max_frames: Annotated[
        int,
        typer.Option("--max-frames", min=1, help="Max crops per labeled range."),
    ] = 40,
    margin: Annotated[
        float,
        typer.Option("--margin", min=0.0, help="Crop margin fraction around the box."),
    ] = 0.35,
    val_fraction: Annotated[
        float,
        typer.Option("--val-fraction", min=0.0, max=1.0, help="Day/source val split."),
    ] = 0.2,
    min_iou: Annotated[
        float,
        typer.Option("--min-iou", min=0.0, max=1.0, help="Track-binding IoU gate."),
    ] = 0.3,
    dog_alias_classes: DogAliasOption = None,
) -> None:
    """Build classifier crops + a CSV manifest from labeled harvested clips.

    Re-detects densely on each sampled frame, binds the crop to the range's dog
    track, and writes an image-classifier folder layout (``behavior/`` and
    ``dog/`` trees split train/val by day+source) plus ``manifest.csv`` and
    ``export_stats.json``.
    """

    from detectivepotty.dataset_export import export_dataset

    alias_classes, alias_nms_iou = _resolve_dog_aliases(dog_alias_classes)
    detector = DogDetector(
        model_name=model,
        long_edge=long_edge,
        conf_threshold=conf,
        device="auto",
        alias_classes=alias_classes,
        alias_nms_iou=alias_nms_iou,
    )
    stats = export_dataset(
        clips_root,
        out_dir,
        detector=detector,
        sample_stride_s=stride_s,
        max_frames_per_range=max_frames,
        crop_margin_frac=margin,
        val_fraction=val_fraction,
        min_iou=min_iou,
    )
    typer.echo(f"Exported {stats.crops_written} crop(s) from {stats.clips} clip(s).")
    typer.echo(f"  behavior: {stats.behavior_counts}")
    typer.echo(f"  dog:      {stats.dog_counts}")
    typer.echo(f"  split:    {stats.split_counts}")
    if stats.dropped_unmatched:
        typer.echo(f"  dropped (unmatched track): {stats.dropped_unmatched}")
    typer.echo(f"  dataset:  {out_dir}")


@app.command("experiment-bakeoff")
def experiment_bakeoff_command(
    input_path: Annotated[
        Optional[Path],
        typer.Option(
            "--input",
            exists=True,
            dir_okay=False,
            readable=True,
            help="A single acquired window video to score strategies over.",
        ),
    ] = None,
    input_dir: Annotated[
        Optional[Path],
        typer.Option(
            "--input-dir",
            exists=True,
            file_okay=False,
            readable=True,
            help="A directory of chunk videos (experiment-acquire output) scored and "
            "aggregated into one window-wide report. Use instead of --input.",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option(
            "--model",
            help="Ground-truth detector. A batched CoreML .mlpackage "
            "(export-coreml --batch 32) is fastest; .pt also works.",
        ),
    ] = "models/yolo11m.pt",
    long_edge: Annotated[
        int,
        typer.Option("--long-edge", min=1, help="YOLO inference long edge (imgsz)."),
    ] = 640,
    conf: Annotated[
        float,
        typer.Option("--conf", min=0.0, max=1.0, help="Detection confidence threshold."),
    ] = 0.25,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", min=1, help="Ground-truth detect batch size (CoreML fastest at 32)."),
    ] = 32,
    thresholds: Annotated[
        str,
        typer.Option(
            "--thresholds",
            help="Comma-separated motion thresholds (fraction of peak second) to sweep.",
        ),
    ] = "0.05,0.10,0.15,0.25,0.40",
    min_dog_frames: Annotated[
        int,
        typer.Option("--min-dog-frames", min=1, help="Dog frames/second to count as a dog-second."),
    ] = 1,
    pad_s: Annotated[
        int,
        typer.Option("--pad-s", min=0, help="Seconds to dilate each motion hit (recall guard)."),
    ] = 1,
    dog_alias_classes: DogAliasOption = None,
) -> None:
    """Score retro-harvest window strategies against an exhaustive dense-YOLO truth.

    Runs the detector on EVERY frame of the input (one ``--input`` file, or every
    chunk under ``--input-dir`` aggregated) to build the ground-truth dog-seconds
    (this is the ``blind-scrub`` baseline), then scores the compressed-domain motion
    pre-filter (swept across ``--thresholds``) on recall of those dog-seconds vs the
    compute it saves. Prints a bake-off table; pick the greediest threshold that
    still holds recall near 1.0. Offline/local — no NVR.
    """

    from detectivepotty.experiment import find_chunk_videos, run_bakeoff, run_bakeoff_dir

    if (input_path is None) == (input_dir is None):
        raise typer.BadParameter("provide exactly one of --input or --input-dir")

    try:
        threshold_vals = tuple(
            float(t) for t in thresholds.split(",") if t.strip()
        )
    except ValueError as exc:
        raise typer.BadParameter(f"--thresholds must be comma-separated floats: {exc}")
    if not threshold_vals:
        raise typer.BadParameter("--thresholds must contain at least one value")

    if input_dir is not None:
        chunks = find_chunk_videos(input_dir)
        if not chunks:
            raise typer.BadParameter(f"no chunk videos found in {input_dir}")

    alias_classes, alias_nms_iou = _resolve_dog_aliases(dog_alias_classes)
    detector = DogDetector(
        model_name=model,
        long_edge=long_edge,
        conf_threshold=conf,
        device="auto",
        alias_classes=alias_classes,
        alias_nms_iou=alias_nms_iou,
    )
    typer.echo(
        f"Building ground truth (model={detector.model_name}, device={detector.device}, "
        f"batch={batch_size}) — this runs YOLO on every frame ..."
    )
    started = time.perf_counter()
    if input_dir is not None:
        def _progress(i: int, n: int, path: Path) -> None:
            typer.echo(f"  chunk {i + 1}/{n}: {path.name}")

        report = run_bakeoff_dir(
            chunks,
            detector,
            source=f"{input_dir} ({len(chunks)} chunks)",
            thresholds=threshold_vals,
            min_dog_frames=min_dog_frames,
            pad_s=pad_s,
            batch_size=batch_size,
            progress=_progress,
        )
    else:
        report = run_bakeoff(
            str(input_path),
            detector,
            source=input_path.name,
            thresholds=threshold_vals,
            min_dog_frames=min_dog_frames,
            pad_s=pad_s,
            batch_size=batch_size,
        )
    elapsed_s = time.perf_counter() - started
    typer.echo("")
    typer.echo(report.format_table())
    typer.echo("")
    typer.echo(f"# ground truth built in {elapsed_s:.1f}s")


@app.command("experiment-acquire")
def experiment_acquire_command(
    config_path: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Path to DetectivePotty YAML config (for Protect host/creds).",
        ),
    ],
    camera: Annotated[
        str,
        typer.Option("--camera", help="Protect camera id or name (see list-cameras)."),
    ],
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

    config = load_config(config_path)
    if not _protect_configured(config):
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
            return future.result()

        chunks = plan_chunks(start_utc, end_utc, chunk_s=chunk_s, overlap_s=0.0)
        return await asyncio.to_thread(
            _download_chunks, chunks, camera_id, download_fn, out_dir
        )


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
