"""Command-line interface for offline detection experiments."""

from __future__ import annotations

import asyncio
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Annotated

import typer

from detectivepotty.cli_common import (
    ConfigPathOption,
    DogAliasOption,
    PROTECT_DOWNLOAD_TIMEOUT_S,
    load_cli_config,
    resolve_dog_aliases,
    set_cli_log_level,
)
from detectivepotty.cli_detect import register_detect_commands
from detectivepotty.cli_export import register_export_commands
from detectivepotty.cli_harvest import register_harvest_commands
from detectivepotty.config import (
    Config,
    resolve_config_path,
)

app = typer.Typer(help="DetectivePotty offline and live utilities.")

_PROTECT_DOWNLOAD_TIMEOUT_S = PROTECT_DOWNLOAD_TIMEOUT_S


def _protect_download_result(future):
    try:
        return future.result(timeout=_PROTECT_DOWNLOAD_TIMEOUT_S)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(
            "Protect recording export timed out "
            f"after {_PROTECT_DOWNLOAD_TIMEOUT_S:.0f}s"
        ) from exc


@app.callback()
def main(
    log_level: Annotated[
        str | None,
        typer.Option(
            "--log-level",
            help="CLI log level override. Defaults to config global.log_level when a config is loaded.",
        ),
    ] = None,
) -> None:
    """DetectivePotty offline and live utilities."""
    set_cli_log_level(log_level)


register_detect_commands(app)
register_export_commands(app)
register_harvest_commands(app, _protect_download_result)


@app.command("run")
def run_command(
    config_path: ConfigPathOption = None,
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

    config = load_cli_config(config_path)
    event_dirs = run_pipeline(config, camera_ids=camera_ids, max_workers=max_workers)
    if not event_dirs:
        typer.echo("No events recorded.")
        return

    typer.echo(f"Recorded {len(event_dirs)} event(s):")
    for event_dir in event_dirs:
        typer.echo(f"  {event_dir}")


@app.command("dedupe-events")
def dedupe_events_command(
    config_path: ConfigPathOption = None,
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

    config = load_cli_config(config_path)
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
    config_path: ConfigPathOption = None,
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

    config = load_cli_config(config_path)
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
    config_path: ConfigPathOption = None,
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

    config = load_cli_config(config_path)
    try:
        from detectivepotty.web import run_server
    except Exception as exc:
        typer.echo(f"Web app is unavailable: {exc}", err=True)
        raise typer.Exit(1) from exc
    run_server(config, host=host, port=port, reload=reload, config_path=config_path)


@app.command("list-cameras")
def list_cameras_command(
    config_path: ConfigPathOption = None,
) -> None:
    """Best-effort UniFi Protect camera discovery."""

    config = load_cli_config(config_path)
    if not config.protect_configured():
        typer.echo("Protect is not configured; set nvr_host and credentials env vars.")
        raise typer.Exit(1)

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


@app.command("tune-detect")
def tune_detect(
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            help="Local video file to tune against. Mutually exclusive with --config/--camera.",
        ),
    ] = None,
    config_path: ConfigPathOption = None,
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
    from detectivepotty.detect.yolo import DogDetector

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
        if camera_id is None:
            raise typer.BadParameter("Provide --input, or --camera.")
        config = load_cli_config(config_path)
        camera = next((cam for cam in config.cameras if cam.id == camera_id), None)
        if camera is None:
            raise typer.BadParameter(
                f"Camera '{camera_id}' not found in {resolve_config_path(config_path)}."
            )
        model_name = model or config.global_settings.model_name
        initial_conf = conf if conf is not None else camera.detection_conf_threshold
        provider = _build_camera_provider(config, camera, FileFrameProvider, LiveFrameProvider)

    alias_classes, alias_nms_iou = resolve_dog_aliases(dog_alias_classes, config)
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
