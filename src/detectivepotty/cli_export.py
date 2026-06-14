"""Dataset, CoreML, and experiment export CLI commands."""

from __future__ import annotations

from pathlib import Path
import time
from typing import Annotated, Optional

import typer

from detectivepotty.cli_common import DogAliasOption, resolve_dog_aliases


def register_export_commands(app: typer.Typer) -> None:
    app.command("export-coreml")(export_coreml_command)
    app.command("export-dataset")(export_dataset_command)
    app.command("experiment-bakeoff")(experiment_bakeoff_command)


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
    from detectivepotty.detect.yolo import DogDetector

    alias_classes, alias_nms_iou = resolve_dog_aliases(dog_alias_classes)
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

    from detectivepotty.detect.yolo import DogDetector
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

    alias_classes, alias_nms_iou = resolve_dog_aliases(dog_alias_classes)
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
