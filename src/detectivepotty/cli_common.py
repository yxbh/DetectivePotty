"""Shared helpers for Typer CLI command modules."""

from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeoutError
import logging
from pathlib import Path
from typing import Annotated, Optional

import typer

from detectivepotty.config import (
    CONFIG_ENV_VAR,
    DEFAULT_CONFIG_PATH,
    Config,
    DEFAULT_DOG_ALIAS_CLASSES,
    load_config,
    resolve_config_path,
)

PROTECT_DOWNLOAD_TIMEOUT_S = 30 * 60.0
LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
CONFIG_HELP = (
    "Path to DetectivePotty YAML config. Defaults to "
    f"${CONFIG_ENV_VAR}, then {DEFAULT_CONFIG_PATH}."
)

_cli_log_level_override: str | None = None

ConfigPathOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        exists=True,
        dir_okay=False,
        readable=True,
        help=CONFIG_HELP,
    ),
]

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


def resolve_dog_aliases(
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


def protect_download_result(future):
    try:
        return future.result(timeout=PROTECT_DOWNLOAD_TIMEOUT_S)
    except FutureTimeoutError as exc:
        future.cancel()
        raise TimeoutError(
            "Protect recording export timed out "
            f"after {PROTECT_DOWNLOAD_TIMEOUT_S:.0f}s"
        ) from exc


def configure_cli_logging(level_name: str) -> None:
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(level=level, format=LOG_FORMAT)
    else:
        root.setLevel(level)
    logging.getLogger("detectivepotty").setLevel(level)


def set_cli_log_level(log_level: str | None) -> None:
    global _cli_log_level_override
    _cli_log_level_override = log_level
    configure_cli_logging(log_level or "INFO")


def load_cli_config(config_path: Path | None) -> Config:
    resolved = resolve_config_path(config_path)
    try:
        config = load_config(resolved)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"Config file not found: {resolved}") from exc
    configure_cli_logging(_cli_log_level_override or config.global_settings.log_level)
    return config
