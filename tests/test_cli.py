"""Focused tests for CLI helper behavior."""

from __future__ import annotations

from concurrent.futures import Future
from datetime import datetime, timezone

import pytest
import typer
from typer.testing import CliRunner

from detectivepotty import cli
from detectivepotty.cli_harvest import _resolve_harvest_window


def test_protect_download_result_cancels_timed_out_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future: Future[object] = Future()
    monkeypatch.setattr(cli, "_PROTECT_DOWNLOAD_TIMEOUT_S", 0.0)

    with pytest.raises(TimeoutError, match="Protect recording export timed out"):
        cli._protect_download_result(future)

    assert future.cancelled()


def test_list_cameras_unconfigured_exits_nonzero(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("global:\n  dataset_dir: dataset\n", encoding="utf-8")
    runner = CliRunner()

    result = runner.invoke(cli.app, ["list-cameras", "--config", str(config_path)])

    assert result.exit_code == 1
    assert "Protect is not configured" in result.output


def test_resolve_harvest_window_rejects_lone_start_with_date() -> None:
    with pytest.raises(typer.BadParameter, match="both --start and --end"):
        _resolve_harvest_window(
            "2026-06-06",
            "2026-06-06T01:00:00+00:00",
            None,
            0,
        )


def test_resolve_harvest_window_start_end_override_date() -> None:
    start, end = _resolve_harvest_window(
        "2026-06-06",
        "2026-06-07T01:00:00+00:00",
        "2026-06-07T02:00:00+00:00",
        10,
    )

    assert start == datetime(2026, 6, 7, 1, tzinfo=timezone.utc)
    assert end == datetime(2026, 6, 7, 2, tzinfo=timezone.utc)
