"""Focused tests for CLI helper behavior."""

from __future__ import annotations

from concurrent.futures import Future

import pytest
from typer.testing import CliRunner

from detectivepotty import cli


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
