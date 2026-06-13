"""Focused tests for CLI helper behavior."""

from __future__ import annotations

from concurrent.futures import Future

import pytest

from detectivepotty import cli


def test_protect_download_result_cancels_timed_out_future(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    future: Future[object] = Future()
    monkeypatch.setattr(cli, "_PROTECT_DOWNLOAD_TIMEOUT_S", 0.0)

    with pytest.raises(TimeoutError, match="Protect recording export timed out"):
        cli._protect_download_result(future)

    assert future.cancelled()
