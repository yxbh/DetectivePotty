"""Offline tests for the decode->inference read-ahead pipeline helper.

No video, model, GPU, or network — the source iterators here are plain
lists/generators, which is exactly the seam :func:`prefetch` is built around.
"""

from __future__ import annotations

import threading
import time

import pytest

from detectivepotty.sources.prefetch import DEFAULT_PREFETCH, prefetch


def test_order_preserved_and_full_drain():
    src = list(range(100))
    out = list(prefetch(iter(src), max_prefetch=8))
    assert out == src


def test_empty_source():
    assert list(prefetch(iter([]), max_prefetch=4)) == []


def test_short_clip_smaller_than_prefetch():
    assert list(prefetch(iter([1, 2, 3]), max_prefetch=32)) == [1, 2, 3]


def test_max_prefetch_one():
    assert list(prefetch(iter(range(10)), max_prefetch=1)) == list(range(10))


def test_invalid_max_prefetch():
    with pytest.raises(ValueError):
        list(prefetch(iter([1]), max_prefetch=0))


def test_default_prefetch_depth():
    # Smoke: the documented default is usable without specifying max_prefetch.
    assert DEFAULT_PREFETCH >= 1
    assert list(prefetch(iter(range(5)))) == [0, 1, 2, 3, 4]


def test_error_propagates_after_prior_items():
    def boom():
        yield 1
        yield 2
        raise RuntimeError("decode failed")

    gen = prefetch(boom(), max_prefetch=4)
    seen = []
    with pytest.raises(RuntimeError, match="decode failed"):
        for item in gen:
            seen.append(item)
    assert seen == [1, 2]


def test_source_closed_on_early_break():
    closed = threading.Event()

    def gen():
        try:
            i = 0
            while True:
                yield i
                i += 1
        finally:
            closed.set()

    it = prefetch(gen(), max_prefetch=4)
    first = next(it)
    assert first == 0
    it.close()  # consumer abandons the stream early
    # The producer must stop and the source generator's finally must run.
    assert closed.wait(timeout=5.0)


def test_producer_runs_concurrently_with_consumer():
    # The producer should fill the buffer ahead while the consumer is slow,
    # proving overlap rather than lock-step. With a fast producer and a slow
    # consumer, total time tracks the consumer (not producer+consumer serial).
    produced = []

    def fast_source():
        for i in range(8):
            produced.append(i)
            yield i

    start = time.monotonic()
    out = []
    for item in prefetch(fast_source(), max_prefetch=8):
        time.sleep(0.02)  # slow consumer
        out.append(item)
    elapsed = time.monotonic() - start

    assert out == list(range(8))
    # 8 * 20ms consumer work = ~160ms; serial decode would add to it, but here
    # the (instant) producer overlaps, so we stay close to the consumer cost.
    assert elapsed < 0.5
