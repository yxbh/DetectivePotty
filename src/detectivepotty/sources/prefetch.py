"""Background read-ahead for decode->inference pipelining.

A serial ``for frame in decode(): infer(frame)`` loop underuses the GPU: while a
batch is running inference the decoder sits idle, and while the decoder produces the
next frames the GPU sits idle. :func:`prefetch` turns that into a pipeline — a
producer thread pulls from the source iterator into a bounded queue while the caller
consumes from it. Because PyAV decode (C/libav) and CoreML/torch inference both
release the GIL, the two genuinely overlap, so a decode-bound dense pass becomes
inference-bound instead.

Contract:

* **Order preserved** — items come out in exactly the source order.
* **Bounded memory** — at most ``max_prefetch`` items are buffered ahead (full-res
  frames are large, so this cap matters).
* **Full drain** — every source item is yielded before ``StopIteration``.
* **Exception propagation** — an error raised by the source surfaces in the consumer
  after the items produced before it.
* **Prompt cleanup** — on early consumer exit (``break``/exception) the producer is
  stopped and the source iterator is ``close()``d (releasing capture handles), and
  the producer thread is always joined.

The helper is decode-backend agnostic (any iterable works), so it is exercised
offline with plain lists / generators — no video, model, GPU, or network.
"""

from __future__ import annotations

import queue
import threading
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

_DONE = object()
"""Sentinel pushed by the producer to mark end-of-stream."""

DEFAULT_PREFETCH = 32
"""Default read-ahead depth (~one inference batch buffered while another runs)."""


def prefetch(iterable: Iterable[T], *, max_prefetch: int = DEFAULT_PREFETCH) -> Iterator[T]:
    """Yield ``iterable`` items while a background thread reads ahead.

    ``max_prefetch < 1`` is rejected. ``max_prefetch`` is the maximum number of
    items buffered ahead of the consumer. The source iterator runs on a daemon
    thread; the consumer is the calling thread.
    """

    if max_prefetch < 1:
        raise ValueError("max_prefetch must be >= 1")

    q: "queue.Queue[object]" = queue.Queue(maxsize=max_prefetch)
    error: list[BaseException] = []
    stop = threading.Event()

    def _put(item: object) -> bool:
        """Block until ``item`` is queued, or ``stop`` is set. Return success."""
        while not stop.is_set():
            try:
                q.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _producer() -> None:
        try:
            for item in iterable:
                if not _put(item):
                    return
        except BaseException as exc:  # noqa: BLE001 - propagated to consumer
            error.append(exc)
        finally:
            closer = getattr(iterable, "close", None)
            if callable(closer):
                try:
                    closer()
                except BaseException:  # noqa: BLE001 - best-effort cleanup
                    pass
            _put(_DONE)

    thread = threading.Thread(target=_producer, name="prefetch-decode", daemon=True)
    thread.start()
    try:
        while True:
            item = q.get()
            if item is _DONE:
                break
            yield item  # type: ignore[misc]
        if error:
            raise error[0]
    finally:
        stop.set()
        # Unblock a producer that may be parked on a full queue.
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        thread.join(timeout=5.0)


__all__ = ["prefetch", "DEFAULT_PREFETCH"]
