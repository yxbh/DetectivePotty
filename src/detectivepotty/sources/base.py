"""Base contracts for live and file-backed video sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Self
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

SENSITIVE_QUERY_KEYS = {
    "access_token",
    "api_key",
    "apikey",
    "auth",
    "key",
    "password",
    "pass",
    "sig",
    "signature",
    "token",
    "x-api-key",
}


def sanitize_source_id(source_id: str) -> str:
    """Strip URL credentials and sensitive query tokens from a source id."""

    parts = urlsplit(source_id)
    if not parts.scheme or not parts.netloc:
        return source_id

    netloc = parts.netloc.rsplit("@", maxsplit=1)[-1]
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if key.lower() not in SENSITIVE_QUERY_KEYS
    ]
    return urlunsplit((parts.scheme, netloc, parts.path, urlencode(query), ""))


@dataclass(slots=True)
class Frame:
    """A decoded frame at the original source resolution."""

    bgr: np.ndarray
    frame_idx: int
    mono_ts: float
    wall_ts: datetime
    source_id: str
    width: int = 0
    height: int = 0

    def __post_init__(self) -> None:
        if self.bgr.ndim < 2:
            raise ValueError("bgr must be an image array")
        self.height, self.width = self.bgr.shape[:2]
        self.source_id = sanitize_source_id(self.source_id)
        if self.wall_ts.tzinfo is None:
            self.wall_ts = self.wall_ts.replace(tzinfo=timezone.utc)
        else:
            self.wall_ts = self.wall_ts.astimezone(timezone.utc)


class VideoSource(ABC):
    """Frame source contract.

    Live sources should support a latest-frame-only mode in their concrete
    implementation: when consumers lag, old frames may be dropped so reads prefer
    current frames over stale buffered frames. Live implementations are also
    expected to reconnect with backoff and return ``None`` only for terminal EOF
    or shutdown, not for transient stalls.
    """

    @abstractmethod
    def open(self) -> Self:
        raise NotImplementedError

    @abstractmethod
    def read(self) -> Frame | None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    @property
    @abstractmethod
    def fps(self) -> float | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int] | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def is_live(self) -> bool:
        raise NotImplementedError

    def __enter__(self) -> Self:
        return self.open()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()
