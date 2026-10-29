"""Raw-message recorder and reader.

Every message the engine sees is appended as one JSON line ``[recv_ms, source, raw]`` to
``<data>/recordings/YYYY-MM-DD.jsonl.zst`` (UTC date of recv_ms). Files are zstd-compressed and flushed
as complete frames, so a crash loses at most the last ``flush_interval_s`` of data and never corrupts
earlier frames. A restart appends a new frame to the same day's file.
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO

import zstandard as zstd

log = logging.getLogger("polybot.recorder")


def day_of(recv_ms: int) -> str:
    return datetime.fromtimestamp(recv_ms / 1000, tz=UTC).strftime("%Y-%m-%d")


class Recorder:
    def __init__(self, directory: str | Path, flush_interval_s: float = 2.0, retention_days: int = 0):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.flush_interval_s = flush_interval_s
        self.retention_days = retention_days
        self._day: str | None = None
        self._fh: BinaryIO | None = None
        self._writer: zstd.ZstdCompressionWriter | None = None
        self._last_flush = time.monotonic()
        self.lines = 0
        self.bytes_in = 0

    def write(self, recv_ms: int, source: str, raw: str) -> None:
        day = day_of(recv_ms)
        if day != self._day:
            self._rotate(day)
        line = json.dumps([recv_ms, source, raw], separators=(",", ":")).encode() + b"\n"
        assert self._writer is not None
        self._writer.write(line)
        self.lines += 1
        self.bytes_in += len(line)
        if time.monotonic() - self._last_flush >= self.flush_interval_s:
            self.flush()

    def flush(self) -> None:
        if self._writer is not None and self._fh is not None:
            self._writer.flush(zstd.FLUSH_FRAME)
            self._fh.flush()
            os.fsync(self._fh.fileno())
        self._last_flush = time.monotonic()

    def _rotate(self, day: str) -> None:
        self.close()
        self._day = day
        self._fh = open(self.dir / f"{day}.jsonl.zst", "ab")  # noqa: SIM115 - long-lived handle
        self._writer = zstd.ZstdCompressor(level=6).stream_writer(self._fh, closefd=False)
        self._apply_retention()

    def _apply_retention(self) -> None:
        if self.retention_days <= 0:
            return
        cutoff = time.time() - self.retention_days * 86400
        for p in self.dir.glob("*.jsonl.zst"):
            if p.stat().st_mtime < cutoff:
                log.info("retention: deleting old recording %s", p.name)
                p.unlink()

    def close(self) -> None:
        if self._writer is not None:
            self.flush()
            self._writer.close()
        if self._fh is not None:
            self._fh.close()
        self._writer = None
        self._fh = None


def read_recording(path: str | Path) -> Iterator[tuple[int, str, str]]:
    """Yield ``(recv_ms, source, raw)``; tolerates a truncated last frame or line."""
    with open(path, "rb") as fh:
        reader = zstd.ZstdDecompressor().stream_reader(fh, read_across_frames=True)
        text = io.TextIOWrapper(io.BufferedReader(reader), encoding="utf-8")
        try:
            for line in text:
                try:
                    recv_ms, source, raw = json.loads(line)
                except (ValueError, TypeError):
                    log.warning("skipping truncated/corrupt line in %s", path)
                    continue
                yield int(recv_ms), source, raw
        except zstd.ZstdError as exc:
            log.warning("recording %s ends with an incomplete frame: %s", path, exc)


def recording_files(directory: str | Path, start_day: str | None = None, end_day: str | None = None) -> list[Path]:
    files = sorted(Path(directory).glob("*.jsonl.zst"))
    return [f for f in files if (start_day is None or f.name[:10] >= start_day)
            and (end_day is None or f.name[:10] <= end_day)]
