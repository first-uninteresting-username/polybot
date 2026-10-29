"""Replay backtester: feeds recorded ``(recv_ms, source, raw)`` tuples through the same Engine."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from .config import Settings
from .engine import Engine
from .recorder import read_recording
from .storage import Storage


def replay(settings: Settings, records: Iterable[tuple[int, str, str]], db_path: str | Path = ":memory:") -> Engine:
    store = Storage(db_path)
    first = True
    engine = Engine(settings, store)
    for recv_ms, source, raw in records:
        if first:
            store.set_meta("first_start_ms", str(recv_ms), only_if_missing=True)
            first = False
        engine.on_raw(source, recv_ms, raw)
    return engine


def iter_files(files: Iterable[Path]) -> Iterable[tuple[int, str, str]]:
    for f in files:
        yield from read_recording(f)
