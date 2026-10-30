"""Live paper trading and replay must make identical decisions on identical data.

The live side runs the real ``LiveApp.consume`` loop (asyncio queue -> recorder -> engine). The replay side
reads the zstd recording the live side wrote and runs ``polybot.replay.replay``. Every table must match.
"""

import asyncio
from pathlib import Path

from synth import generate

from polybot.app import LiveApp
from polybot.config import load_settings
from polybot.recorder import read_recording, recording_files
from polybot.replay import iter_files, replay

TABLES = {
    "signals": "SELECT * FROM signals ORDER BY id",
    "fills": "SELECT * FROM fills ORDER BY id",
    "settlements": "SELECT * FROM settlements ORDER BY id",
    "predictions": "SELECT * FROM predictions ORDER BY slug, ts_ms",
    "markets": "SELECT slug, strike, winner, settled_by FROM markets ORDER BY slug",
}


async def _run_live(tmp: Path, records: list[tuple[int, str, str]], **kw: object) -> LiveApp:
    s = load_settings(env={}, data_dir=str(tmp / "live"), **kw)
    app = LiveApp(s)
    task = asyncio.create_task(app.consume())
    for t, src, raw in records:  # the feeds' put() calls, in arrival order
        await app.queue.put((src, t, raw))
    while not app.queue.empty():
        await asyncio.sleep(0)
    await asyncio.sleep(0)
    task.cancel()
    app.recorder.close()
    return app


def _dump(store) -> dict[str, list[tuple]]:
    return {k: [tuple(r) for r in store.query(q)] for k, q in TABLES.items()}


async def test_live_equals_replay_synthetic(tmp_path: Path) -> None:
    kw = dict(min_vol_samples=60, max_model_market_gap=0)
    records = generate()
    app = await _run_live(tmp_path, records, **kw)
    live = _dump(app.store)
    assert live["fills"], "test stream must produce trades to be meaningful"
    assert live["settlements"]

    files = recording_files(tmp_path / "live" / "recordings")
    recorded = list(iter_files(files))
    assert recorded == [(t, s, r) for t, s, r in records]  # recorder is lossless

    eng = replay(load_settings(env={}, **kw), iter_files(files))
    assert _dump(eng.store) == live


async def test_live_equals_replay_real_capture(tmp_path: Path) -> None:
    """25 s of a real live capture (2026-10-29) through both paths."""
    files = recording_files(Path(__file__).parent / "fixtures" / "live_slice")
    records = [(t, s, r) for t, s, r in read_recording(files[0])]
    assert len(records) > 1000
    app = await _run_live(tmp_path, records, min_vol_samples=10)
    live = _dump(app.store)
    live_snapshot = app.engine.snapshot()
    eng = replay(load_settings(env={}, min_vol_samples=10), iter_files(recording_files(tmp_path / "live" / "recordings")))
    assert _dump(eng.store) == live
    assert eng.snapshot()["counters"] == live_snapshot["counters"]
    assert eng.snapshot()["markets"] == live_snapshot["markets"]
