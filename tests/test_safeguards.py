"""Book reconciliation, robust edge (sd band), model-vs-market guard, Kelly sizing and the account reset."""

import json
from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer
from synth import START, generate

from polybot.book import OrderBook
from polybot.config import load_settings
from polybot.engine import Engine
from polybot.events import BookDelta, BookSnapshot, parse_raw
from polybot.recorder import read_recording, recording_files
from polybot.storage import Storage
from polybot.web import make_app

MECH = dict(min_vol_samples=60, max_model_market_gap=0)  # settings the synthetic stream trades under


def run(records, store=None, **kw):
    e = Engine(load_settings(env={}, **{**MECH, **kw}), store or Storage(":memory:"))
    for t, s, r in records:
        e.on_raw(s, t, r)
    return e


# ---------------------------------------------------------------- order book


def _book() -> OrderBook:
    b = OrderBook("a")
    b.apply_snapshot(BookSnapshot("a", 0, 0, ((0.50, 10), (0.49, 10)), ((0.55, 10), (0.56, 10))))
    return b


def test_parses_reported_best_bid_ask() -> None:
    (ev,) = parse_raw("clob", 1, json.dumps({"market": "m", "event_type": "price_change", "timestamp": "1",
                                             "price_changes": [{"asset_id": "a", "price": "0.5", "size": "3",
                                                                "side": "BUY", "best_bid": "0.5", "best_ask": "1"}]}))
    assert isinstance(ev, BookDelta) and (ev.best_bid, ev.best_ask) == (0.5, 1.0)
    (ev,) = parse_raw("clob", 1, json.dumps({"market": "m", "event_type": "price_change", "timestamp": "1",
                                             "price_changes": [{"asset_id": "a", "price": "0.5", "size": "3",
                                                                "side": "BUY"}]}))
    assert (ev.best_bid, ev.best_ask) == (None, None)


def test_phantom_levels_are_dropped() -> None:
    b = _book()
    # The 0.55 ask was taken by a trade we only saw as a snapshot we missed; the exchange says best ask 0.56.
    b.apply_delta(BookDelta("a", 1, 1, "BUY", 0.49, 12, best_bid=0.50, best_ask=0.56))
    assert b.best_ask() == 0.56 and b.asks_ascending() == [(0.56, 10)] and not b.desynced
    b.apply_delta(BookDelta("a", 2, 2, "SELL", 0.56, 5, best_bid=0.49, best_ask=0.56))
    assert b.best_bid() == 0.49 and not b.desynced
    # empty sides are reported as 0 / 1
    b.apply_delta(BookDelta("a", 3, 3, "SELL", 0.56, 0, best_bid=0.0, best_ask=1.0))
    assert b.best_bid() is None and b.best_ask() is None and not b.desynced


def test_missing_level_flags_book_until_snapshot() -> None:
    b = _book()
    b.apply_delta(BookDelta("a", 1, 1, "BUY", 0.49, 12, best_bid=0.50, best_ask=0.54))  # we never saw 0.54
    assert b.desynced
    b.apply_snapshot(BookSnapshot("a", 2, 2, ((0.50, 10),), ((0.54, 5),)))
    assert not b.desynced


def test_reconstructed_book_matches_exchange_on_live_capture() -> None:
    files = recording_files(Path(__file__).parent / "fixtures" / "live_slice")
    books: dict[str, OrderBook] = {}
    checked = 0
    for t, s, r in read_recording(files[0]):
        if s != "clob":
            continue
        for ev in parse_raw(s, t, r):
            if isinstance(ev, BookSnapshot):
                books.setdefault(ev.asset_id, OrderBook(ev.asset_id)).apply_snapshot(ev)
            elif isinstance(ev, BookDelta) and ev.asset_id in books:
                b = books[ev.asset_id]
                b.apply_delta(ev)
                assert (b.best_bid() or 0.0) == ev.best_bid and (b.best_ask() or 1.0) == ev.best_ask
                assert not b.desynced
                checked += 1
    assert checked > 10_000


# ---------------------------------------------------------------- strategy


def test_band_brackets_central_prediction() -> None:
    first_minute = [r for r in generate(resolve=False) if r[0] < (START + 60) * 1000]
    e = run(first_minute)
    pred, lo, hi = e.predict_band(next(iter(e.markets.values())))
    assert 0.5 < lo < pred.p_up < hi < 1.0  # Up favoured: the wide end is the worst case for Up
    e1 = run(first_minute, model_sd_band=1.0)
    p1, lo1, hi1 = e1.predict_band(next(iter(e1.markets.values())))
    assert lo1 == p1.p_up == hi1


def test_wider_band_needs_more_evidence() -> None:
    (base,) = run(generate()).store.query("SELECT * FROM signals")
    (wide,) = run(generate(), model_sd_band=20.0).store.query("SELECT * FROM signals")
    assert wide["tau_s"] < base["tau_s"] - 30  # trades only once the move is far larger than the noise


def test_signal_records_central_p_and_worst_case_edge() -> None:
    e = run(generate())
    (sig,) = e.store.query("SELECT * FROM signals")
    from polybot.fills import all_in_cost_per_share

    assert sig["edge"] < sig["p_win"] - all_in_cost_per_share(sig["best_ask"], 0.07) - 1e-9


def test_model_far_from_market_does_not_trade() -> None:
    # Up book mid 0.525; any trade needs P(win) >= cost(0.55) + min_edge > 0.575 = mid + gap.
    e = run(generate(), max_model_market_gap=0.05)
    assert e.store.query("SELECT * FROM signals") == []
    assert e.counters.get("skip_model_market_gap", 0) > 0


def test_kelly_sizing() -> None:
    e = Engine(load_settings(env={}), Storage(":memory:"))  # equity 1000, stake_fraction 0.02, kelly 0.25
    assert abs(e.stake_budget(0.57, 0.55) - 1000 * 0.25 * (0.02 / 0.45)) < 1e-9  # small edge: below the cap
    assert e.stake_budget(0.90, 0.55) == 20.0  # large edge: capped at stake_fraction
    assert e.stake_budget(0.50, 0.55) == 0.0
    flat = Engine(load_settings(env={}, kelly_fraction=0), Storage(":memory:"))
    assert flat.stake_budget(0.57, 0.55) == 20.0


# ---------------------------------------------------------------- reset


def test_reset_wipes_account_but_keeps_markets() -> None:
    e = run(generate())
    assert e.store.query("SELECT * FROM fills") and e.store.cash(1000) != 1000
    e.store.reset_trading(123)
    e.reset_trading()
    for table in ("signals", "fills", "settlements", "predictions"):
        assert e.store.query(f"SELECT * FROM {table}") == []
    assert e.store.query("SELECT * FROM markets")
    assert e.store.cash(1000) == 1000 and e.equity() == 1000
    assert e.store.get_meta("first_start_ms") == "123" and e.store.get_meta("last_reset_ms") == "123"
    assert not e.pending and not e.open_slugs and e.counters == {}


async def test_reset_endpoint() -> None:
    calls: list[int] = []

    def reset() -> dict:
        calls.append(1)
        return {"ok": True}

    app = make_app(lambda: {"ok": True}, dict, dict, lambda n: [], reset, admin_token="s3cret")
    async with TestClient(TestServer(app)) as c:
        assert (await c.post("/api/reset", data="confirm=RESET")).status == 415  # plain form post (CSRF)
        assert (await c.post("/api/reset", json={"confirm": "RESET"})).status == 401
        r = await c.post("/api/reset", json={"confirm": "RESET"}, headers={"X-Admin-Token": "wrong"})
        assert r.status == 401 and (await r.json())["token_required"]
        hdr = {"X-Admin-Token": "s3cret"}
        assert (await c.post("/api/reset", json={}, headers=hdr)).status == 400
        assert (await c.get("/api/reset")).status == 405
        assert not calls
        r = await c.post("/api/reset", json={"confirm": "RESET"}, headers=hdr)
        assert r.status == 200 and (await r.json())["ok"] and calls == [1]
    open_app = make_app(lambda: {"ok": True}, dict, dict, lambda n: [], reset)
    async with TestClient(TestServer(open_app)) as c:
        assert (await c.post("/api/reset", json={"confirm": "RESET"})).status == 200


async def test_live_app_reset(tmp_path: Path) -> None:
    from polybot.app import LiveApp

    app = LiveApp(load_settings(env={}, data_dir=str(tmp_path), **MECH))
    for t, s, r in generate():
        app.pipeline.process(s, t, r)
    assert app.store.query("SELECT * FROM fills")
    out = app.reset()
    assert out["ok"] and app.store.query("SELECT * FROM fills") == []
    assert app.report()["bankroll"]["equity"] == 1000
    app.recorder.close()
    app.store.close()
