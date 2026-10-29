"""Engine: signals, fills, settlement on Polymarket's outcome, fallback, idempotency, restart recovery."""

import json

from synth import START, gamma_market, generate, resolution_event

from polybot.config import load_settings
from polybot.engine import Engine
from polybot.replay import replay
from polybot.storage import Storage

S = load_settings(env={}, min_vol_samples=60, fallback_after_s=600)


def run(records, store=None):
    store = store or Storage(":memory:")
    e = Engine(S, store)
    for t, s, r in records:
        e.on_raw(s, t, r)
    return e


def test_trades_and_settles_on_gamma_outcome() -> None:
    e = run(generate())
    fills = e.store.query("SELECT * FROM fills")
    assert fills, e.counters
    f = fills[0]
    assert f["side"] == "Up"
    sig = e.store.query("SELECT * FROM signals WHERE id=?", (f["id"],))[0]
    assert f["ts_ms"] >= sig["arrival_ms"] >= sig["ts_ms"] + S.latency_ms_floor  # filled after latency
    assert f["avg_price"] <= sig["limit_price"] + 1e-9
    st = e.store.query("SELECT * FROM settlements")[0]
    assert st["settled_by"] == "gamma" and st["won"] == 1
    assert abs(st["payout"] - f["net_shares"]) < 1e-9 and abs(st["pnl"] - (f["net_shares"] - f["cost"])) < 1e-9
    m = e.store.market(f["slug"])
    assert m["strike"] is not None and abs(m["strike"] - m["gamma_strike"]) < 1e-6  # synthetic strike == ptb


def test_settles_on_official_outcome_even_if_prices_disagree() -> None:
    e = run(generate(winner="Down"))  # Polymarket says Down although our TWAP says Up
    st = e.store.query("SELECT * FROM settlements")
    assert st and all(r["won"] == 0 and r["payout"] == 0 for r in st)


def test_resolution_twice_is_idempotent() -> None:
    recs = generate()
    res = [r for r in recs if r[1] == "resolution"][0]
    e = run(recs + [(res[0] + 5000, "resolution", res[2])])
    n_fills = len(e.store.query("SELECT * FROM fills"))
    assert len(e.store.query("SELECT * FROM settlements")) == n_fills
    assert abs(e.store.cash(S.starting_bankroll) - (S.starting_bankroll + sum(
        r["pnl"] for r in e.store.query("SELECT pnl FROM settlements")))) < 1e-9


def test_fallback_then_official_override() -> None:
    recs = [r for r in generate() if r[1] != "resolution"]
    end = recs[-1][0]
    # keep event time moving past end + fallback_after_s without any resolution
    later = [(end + k * 1000, "latency", json.dumps({"rtt_ms": 100})) for k in range(1, 700)]
    e = run(recs + later)
    st = e.store.query("SELECT * FROM settlements")
    assert st and all(r["settled_by"] in ("fallback", "fallback_loss") for r in st)
    # official result arrives later and overrides the provisional one
    m = gamma_market()
    ev = resolution_event(m, "Down", 80000.0, 79000.0)
    e.on_raw("resolution", later[-1][0] + 1000, json.dumps(ev))
    st2 = e.store.query("SELECT * FROM settlements")
    assert all(r["settled_by"] == "gamma" and r["won"] == 0 for r in st2)
    assert len(st2) == len(st)


def test_restart_recovery_no_double_count(tmp_path) -> None:
    recs = generate()
    res_i = next(i for i, r in enumerate(recs) if r[1] == "resolution")
    db = tmp_path / "s.sqlite"
    e1 = run(recs[:res_i], Storage(db))
    n_fills = len(e1.store.query("SELECT * FROM fills"))
    assert n_fills >= 1
    e1.store.close()
    # "restart": new engine, same DB; replays nothing, receives only the resolution
    e2 = Engine(S, Storage(db))
    assert e2.open_slugs == {f"btc-updown-5m-{START}"}
    for t, s, r in recs[res_i:] + recs[res_i:]:  # resolution delivered twice across the restart
        e2.on_raw(s, t, r)
    assert len(e2.store.query("SELECT * FROM fills")) == n_fills
    assert len(e2.store.query("SELECT * FROM settlements")) == n_fills


def test_pending_orders_cancelled_on_restart(tmp_path) -> None:
    recs = generate()
    db = tmp_path / "s.sqlite"
    store = Storage(db)
    e = Engine(S, store)
    first_signal_at = None
    for t, s, r in recs:
        e.on_raw(s, t, r)
        if e.pending:
            first_signal_at = t
            break
    assert first_signal_at is not None
    store.close()
    e2 = Engine(S, Storage(db))
    rows = e2.store.query("SELECT status FROM signals")
    assert [r["status"] for r in rows] == ["cancelled_restart"]
    assert e2.store.query("SELECT * FROM fills") == []


def test_no_trading_without_latency_samples() -> None:
    recs = [r for r in generate() if r[1] != "latency"]
    e = run(recs)
    assert e.store.query("SELECT * FROM signals") == []
    assert e.counters.get("skip_no_latency_samples", 0) > 0


def test_stale_feed_blocks_trading() -> None:
    recs = [r for r in generate() if not (r[1] == "coinbase" and r[0] > (START + 5) * 1000)]
    e = run(recs)
    assert e.store.query("SELECT * FROM signals") == []


def test_replay_function_matches_direct() -> None:
    recs = generate()
    a = run(recs)
    b = replay(S, recs)
    q = "SELECT * FROM fills ORDER BY id"
    assert [tuple(r) for r in a.store.query(q)] == [tuple(r) for r in b.store.query(q)]
