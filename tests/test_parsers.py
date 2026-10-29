"""Parsers against real responses captured on 2026-10-29 (tests/fixtures)."""

import json
from pathlib import Path

from polybot.events import (
    BookDelta,
    BookSnapshot,
    MarketInfo,
    OracleTick,
    ParseStats,
    PriceTick,
    Resolution,
    parse_raw,
)

FIX = Path(__file__).parent / "fixtures"


def load(name: str) -> list[dict]:
    return json.load(open(FIX / f"{name}.json"))


def run(name: str, source: str) -> tuple[list, ParseStats]:
    st = ParseStats()
    evs = []
    for x in load(name):
        if "raw" in x:
            evs += parse_raw(source, x["recv_ms"], x["raw"], st)
    return evs, st


def test_gamma_market() -> None:
    m = load("gamma_market_live")[0]
    (mi,) = parse_raw("gamma_market", 1, json.dumps(m))
    assert isinstance(mi, MarketInfo)
    assert mi.slug == "btc-updown-5m-1793304300" and mi.start_ts == 1793304300 and mi.end_ts == mi.start_ts + 300
    toks = json.loads(m["clobTokenIds"])
    assert json.loads(m["outcomes"]) == ["Up", "Down"]
    assert (mi.token_up, mi.token_down) == (toks[0], toks[1])
    assert mi.fees_enabled and mi.fee_rate == 0.07 and mi.fee_exponent == 1
    assert mi.tick_size == 0.01 and mi.min_size == 5
    assert mi.twap_lookback_s == 60 and "twap" in mi.resolution_source


def test_rules_text_confirms_twap_and_ties_up() -> None:
    d = load("gamma_market_live")[0]["description"]
    assert "time-weighted average price (TWAP)" in d and "greater than or equal to" in d and "Chainlink" in d


def test_resolution_closed_event() -> None:
    e = load("gamma_event_closed")[0]
    (r,) = parse_raw("resolution", 1, json.dumps(e))
    assert isinstance(r, Resolution)
    assert r.winner == "Up" and r.price_to_beat == 80782.8690547537 and r.final_price == 80794.92107026707


def test_resolution_ignores_open_event() -> None:
    e = load("gamma_event_live")[0]
    assert parse_raw("resolution", 1, json.dumps(e)) == []


def test_clob_ws() -> None:
    evs, st = run("ws_clob_market", "clob")
    assert st.errors == 0
    snaps = [e for e in evs if isinstance(e, BookSnapshot)]
    deltas = [e for e in evs if isinstance(e, BookDelta)]
    assert len(snaps) == 2 and len(deltas) > 50
    assert all(d.side in ("BUY", "SELL") for d in deltas)
    assert all(0 < p < 1 for s in snaps for p, _ in s.asks)


def test_rtds_spot_and_twap() -> None:
    evs, st = run("ws_rtds_twap60", "rtds")
    assert st.errors == 0 and evs and all(isinstance(e, OracleTick) and e.kind == "twap60" for e in evs)
    upd = [e for e in evs if e.recv_ms > evs[0].recv_ms]
    # E18 full_accuracy_value agrees with the float value
    raw = [json.loads(x["raw"]) for x in load("ws_rtds_twap60") if x["raw"].startswith('{"connection')][0]
    assert abs(int(raw["payload"]["full_accuracy_value"]) / 1e18 - raw["payload"]["value"]) < 1e-6
    assert upd
    evs2, st2 = run("ws_rtds_chainlink", "rtds")
    assert st2.errors == 0 and evs2 and all(e.kind == "spot" for e in evs2)  # snapshot topic is "crypto_prices"
    assert 10_000 < evs2[0].value < 1_000_000


def test_exchanges() -> None:
    b, sb = run("ws_binance_trade", "binance")
    c, sc = run("ws_coinbase", "coinbase")
    assert sb.errors == sc.errors == 0
    assert b and c and all(isinstance(e, PriceTick) for e in b + c)
    assert abs(b[0].price - c[0].price) / b[0].price < 0.01
    assert all(e.recv_ms >= e.exch_ms - 1000 for e in b)


def test_garbage_does_not_raise() -> None:
    st = ParseStats()
    assert parse_raw("clob", 1, "PONG", st) == []
    assert parse_raw("clob", 1, '{"event_type":"book"}', st) == []
    assert parse_raw("rtds", 1, "", st) == []
    assert st.errors == 1
