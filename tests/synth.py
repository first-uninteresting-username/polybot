"""Synthetic raw-message stream in the exact wire formats of the saved live fixtures.

Used where the test needs a complete, deterministic window (signals, fills, resolution), which a short
live capture cannot guarantee.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path

FIX = Path(__file__).parent / "fixtures"
START = 1793304300  # window start (unix s), matches the saved gamma market fixture


def gamma_market(start: int = START) -> dict:
    m = json.loads((FIX / "gamma_market_live.json").read_text())[0]
    m = dict(m)
    m["slug"] = f"btc-updown-5m-{start}"
    m["endDate"] = datetime.fromtimestamp(start + 300, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return m


def tokens(m: dict) -> tuple[str, str]:
    t = json.loads(m["clobTokenIds"])
    return t[0], t[1]  # outcomes ["Up","Down"]


def resolution_event(m: dict, winner: str, ptb: float, final: float) -> dict:
    e = json.loads((FIX / "gamma_event_closed.json").read_text())[0]
    e = dict(e)
    mk = dict(e["markets"][0])
    mk["slug"] = m["slug"]
    mk["outcomePrices"] = json.dumps(["1", "0"] if winner == "Up" else ["0", "1"])
    mk["closed"] = True
    mk["umaResolutionStatus"] = "resolved"
    e["slug"] = m["slug"]
    e["markets"] = [mk]
    e["eventMetadata"] = {"priceToBeat": ptb, "finalPrice": final}
    return e


def book_msg(m: dict, asset: str, t_ms: int, bids: list[tuple[float, float]], asks: list[tuple[float, float]]) -> str:
    return json.dumps([{
        "market": m["conditionId"], "asset_id": asset, "timestamp": str(t_ms), "hash": "x",
        "bids": [{"price": f"{p:.2f}", "size": f"{s}"} for p, s in bids],
        "asks": [{"price": f"{p:.2f}", "size": f"{s}"} for p, s in asks],
        "event_type": "book",
    }])


def price_change(m: dict, asset: str, t_ms: int, side: str, price: float, size: float) -> str:
    return json.dumps({"market": m["conditionId"], "price_changes": [
        {"asset_id": asset, "price": f"{price:.2f}", "size": f"{size}", "side": side, "hash": "y",
         "best_bid": "0.4", "best_ask": "0.6"}], "timestamp": str(t_ms), "event_type": "price_change"})


def generate(seed: int = 7, drift_per_s: float = 2.0, winner: str | None = None,
             resolve: bool = True, warmup_s: int = 200) -> list[tuple[int, str, str]]:
    """Return a time-ordered list of ``(recv_ms, source, raw)``.

    BTC drifts up by ``drift_per_s`` USD/s from the strike while the Up book stays offered at 0.55, so
    a correct model should find edge buying Up.
    """
    rng = random.Random(seed)
    m = gamma_market()
    up, down = tokens(m)
    out: list[tuple[int, str, str]] = []
    t0 = (START - warmup_s) * 1000
    t_end = (START + 300 + 5) * 1000
    price = 80_000.0
    spots: dict[int, float] = {}
    out.append((t0, "gamma_market", json.dumps(m)))
    for k in range(8):
        out.append((t0 + 100 + k * 1000, "latency", json.dumps({"rtt_ms": 100.0 + 10 * k})))
    strike = None
    t = t0
    while t <= t_end:
        sec = t // 1000
        if t % 1000 == 0:
            if sec >= START:
                price += drift_per_s + rng.gauss(0, 4)
            else:
                price += rng.gauss(0, 4)
            spots[sec] = price - 3.0  # Chainlink sits a little below exchanges
        p_b = price + rng.gauss(0, 1)
        p_c = price + 1.0 + rng.gauss(0, 1)
        out.append((t + 3, "binance", json.dumps({"e": "trade", "E": t, "s": "BTCUSDT", "t": t, "p": f"{p_b:.2f}",
                                                   "q": "0.01", "T": t, "m": False, "M": True})))
        iso = datetime.fromtimestamp(t / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        out.append((t + 7, "coinbase", json.dumps({"type": "ticker", "sequence": t, "product_id": "BTC-USD",
                                                    "price": f"{p_c:.2f}", "best_bid": "0", "best_ask": "0",
                                                    "side": "buy", "time": iso, "trade_id": t})))
        if t % 1000 == 0 and sec - 1 in spots:
            s = sec - 1
            cl = spots[s]
            window = [spots[k] for k in range(s - 59, s + 1) if k in spots]
            tw = sum(window) / len(window)
            if s == START:
                strike = tw
            for topic, val, extra in (("crypto_prices_chainlink", cl, {}), ("crypto_prices_twap_sixty", tw, {"window_s": 60})):
                payload = {"full_accuracy_value": str(int(val * 1e18)), "symbol": "btc/usd", "timestamp": s * 1000,
                           "value": val, **extra}
                out.append((t + 450, "rtds", json.dumps({"connection_id": "c", "payload": payload, "timestamp": t + 400,
                                                         "topic": topic, "type": "update"})))
        if t == (START - 30) * 1000 or (t % 20000 == 0 and t >= START * 1000):
            out.append((t + 20, "clob", book_msg(m, up, t, [(0.50, 100)], [(0.55, 40), (0.56, 200), (0.60, 500)])))
            out.append((t + 21, "clob", book_msg(m, down, t, [(0.44, 100)], [(0.48, 300), (0.50, 500)])))
        elif t % 3000 == 1000 and t >= START * 1000:
            out.append((t + 30, "clob", price_change(m, up, t, "SELL", 0.55, 40 + (t // 1000) % 7)))
        t += 250
    final_sec = START + 300
    final = sum(spots[k] for k in range(final_sec - 59, final_sec + 1) if k in spots) / 60
    if resolve:
        assert strike is not None
        w = winner or ("Up" if final >= strike else "Down")
        out.append((t_end + 60_000, "resolution", json.dumps(resolution_event(m, w, strike, final))))
        # keep the clock moving after the resolution so evaluation runs
        out.append((t_end + 61_000, "latency", json.dumps({"rtt_ms": 120.0})))
    out.sort(key=lambda x: x[0])
    return out

