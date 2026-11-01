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


def price_change(m: dict, asset: str, t_ms: int, side: str, price: float, size: float,
                 best_bid: float, best_ask: float) -> str:
    """``best_bid``/``best_ask`` are the exchange's top of book after the change, as on the real feed."""
    return json.dumps({"market": m["conditionId"], "price_changes": [
        {"asset_id": asset, "price": f"{price:.2f}", "size": f"{size}", "side": side, "hash": "y",
         "best_bid": f"{best_bid:.2f}", "best_ask": f"{best_ask:.2f}"}], "timestamp": str(t_ms),
        "event_type": "price_change"})


def generate(seed: int = 7, drift_per_s: float = 2.0, winner: str | None = None,
             resolve: bool = True, warmup_s: int = 200) -> list[tuple[int, str, str]]:
    """Return a time-ordered list of ``(recv_ms, source, raw)``.

    BTC drifts up by ``drift_per_s`` USD/s from the strike while the Up book stays offered at 0.55, so
    a correct model should find edge buying Up. The market is deliberately far from the model, so tests of
    the trading mechanics turn the ``max_model_market_gap`` guard off.
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
            out.append((t + 30, "clob", price_change(m, up, t, "SELL", 0.55, 40 + (t // 1000) % 7, 0.50, 0.55)))
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



def _true_p_up(cl: dict[int, float], now: int, start: int, strike: float, sigma: float) -> float:
    """P(Up) from the synthetic world's own Chainlink path (driftless, known sigma per sqrt s)."""
    import math

    from polybot.model import norm_cdf

    end = start + 300
    tau = end - now
    x = math.log(cl[now])
    if tau >= 60:
        mean, var = x, sigma**2 * ((tau - 60) + 20.0)
    else:
        known = [math.log(cl[s]) for s in range(end - 59, now + 1) if s in cl]
        k = len(known) / 60
        mean = k * (sum(known) / len(known)) + (1 - k) * x
        var = (1 - k) ** 2 * sigma**2 * tau / 3.0
    return norm_cdf((mean - math.log(strike)) / math.sqrt(max(var, 1e-18)))


def generate_windows(n: int, seed: int = 11, market_noise: float = 0.6, vol: float = 4.0,
                     warmup_s: int = 200) -> list[tuple[int, str, str]]:
    """``n`` consecutive 5-minute windows (1 exchange tick per second) with resolutions.

    Both books are quoted 2 ticks wide around ``sigmoid(logit(true P(Up)) + noise)``, noise sd ``market_noise``
    redrawn at every quote, so the model's inputs carry information the market's prices only partly hold.
    """
    import math

    rng = random.Random(seed)
    out: list[tuple[int, str, str]] = []
    first = START
    t0 = (first - warmup_s) * 1000
    last_end = first + 300 * n
    price = 80_000.0
    sigma = vol / price
    cl: dict[int, float] = {}
    twap: dict[int, float] = {}
    windows = []
    for k in range(n):
        m = gamma_market(first + 300 * k)
        m["conditionId"] = f"0xcond{k}"
        m["clobTokenIds"] = json.dumps([f"{k + 1}01", f"{k + 1}02"])
        windows.append(m)
        out.append(((first + 300 * k - 250) * 1000, "gamma_market", json.dumps(m)))
    for j in range(8):
        out.append((t0 + 100 + j * 1000, "latency", json.dumps({"rtt_ms": 100.0 + 10 * j})))
    strikes: dict[int, float] = {}
    for sec in range(first - warmup_s, last_end + 5):
        t = sec * 1000
        price += rng.gauss(0, vol)
        cl[sec] = price - 3.0
        win = [cl[s] for s in range(sec - 59, sec + 1) if s in cl]
        twap[sec] = sum(win) / len(win)
        if sec >= first and (sec - first) % 300 == 0 and sec < last_end:
            strikes[sec] = twap[sec]
        p_b = price + rng.gauss(0, 1)
        p_c = price + 1.0 + rng.gauss(0, 1)
        out.append((t + 3, "binance", json.dumps({"e": "trade", "E": t, "s": "BTCUSDT", "t": t, "p": f"{p_b:.2f}",
                                                   "q": "0.01", "T": t, "m": False, "M": True})))
        iso = datetime.fromtimestamp(t / 1000, tz=UTC).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        out.append((t + 7, "coinbase", json.dumps({"type": "ticker", "sequence": t, "product_id": "BTC-USD",
                                                    "price": f"{p_c:.2f}", "best_bid": "0", "best_ask": "0",
                                                    "side": "buy", "time": iso, "trade_id": t})))
        s = sec - 1
        if s in cl:
            for topic, val, extra in (("crypto_prices_chainlink", cl[s], {}),
                                      ("crypto_prices_twap_sixty", twap[s], {"window_s": 60})):
                payload = {"full_accuracy_value": str(int(val * 1e18)), "symbol": "btc/usd", "timestamp": s * 1000,
                           "value": val, **extra}
                out.append((t + 450, "rtds", json.dumps({"connection_id": "c", "payload": payload,
                                                         "timestamp": t + 400, "topic": topic, "type": "update"})))
        if sec % 15 == 0:
            out.append((t + 500, "latency", json.dumps({"rtt_ms": 110.0 + rng.random() * 40})))
        k = (sec - first) // 300
        if 0 <= k < n and sec % 5 == 0:
            start = first + 300 * k
            m = windows[k]
            up, down = tokens(m)
            p = _true_p_up(cl, sec, start, strikes[start], sigma) if start in strikes else 0.5
            z = math.log(max(min(p, 1 - 1e-6), 1e-6) / (1 - max(min(p, 1 - 1e-6), 1e-6))) + rng.gauss(0, market_noise)
            pm = 1 / (1 + math.exp(-z))
            bid = min(max(math.floor((pm - 0.01) * 100) / 100, 0.01), 0.97)
            ask = round(bid + 0.02, 2)
            out.append((t + 20, "clob", book_msg(m, up, t, [(bid, 200), (round(bid - 0.01, 2), 500)],
                                                 [(ask, 200), (round(ask + 0.01, 2), 500)])))
            out.append((t + 21, "clob", book_msg(m, down, t, [(round(1 - ask, 2), 200), (round(0.99 - ask, 2), 500)],
                                                 [(round(1 - bid, 2), 200), (round(1.01 - bid, 2), 500)])))
    for k, m in enumerate(windows):
        start = first + 300 * k
        end = start + 300
        if end + 60 > last_end + 4:
            continue  # the stream stops before this window's result would arrive
        final = twap[end]
        winner = "Up" if final >= strikes[start] else "Down"
        out.append(((end + 60) * 1000, "resolution", json.dumps(resolution_event(m, winner, strikes[start], final))))
    out.sort(key=lambda x: x[0])
    return out
