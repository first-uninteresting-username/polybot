"""Normalized events and the parsers that turn raw recorded messages into them.

Live trading and replay both go through ``parse_raw(source, recv_ms, raw)``. The recorder stores
exactly ``(recv_ms, source, raw)``, so the engine sees identical inputs in both modes.

Raw formats were checked against live responses saved in ``tests/fixtures`` (2026-10-29).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

# --- sources (the value stored in recordings) ---
SRC_BINANCE = "binance"
SRC_COINBASE = "coinbase"
SRC_CLOB = "clob"
SRC_RTDS = "rtds"
SRC_GAMMA_MARKET = "gamma_market"  # a Gamma /markets object (JSON)
SRC_RESOLUTION = "resolution"  # a Gamma /events object for a finished window (JSON)
SRC_LATENCY = "latency"  # {"rtt_ms": float} from our own CLOB round-trip probe


@dataclass(frozen=True, slots=True)
class PriceTick:
    venue: str
    recv_ms: int
    exch_ms: int
    price: float


@dataclass(frozen=True, slots=True)
class OracleTick:
    kind: str  # "spot" (Chainlink BTC/USD) or "twap60" (Chainlink 60s TWAP stream)
    recv_ms: int
    oracle_ms: int
    value: float


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    asset_id: str
    recv_ms: int
    exch_ms: int
    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class BookDelta:
    asset_id: str
    recv_ms: int
    exch_ms: int
    side: str  # "BUY" (bid side) or "SELL" (ask side)
    price: float
    size: float  # new aggregate size at this level; 0 removes it
    best_bid: float | None = None  # exchange's best bid/ask after this change ("0"/"1" = side empty)
    best_ask: float | None = None


@dataclass(frozen=True, slots=True)
class TradePrint:
    asset_id: str
    recv_ms: int
    exch_ms: int
    price: float
    size: float
    side: str  # taker side as reported


@dataclass(frozen=True, slots=True)
class TickSizeChange:
    asset_id: str
    recv_ms: int
    tick: float


@dataclass(frozen=True, slots=True)
class MarketInfo:
    recv_ms: int
    slug: str
    condition_id: str
    start_ts: int  # unix seconds, window start
    end_ts: int  # unix seconds, window end
    token_up: str
    token_down: str
    tick_size: float
    min_size: float
    fees_enabled: bool
    fee_rate: float  # taker fee rate in fee = C * rate * p * (1-p)
    fee_exponent: float
    resolution_source: str
    twap_lookback_s: int
    accepting_orders: bool


@dataclass(frozen=True, slots=True)
class Resolution:
    recv_ms: int
    slug: str
    winner: str  # "Up" or "Down"
    price_to_beat: float | None
    final_price: float | None
    source: str = "gamma"


@dataclass(frozen=True, slots=True)
class LatencySample:
    recv_ms: int
    rtt_ms: float


Event = (
    PriceTick
    | OracleTick
    | BookSnapshot
    | BookDelta
    | TradePrint
    | TickSizeChange
    | MarketInfo
    | Resolution
    | LatencySample
)


@dataclass(slots=True)
class ParseStats:
    ok: int = 0
    ignored: int = 0
    errors: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    last_error: dict[str, str] = field(default_factory=dict)


def _iso_ms(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def _levels(rows: list[dict[str, str]]) -> tuple[tuple[float, float], ...]:
    return tuple((float(r["price"]), float(r["size"])) for r in rows)


def parse_binance(recv_ms: int, raw: str) -> list[Event]:
    d = json.loads(raw)
    if d.get("e") == "trade":
        return [PriceTick("binance", recv_ms, int(d["T"]), float(d["p"]))]
    if "b" in d and "a" in d and "u" in d:  # bookTicker: mid
        return [PriceTick("binance", recv_ms, recv_ms, (float(d["b"]) + float(d["a"])) / 2)]
    return []


def parse_coinbase(recv_ms: int, raw: str) -> list[Event]:
    d = json.loads(raw)
    if d.get("type") != "ticker" or d.get("product_id") != "BTC-USD":
        return []
    return [PriceTick("coinbase", recv_ms, _iso_ms(d["time"]), float(d["price"]))]


def _opt_float(v: Any) -> float | None:
    return float(v) if v not in (None, "") else None


def _clob_one(recv_ms: int, d: dict[str, Any]) -> list[Event]:
    et = d.get("event_type")
    ts = int(d.get("timestamp") or recv_ms)
    if et == "book":
        return [BookSnapshot(d["asset_id"], recv_ms, ts, _levels(d.get("bids", [])), _levels(d.get("asks", [])))]
    if et == "price_change":
        return [
            BookDelta(c["asset_id"], recv_ms, ts, c["side"], float(c["price"]), float(c["size"]),
                      _opt_float(c.get("best_bid")), _opt_float(c.get("best_ask")))
            for c in d.get("price_changes", [])
        ]
    if et == "last_trade_price":
        return [
            TradePrint(d["asset_id"], recv_ms, ts, float(d["price"]), float(d.get("size") or 0), d.get("side", ""))
        ]
    if et == "tick_size_change":
        return [TickSizeChange(d["asset_id"], recv_ms, float(d["new_tick_size"]))]
    return []


def parse_clob(recv_ms: int, raw: str) -> list[Event]:
    if not raw or raw[0] not in "[{":
        return []  # "PONG" and empty frames
    d = json.loads(raw)
    out: list[Event] = []
    for item in d if isinstance(d, list) else [d]:
        out.extend(_clob_one(recv_ms, item))
    return out


def _rtds_value(item: dict[str, Any]) -> float:
    fav = item.get("full_accuracy_value")
    if fav:  # E18 fixed point on the legacy RTDS feed (verified against "value")
        return int(fav) / 1e18
    return float(item["value"])


def parse_rtds(recv_ms: int, raw: str) -> list[Event]:
    if not raw or raw[0] != "{":
        return []
    d = json.loads(raw)
    p = d.get("payload") or {}
    if p.get("symbol") != "btc/usd":
        return []
    topic = d.get("topic", "")
    # The Chainlink snapshot frame arrives with topic "crypto_prices" (not "..._chainlink");
    # the TWAP stream carries window_s=60. Classify by both.
    if "twap" in topic or p.get("window_s") is not None:
        if int(p.get("window_s", 60)) != 60:
            return []
        kind = "twap60"
    elif topic in ("crypto_prices_chainlink", "crypto_prices"):
        kind = "spot"
    else:
        return []
    if "data" in p:  # snapshot of recent history
        return [OracleTick(kind, recv_ms, int(x["timestamp"]), _rtds_value(x)) for x in p["data"]]
    return [OracleTick(kind, recv_ms, int(p["timestamp"]), _rtds_value(p))]


def _jsonlist(v: Any) -> list[Any]:
    return json.loads(v) if isinstance(v, str) else list(v or [])


def parse_gamma_market(recv_ms: int, raw: str, slug_prefix: str = "btc-updown-5m-") -> list[Event]:
    m = json.loads(raw)
    slug = m["slug"]
    if not slug.startswith(slug_prefix):
        return []
    start_ts = int(slug[len(slug_prefix) :])
    outcomes = _jsonlist(m["outcomes"])
    tokens = _jsonlist(m["clobTokenIds"])
    mapping = dict(zip(outcomes, tokens, strict=True))
    fs = m.get("feeSchedule") or {}
    fees_enabled = bool(m.get("feesEnabled"))
    end_ts = _iso_ms(m["endDate"]) // 1000
    cfg = m.get("cryptoMarketConfig") or {}
    return [
        MarketInfo(
            recv_ms=recv_ms,
            slug=slug,
            condition_id=m["conditionId"],
            start_ts=start_ts,
            end_ts=end_ts,
            token_up=mapping["Up"],
            token_down=mapping["Down"],
            tick_size=float(m.get("orderPriceMinTickSize") or 0.01),
            min_size=float(m.get("orderMinSize") or 5),
            fees_enabled=fees_enabled,
            # If fees are enabled but no schedule is given, fall back to the documented crypto rate
            # (0.07, docs.polymarket.com/trading/fees) rather than assuming zero.
            fee_rate=float(fs.get("rate", 0.07)) if fees_enabled else 0.0,
            fee_exponent=float(fs.get("exponent", 1)),
            resolution_source=m.get("resolutionSource", ""),
            twap_lookback_s=int(cfg.get("twapLookbackSeconds") or 0),
            accepting_orders=bool(m.get("acceptingOrders")),
        )
    ]


def parse_resolution(recv_ms: int, raw: str) -> list[Event]:
    """A Gamma /events object. Returns a Resolution only when the market is unambiguously final."""
    e = json.loads(raw)
    markets = e.get("markets") or []
    if len(markets) != 1:
        return []
    m = markets[0]
    if not m.get("closed") or m.get("umaResolutionStatus") not in (None, "resolved"):
        return []
    outcomes = _jsonlist(m.get("outcomes"))
    prices = [float(x) for x in _jsonlist(m.get("outcomePrices"))]
    winners = [o for o, p in zip(outcomes, prices, strict=True) if p == 1.0]
    if len(winners) != 1 or sorted(prices) != [0.0, 1.0]:
        return []
    meta = e.get("eventMetadata") or {}
    ptb = meta.get("priceToBeat")
    fin = meta.get("finalPrice")
    return [
        Resolution(
            recv_ms,
            e["slug"],
            winners[0],
            float(ptb) if ptb is not None else None,
            float(fin) if fin is not None else None,
        )
    ]


def parse_latency(recv_ms: int, raw: str) -> list[Event]:
    return [LatencySample(recv_ms, float(json.loads(raw)["rtt_ms"]))]


_PARSERS = {
    SRC_BINANCE: parse_binance,
    SRC_COINBASE: parse_coinbase,
    SRC_CLOB: parse_clob,
    SRC_RTDS: parse_rtds,
    SRC_GAMMA_MARKET: parse_gamma_market,
    SRC_RESOLUTION: parse_resolution,
    SRC_LATENCY: parse_latency,
}


def parse_raw(source: str, recv_ms: int, raw: str, stats: ParseStats | None = None) -> list[Event]:
    fn = _PARSERS.get(source)
    if fn is None:
        if stats:
            stats.ignored += 1
        return []
    try:
        evs = fn(recv_ms, raw)
    except (ValueError, KeyError, TypeError) as exc:  # malformed message: count, never crash
        if stats:
            stats.errors += 1
            stats.by_source[source] = stats.by_source.get(source, 0) + 1
            stats.last_error[source] = repr(exc)[:200]
        return []
    if stats:
        if evs:
            stats.ok += 1
        else:
            stats.ignored += 1
    return evs
