"""Deterministic trading engine shared by live paper trading and replay.

The engine is driven only by events and their local receive timestamps (event time). It never reads the
wall clock or any other hidden input, so feeding it the same recorded ``(recv_ms, source, raw)`` stream
always produces the same signals, fills and settlements.

Order of operations for every incoming event at time ``t``:
 1. simulated orders whose arrival time <= t are executed against the book state *before* this event
    (i.e. the book as it was at the arrival moment);
 2. the event is applied to the state;
 3. if ``eval_interval_ms`` of event time has passed, strategy evaluation and settlement checks run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .book import OrderBook
from .config import Settings
from .events import (
    BookDelta,
    BookSnapshot,
    Event,
    LatencySample,
    MarketInfo,
    OracleTick,
    ParseStats,
    PriceTick,
    Resolution,
    TickSizeChange,
    TradePrint,
    parse_raw,
)
from .fills import RestingBuy, all_in_cost_per_share, floor_size, floor_to_tick, simulate_taker_buy
from .model import Model, ModelInput, Prediction, get_model
from .storage import Storage

log = logging.getLogger("polybot.engine")

VENUES = ("binance", "coinbase")
PREDICTION_EVERY_MS = 10_000
HISTORY_S = 3600


def _seed_for(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


@dataclass
class VenueState:
    price: float = 0.0
    recv_ms: int = 0
    by_sec: dict[int, float] = field(default_factory=dict)  # exchange second -> last price
    basis: float | None = None  # EWMA of log(chainlink) - log(venue)
    basis_var: float = 0.0  # EWMA of squared innovation
    basis_n: int = 0


@dataclass
class PendingOrder:
    sid: str
    slug: str
    side: str
    token: str
    kind: str  # "taker" or "maker"
    arrival_ms: int
    limit_price: float
    budget: float
    p_win: float


@dataclass
class MarketState:
    info: MarketInfo
    strike: float | None = None
    last_pred_ms: int = 0


class Engine:
    def __init__(self, settings: Settings, store: Storage, model: Model | None = None):
        self.s = settings
        self.store = store
        self.model = model or get_model(settings.model)
        self.now_ms = 0
        self.stats = ParseStats()
        self.counters: dict[str, int] = {}
        self.venues = {v: VenueState() for v in VENUES}
        self.cl_spot: dict[int, float] = {}  # oracle second -> Chainlink spot
        self.cl_spot_recv_ms = 0
        self.cl_spot_last_ms = 0  # latest Chainlink timestamp seen
        self.twap_last_ms = 0
        self.twap: dict[int, float] = {}  # oracle second -> Chainlink 60s TWAP
        self.twap_recv_ms = 0
        self.comp_log: deque[tuple[int, float]] = deque()  # (second, log composite) for realized vol
        self.last_comp_sec = 0
        self.markets: dict[str, MarketState] = {}
        self.token_map: dict[str, tuple[str, str]] = {}  # token -> (slug, "Up"/"Down")
        self.books: dict[str, OrderBook] = {}
        self.clob_recv_ms = 0
        self.rtts: deque[float] = deque(maxlen=500)
        self.pending: list[PendingOrder] = []
        self.resting: list[tuple[PendingOrder, RestingBuy]] = []
        self.open_slugs: set[str] = set()
        self.next_eval_ms = 0
        self.last_prune_ms = 0
        self.last_decision: dict[str, Any] = {}
        self._recover()

    # ------------------------------------------------------------------ recovery
    def _recover(self) -> None:
        """After a restart: in-flight simulated orders are cancelled (we cannot know the book they would
        have met), and positions without a settlement are reloaded so they still get settled."""
        for row in self.store.pending_signals():
            self.store.update_signal(row["id"], "cancelled_restart")
            self._inc("cancelled_restart")
        for r in self.store.query(
            "SELECT DISTINCT f.slug FROM fills f LEFT JOIN settlements s ON s.id=f.id WHERE s.id IS NULL"
        ):
            self.open_slugs.add(r[0])

    def _inc(self, k: str, n: int = 1) -> None:
        self.counters[k] = self.counters.get(k, 0) + n

    # ------------------------------------------------------------------ input
    def on_raw(self, source: str, recv_ms: int, raw: str) -> None:
        for ev in parse_raw(source, recv_ms, raw, self.stats):
            self.on_event(ev)

    def on_event(self, ev: Event) -> None:
        t = ev.recv_ms
        if t > self.now_ms:
            self.now_ms = t
        self._execute_due()
        self._apply(ev)
        if self.now_ms >= self.next_eval_ms:
            self.next_eval_ms = self.now_ms - self.now_ms % self.s.eval_interval_ms + self.s.eval_interval_ms
            self._evaluate()
            self._settle_fallbacks()

    def _apply(self, ev: Event) -> None:
        if isinstance(ev, PriceTick):
            v = self.venues.get(ev.venue)
            if v is None:
                return
            v.price, v.recv_ms = ev.price, ev.recv_ms
            v.by_sec[ev.exch_ms // 1000] = ev.price
            self._sample_composite()
        elif isinstance(ev, OracleTick):
            sec = ev.oracle_ms // 1000
            if ev.kind == "spot":
                if sec not in self.cl_spot:
                    self.cl_spot[sec] = ev.value
                    self._update_basis(sec, ev.value)
                self.cl_spot_recv_ms = max(self.cl_spot_recv_ms, ev.recv_ms)
                self.cl_spot_last_ms = max(self.cl_spot_last_ms, ev.oracle_ms)
            else:
                self.twap.setdefault(sec, ev.value)
                self.twap_recv_ms = max(self.twap_recv_ms, ev.recv_ms)
                self.twap_last_ms = max(self.twap_last_ms, ev.oracle_ms)
                self._maybe_set_strike(sec)
        elif isinstance(ev, BookSnapshot):
            self.clob_recv_ms = ev.recv_ms
            if ev.asset_id in self.token_map:
                self.books.setdefault(ev.asset_id, OrderBook(ev.asset_id)).apply_snapshot(ev)
        elif isinstance(ev, BookDelta):
            self.clob_recv_ms = ev.recv_ms
            b = self.books.get(ev.asset_id)
            if b is not None and b.has_snapshot:
                b.apply_delta(ev)
        elif isinstance(ev, TradePrint):
            self.clob_recv_ms = ev.recv_ms
            self._maker_on_trade(ev)
        elif isinstance(ev, TickSizeChange):
            b = self.books.get(ev.asset_id)
            if b is not None:
                b.tick = ev.tick
        elif isinstance(ev, MarketInfo):
            self._on_market(ev)
        elif isinstance(ev, Resolution):
            self._on_resolution(ev)
        elif isinstance(ev, LatencySample):
            self.rtts.append(ev.rtt_ms)
        self._prune()

    # ------------------------------------------------------------------ price state
    def _sample_composite(self) -> None:
        sec = self.now_ms // 1000
        if sec <= self.last_comp_sec:
            return
        prices = [v.price for v in self.venues.values() if v.price > 0 and self.now_ms - v.recv_ms < 5000]
        if len(prices) == len(VENUES):
            self.comp_log.append((sec, sum(math.log(p) for p in prices) / len(prices)))
            self.last_comp_sec = sec
        cutoff = sec - self.s.vol_lookback_s
        while self.comp_log and self.comp_log[0][0] < cutoff:
            self.comp_log.popleft()

    def _update_basis(self, sec: int, cl_value: float) -> None:
        alpha = 1.0 - 0.5 ** (1.0 / self.s.basis_halflife_s)
        for v in self.venues.values():
            p = v.by_sec.get(sec)
            if p is None:
                continue
            obs = math.log(cl_value) - math.log(p)
            if v.basis is None:
                v.basis = obs
            else:
                innov = obs - v.basis
                v.basis_var = (1 - alpha) * v.basis_var + alpha * innov * innov
                v.basis += alpha * innov
            v.basis_n += 1

    def realized_sigma(self) -> float | None:
        """Per-sqrt-second vol of the composite log price over the lookback, floored and inflated."""
        if len(self.comp_log) < self.s.min_vol_samples:
            return None
        ss, dt = 0.0, 0
        prev = None
        for sec, lp in self.comp_log:
            if prev is not None:
                ss += (lp - prev[1]) ** 2
                dt += sec - prev[0]
            prev = (sec, lp)
        if dt <= 0:
            return None
        sigma = math.sqrt(ss / dt)
        return max(sigma, self.s.vol_floor_per_sqrt_s) * self.s.vol_inflation

    def spot_estimate(self) -> tuple[float, float] | None:
        """(log Chainlink-spot estimate, basis error sd) from fresh exchange prices, or None."""
        ests, vars_ = [], []
        for v in self.venues.values():
            if v.basis is None or v.basis_n < 30 or self.now_ms - v.recv_ms > self.s.stale_exchange_ms:
                return None  # require every venue fresh and calibrated (pessimistic: no partial composite)
            ests.append(math.log(v.price) + v.basis)
            vars_.append(v.basis_var)
        sd = math.sqrt(sum(vars_) / len(vars_))  # average variance, no diversification credit
        sd = max(sd, self.s.basis_err_floor) * self.s.basis_err_inflation
        return sum(ests) / len(ests), sd

    # ------------------------------------------------------------------ markets
    def _on_market(self, m: MarketInfo) -> None:
        if m.slug in self.markets:
            self.markets[m.slug].info = m
        else:
            self.markets[m.slug] = MarketState(m)
            self.token_map[m.token_up] = (m.slug, "Up")
            self.token_map[m.token_down] = (m.slug, "Down")
            if m.twap_lookback_s != 60 or "twap" not in m.resolution_source:
                log.warning("market %s has unexpected resolution config: %s / %s",
                            m.slug, m.resolution_source, m.twap_lookback_s)
        self.store.upsert_market(
            dict(slug=m.slug, start_ts=m.start_ts, end_ts=m.end_ts, token_up=m.token_up, token_down=m.token_down,
                 fee_rate=m.fee_rate, fee_exponent=m.fee_exponent, tick=m.tick_size, min_size=m.min_size,
                 twap_window_s=m.twap_lookback_s)
        )
        self._maybe_set_strike(m.start_ts)

    def _maybe_set_strike(self, sec: int) -> None:
        for ms in self.markets.values():
            if ms.strike is None and ms.info.start_ts == sec and sec in self.twap:
                ms.strike = self.twap[sec]
                self.store.set_strike(ms.info.slug, ms.strike, "rtds_twap60")

    def wanted_tokens(self) -> set[str]:
        """Tokens whose books we need: windows that have not ended and start within one window."""
        now_s = self.now_ms // 1000
        out: set[str] = set()
        for ms in self.markets.values():
            m = ms.info
            if m.end_ts > now_s and m.start_ts <= now_s + self.s.window_seconds:
                out.update((m.token_up, m.token_down))
        return out

    def _prune(self) -> None:
        if self.now_ms - self.last_prune_ms < 60_000:
            return
        self.last_prune_ms = self.now_ms
        cutoff = self.now_ms // 1000 - HISTORY_S
        for d in (self.cl_spot, self.twap, *(v.by_sec for v in self.venues.values())):
            for k in [k for k in d if k < cutoff]:
                del d[k]
        now_s = self.now_ms // 1000
        for slug in [s for s, ms in self.markets.items() if ms.info.end_ts < now_s - 600]:
            m = self.markets.pop(slug).info
            for t in (m.token_up, m.token_down):
                self.token_map.pop(t, None)
                self.books.pop(t, None)

    # ------------------------------------------------------------------ model input
    def model_input(self, ms: MarketState) -> ModelInput | None:
        m = ms.info
        if ms.strike is None:
            return None
        est = self.spot_estimate()
        sigma = self.realized_sigma()
        if est is None or sigma is None:
            return None
        w = float(m.twap_lookback_s or 60)
        tau = m.end_ts - self.now_ms / 1000.0
        known_avg, known_w = None, 0.0
        if tau < w:
            lo = m.end_ts - int(w)
            vals = [math.log(v) for s, v in self.cl_spot.items() if lo < s <= self.now_ms // 1000]
            if vals:
                known_avg = sum(vals) / len(vals)
                # only seconds we actually observed count as known; unobserved ones (feed lag) stay uncertain
                known_w = min(len(vals), w - tau) / w
        return ModelInput(
            tau_s=tau, twap_window_s=w, log_strike=math.log(ms.strike), log_spot_est=est[0],
            known_avg_log=known_avg, known_weight=known_w, sigma=sigma, basis_sd=est[1],
            twap_err_sd=self.s.twap_err_sd,
        )

    def predict(self, ms: MarketState) -> Prediction | None:
        inp = self.model_input(ms)
        return self.model.predict(inp) if inp else None

    # ------------------------------------------------------------------ strategy
    def fresh(self, m: MarketInfo) -> str | None:
        """Reason string if any required feed is stale (event time), else None."""
        now = self.now_ms
        for name, v in self.venues.items():
            if now - v.recv_ms > self.s.stale_exchange_ms:
                return f"stale_{name}"
        # Judge the oracle by its own timestamps: a fresh snapshot of old data is still old data.
        if now - self.cl_spot_last_ms > self.s.stale_oracle_ms:
            return "stale_chainlink"
        if now - self.twap_last_ms > self.s.stale_oracle_ms:
            return "stale_twap"
        if now - self.clob_recv_ms > self.s.stale_book_ms:
            return "stale_clob"
        for t in (m.token_up, m.token_down):
            b = self.books.get(t)
            if b is None or not b.has_snapshot:
                return "no_book"
        return None

    def available_cash(self) -> float:
        reserved = sum(p.budget for p in self.pending) + sum(p.budget for p, _ in self.resting)
        return self.store.cash(self.s.starting_bankroll) - reserved

    def equity(self) -> float:
        return self.store.cash(self.s.starting_bankroll) + self.store.open_cost()

    def _evaluate(self) -> None:
        now_s = self.now_ms / 1000.0
        if self.resting:
            keep = []
            for po, rb in self.resting:
                if self.now_ms >= rb.expires_ms:
                    self.store.update_signal(po.sid, "filled" if rb.filled > 0 else "expired",
                                             {"maker_filled": rb.filled})
                else:
                    keep.append((po, rb))
            self.resting = keep
        for ms in list(self.markets.values()):
            m = ms.info
            if not (m.start_ts <= now_s < m.end_ts):
                continue
            pred = self.predict(ms)
            bucket = self.now_ms - self.now_ms % PREDICTION_EVERY_MS
            if pred is not None and bucket > ms.last_pred_ms:
                ms.last_pred_ms = bucket
                self.store.insert_prediction(m.slug, bucket, m.end_ts - now_s, pred.p_up)
            self._consider_entry(ms, pred)

    def _consider_entry(self, ms: MarketState, pred: Prediction | None) -> None:
        m = ms.info
        tau = m.end_ts - self.now_ms / 1000.0
        if pred is None or not (self.s.min_tau_s <= tau <= self.s.max_tau_s) or not m.accepting_orders:
            return
        if self.store.entries_for(m.slug) >= self.s.max_entries_per_market:
            return
        if any(p.slug == m.slug for p in self.pending):
            return
        if len(self.rtts) < 5:
            self._inc("skip_no_latency_samples")
            return
        stale = self.fresh(m)
        if stale:
            self._inc("skip_" + stale)
            return
        clip = self.s.p_clip
        best: tuple[float, str, str, float, float, float] | None = None
        for side, token, p in (("Up", m.token_up, pred.p_up), ("Down", m.token_down, 1.0 - pred.p_up)):
            p_win = min(max(p, 1.0 - clip), clip)
            book = self.books[token]
            ask = book.best_ask()
            if ask is None:
                continue
            edge = p_win - all_in_cost_per_share(ask, m.fee_rate, m.fee_exponent)
            if edge < self.s.min_edge:
                continue
            tick = book.tick or m.tick_size
            limit = floor_to_tick(ask + self.s.limit_slippage_ticks * tick, tick)
            while limit > ask + 1e-9 and p_win - all_in_cost_per_share(limit, m.fee_rate, m.fee_exponent) < self.s.min_edge:
                limit = round(limit - tick, 6)
            limit = min(limit, 0.99)
            if best is None or edge > best[0]:
                best = (edge, side, token, p_win, ask, limit)
        if best is None:
            self._maybe_maker(ms, pred)
            return
        edge, side, token, p_win, ask, limit = best
        budget = min(self.s.stake_fraction * self.equity(), self.s.max_stake_usd, self.available_cash())
        pre = simulate_taker_buy(self.books[token].asks_ascending(), limit, budget, m.min_size, m.fee_rate,
                                 m.fee_exponent, self.s.book_take_fraction, self.s.fee_round_up)
        if not pre.levels:
            self._inc("skip_" + pre.reason)
            return
        self._submit(ms, side, token, "taker", p_win, ask, limit, edge, tau, budget)

    def _maybe_maker(self, ms: MarketState, pred: Prediction) -> None:
        if not self.s.enable_maker:
            return
        m = ms.info
        tau = m.end_ts - self.now_ms / 1000.0
        for side, token, p in (("Up", m.token_up, pred.p_up), ("Down", m.token_down, 1.0 - pred.p_up)):
            p_win = min(max(p, 1.0 - self.s.p_clip), self.s.p_clip)
            book = self.books[token]
            ask = book.best_ask()
            tick = book.tick or m.tick_size
            price = floor_to_tick(p_win - self.s.min_edge, tick)  # maker pays no fee
            if ask is None or price <= 0 or price >= ask - 1e-9:
                continue
            budget = min(self.s.stake_fraction * self.equity(), self.s.max_stake_usd, self.available_cash())
            if floor_size(budget / price) < m.min_size:
                continue
            self._submit(ms, side, token, "maker", p_win, ask, price, p_win - price, tau, budget)
            return

    def sample_latency_ms(self, sid: str) -> float:
        rng = random.Random(_seed_for(self.s.seed, sid))
        base = rng.choice(list(self.rtts))
        jitter = math.exp(rng.gauss(0.0, self.s.latency_jitter_sigma))
        return max(self.s.latency_ms_floor, base * jitter + self.s.latency_extra_ms)

    def _submit(self, ms: MarketState, side: str, token: str, kind: str, p_win: float, ask: float,
                limit: float, edge: float, tau: float, budget: float) -> None:
        sid = f"{ms.info.slug}:{self.now_ms}:{side}:{kind}"
        lat = self.sample_latency_ms(sid)
        arrival = self.now_ms + int(math.ceil(lat))
        ok = self.store.insert_signal(dict(
            id=sid, slug=ms.info.slug, ts_ms=self.now_ms, side=side, kind=kind, p_win=p_win, best_ask=ask,
            limit_price=limit, edge=edge, tau_s=tau, latency_ms=lat, arrival_ms=arrival, budget=budget,
            status="pending", detail="{}",
        ))
        if not ok:
            return
        self._inc("signals")
        self.last_decision = dict(id=sid, side=side, kind=kind, p_win=round(p_win, 4), ask=ask, limit=limit,
                                  edge=round(edge, 4), latency_ms=round(lat, 1))
        log.info("signal %s", self.last_decision)
        self.pending.append(PendingOrder(sid, ms.info.slug, side, token, kind, arrival, limit, budget, p_win))

    # ------------------------------------------------------------------ execution
    def _execute_due(self) -> None:
        if not self.pending:
            return
        due = [p for p in self.pending if p.arrival_ms <= self.now_ms]
        if not due:
            return
        self.pending = [p for p in self.pending if p.arrival_ms > self.now_ms]
        for po in due:
            self._execute(po)

    def _arrival_p_win(self, po: PendingOrder) -> float | None:
        ms = self.markets.get(po.slug)
        pred = self.predict(ms) if ms else None
        if pred is None:
            return None
        p = pred.p_up if po.side == "Up" else 1.0 - pred.p_up
        return min(max(p, 1.0 - self.s.p_clip), self.s.p_clip)

    def _execute(self, po: PendingOrder) -> None:
        ms = self.markets.get(po.slug)
        if ms is None or self.now_ms / 1000.0 >= ms.info.end_ts:
            self.store.update_signal(po.sid, "expired")
            self._inc("expired")
            return
        m = ms.info
        p_arr = self._arrival_p_win(po)
        if po.kind == "maker":
            n = floor_size(po.budget / po.limit_price)
            rb = RestingBuy(po.sid, po.token, po.limit_price, n, self.now_ms,
                            min(self.now_ms + int(self.s.maker_ttl_s * 1000), m.end_ts * 1000))
            self.resting.append((po, rb))
            self.store.update_signal(po.sid, "resting", {"p_win_arrival": p_arr})
            return
        book = self.books.get(po.token)
        asks = book.asks_ascending() if book else []
        fill = simulate_taker_buy(asks, po.limit_price, po.budget, m.min_size, m.fee_rate, m.fee_exponent,
                                  self.s.book_take_fraction, self.s.fee_round_up)
        if not fill.levels:
            self.store.update_signal(po.sid, "missed", {"reason": fill.reason, "p_win_arrival": p_arr,
                                                        "best_ask_arrival": book.best_ask() if book else None})
            self._inc("missed")
            return
        edge_arr = (p_arr - fill.cost_per_net_share) if p_arr is not None else None
        self._record_fill(po, po.sid, fill.gross_shares, fill.net_shares, fill.cost_usdc, fill.fee_usdc,
                          fill.avg_price, p_arr, edge_arr, [(lv.price, lv.shares, lv.fee_usdc) for lv in fill.levels])
        self.store.update_signal(po.sid, "filled" if fill.reason == "filled" else "partial",
                                 {"reason": fill.reason, "p_win_arrival": p_arr, "edge_arrival": edge_arr})

    def _record_fill(self, po: PendingOrder, fid: str, gross: float, net: float, cost: float, fee: float,
                     avg: float, p_arr: float | None, edge_arr: float | None, levels: list[Any]) -> None:
        if self.store.insert_fill(dict(
            id=fid, slug=po.slug, side=po.side, ts_ms=self.now_ms, kind=po.kind, gross_shares=gross,
            net_shares=net, cost=cost, fee=fee, avg_price=avg, p_win_decision=po.p_win,
            p_win_arrival=p_arr, edge_arrival=edge_arr, levels=json.dumps(levels),
        )):
            self.open_slugs.add(po.slug)
            self._inc("fills")
            log.info("fill %s side=%s net=%.2f cost=%.4f fee=%.5f avg=%.4f", fid, po.side, net, cost, fee, avg)

    def _maker_on_trade(self, ev: TradePrint) -> None:
        if not self.resting:
            return
        keep = []
        for po, rb in self.resting:
            if rb.asset_id == ev.asset_id:
                q = rb.on_trade(ev.price, ev.size)
                if q > 0:
                    self._record_fill(po, f"{po.sid}:{self.counters.get('maker_fills', 0)}", q, q, q * rb.price,
                                      0.0, rb.price, None, None, [(rb.price, q, 0.0)])
                    self._inc("maker_fills")
            if rb.filled >= rb.shares - 1e-9 or self.now_ms >= rb.expires_ms:
                self.store.update_signal(po.sid, "filled" if rb.filled > 0 else "expired",
                                         {"maker_filled": rb.filled})
            else:
                keep.append((po, rb))
        self.resting = keep

    # ------------------------------------------------------------------ settlement
    def _on_resolution(self, r: Resolution) -> None:
        row = self.store.market(r.slug)
        if row is not None and row["settled_by"] == "gamma":
            return
        self.store.set_resolution(r.slug, r.winner, r.final_price, r.price_to_beat, "gamma", r.recv_ms)
        if row is not None and row["strike"] is not None and r.price_to_beat is not None:
            diff = row["strike"] - r.price_to_beat
            if abs(diff) > 1e-6 * r.price_to_beat:
                log.warning("strike mismatch %s ours=%.6f gamma=%.6f", r.slug, row["strike"], r.price_to_beat)
                self._inc("strike_mismatch")
        self._settle(r.slug, r.winner, "gamma")

    def _settle(self, slug: str, winner: str, by: str) -> None:
        for f in self.store.fills_for(slug):
            won = f["side"] == winner
            payout = float(f["net_shares"]) if won else 0.0
            self.store.upsert_settlement(dict(id=f["id"], slug=slug, ts_ms=self.now_ms, won=int(won),
                                              payout=payout, pnl=payout - float(f["cost"]), settled_by=by))
        if by == "gamma":
            self.open_slugs.discard(slug)
        self._inc("settled_" + by)

    def _settle_fallbacks(self) -> None:
        now_s = self.now_ms // 1000
        for slug in list(self.open_slugs):
            row = self.store.market(slug)
            if row is None or row["settled_by"] is not None:
                continue
            if now_s < row["end_ts"] + self.s.fallback_after_s:
                continue
            end_twap = self.twap.get(int(row["end_ts"]))
            strike = row["strike"]
            if end_twap is not None and strike is not None:
                winner, by = ("Up" if end_twap >= strike else "Down"), "fallback"
            else:
                # Pessimistic: if we cannot tell, every position in this market loses.
                sides = {f["side"] for f in self.store.fills_for(slug)}
                winner, by = ("Down" if "Up" in sides else "Up"), "fallback_loss"
                if len(sides) > 1:
                    winner = "none"
            log.warning("fallback settlement %s winner=%s by=%s", slug, winner, by)
            self.store.set_resolution(slug, winner, end_twap, None, by, self.now_ms)
            self._settle(slug, winner, by)

    # ------------------------------------------------------------------ introspection
    def snapshot(self) -> dict[str, Any]:
        now = self.now_ms
        return dict(
            event_time_ms=now,
            venues={k: dict(price=v.price, age_ms=now - v.recv_ms, basis=v.basis,
                            basis_sd=math.sqrt(v.basis_var) if v.basis_n else None) for k, v in self.venues.items()},
            chainlink_age_ms=now - self.cl_spot_last_ms,
            twap_age_ms=now - self.twap_last_ms,
            clob_age_ms=now - self.clob_recv_ms,
            sigma_per_sqrt_s=self.realized_sigma(),
            rtt_samples=len(self.rtts),
            rtt_median_ms=sorted(self.rtts)[len(self.rtts) // 2] if self.rtts else None,
            markets=[dict(slug=ms.info.slug, strike=ms.strike, start=ms.info.start_ts, end=ms.info.end_ts)
                     for ms in self.markets.values()],
            pending=len(self.pending),
            resting=len(self.resting),
            counters=dict(self.counters),
            parse=dict(ok=self.stats.ok, ignored=self.stats.ignored, errors=self.stats.errors,
                       last_error=self.stats.last_error),
            last_decision=self.last_decision,
        )
