"""Deterministic trading engine shared by live paper trading and replay.

The engine is driven only by events and their local receive timestamps (event time). It never reads the
wall clock or any other hidden input, so feeding it the same recorded ``(recv_ms, source, raw)`` stream
always produces the same signals, fills and settlements.

Order of operations for every incoming event at time ``t``:
 1. simulated orders whose arrival time <= t are executed against the book state *before* this event
    (i.e. the book as it was at the arrival moment);
 2. the event is applied to the state;
 3. if ``eval_interval_ms`` of event time has passed, strategy evaluation and settlement checks run.

Several agents (``Settings.agents``) can trade: the original strategy (named after its model, ``gbm_twap``)
and the online-learning agents of ``rl.py``. They all trade from one paper account: one cash pool sizes every
stake, and an order of one agent sees the displayed depth minus what our own simulated orders already took.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from . import rl
from .book import OrderBook
from .config import Settings
from .events import (
    BookDelta,
    BookSnapshot,
    Deposit,
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
HISTORY_S = 3600
RL_KEEP_S = 6 * 3600  # unresolved RL experience is dropped this long after its window ended
MOMENTUM_S = (5, 15, 30, 60, 120)


def _seed_for(*parts: Any) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "big")


def _clamp(x: float, lo: float, hi: float) -> float:
    return min(max(x, lo), hi)


def _book_shape(book: OrderBook, tick: float) -> tuple[float, float]:
    """(size imbalance at the touch in [-1, 1], log(bid depth / ask depth) within 3 ticks of the touch)."""
    bb, ba = book.best_bid(), book.best_ask()
    imb = 0.0
    if bb is not None and ba is not None:
        sb, sa = book.bids[bb], book.asks[ba]
        imb = (sb - sa) / (sb + sa)
    reach = 3 * tick + 1e-9
    db = sum(sz for p, sz in book.bids.items() if p >= bb - reach) if bb is not None else 0.0
    da = sum(sz for p, sz in book.asks.items() if p <= ba + reach) if ba is not None else 0.0
    return imb, _clamp(math.log((db + 1.0) / (da + 1.0)), -3.0, 3.0)


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
    agent: str


@dataclass
class MarketState:
    info: MarketInfo
    strike: float | None = None
    last_pred_ms: int = 0
    last_rl_ms: int = 0  # last RL decision bucket
    last_rl_sample_ms: int = 0  # last RL training-sample bucket


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
        self._comp_sigma_cache: tuple[Any, float | None] = (None, None)
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
        # Shares our own simulated takers (all agents) took per token and price level in the live markets.
        # The exchange never saw these paper fills, so its book still shows that size.
        self.own_taken: dict[str, dict[float, float]] = {}
        self.last_trade: dict[str, tuple[float, int]] = {}  # token -> (last trade price, recv_ms)
        self._init_agents()
        self._recover()

    def _init_agents(self) -> None:
        names = [a.strip() for a in self.s.agents.split(",") if a.strip()]
        self.baseline_name = self.model.name  # the original strategy is named after its model
        unknown = [a for a in names if a != self.baseline_name and a not in rl.PRESETS]
        if unknown or len(set(names)) != len(names):
            raise ValueError(f"bad agents {self.s.agents!r}; available: {[self.baseline_name, *rl.PRESETS]}")
        self.agent_names = names
        self.baseline_on = self.baseline_name in names
        self.rl_agents = [rl.make_agent(a, self.s.rl_prior_sd, self.s.rl_memory_windows)
                          for a in names if a in rl.PRESETS]
        self.rl_by_name = {a.name: a for a in self.rl_agents}
        for ag in self.rl_agents:
            raw = self.store.get_rl_state(ag.name)
            if raw is not None and not ag.load_json(raw):
                log.warning("starting %s from its prior: the saved state is unreadable or used other inputs", ag.name)
        # slug -> (window end, feature vectors sampled in that window), until the official outcome arrives
        self.rl_experience: dict[str, tuple[int, list[list[float]]]] = {}
        self.rl_done: dict[str, int] = {}  # slug -> window end, for windows already learned from

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

    def reset_trading(self) -> None:
        """Forget in-memory orders and positions after ``Storage.reset_trading``. Market state stays, and so
        does what the learning agents have learned (it is about markets, not about the account)."""
        self.pending.clear()
        self.resting.clear()
        self.open_slugs.clear()
        self.counters.clear()
        self.last_decision = {}
        self.own_taken.clear()

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
            if ev.asset_id in self.token_map:
                self.last_trade[ev.asset_id] = (ev.price, ev.recv_ms)
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
        elif isinstance(ev, Deposit):
            if self.store.insert_deposit(ev.deposit_id, ev.recv_ms, ev.amount):
                self._inc("deposits")
                log.info("paper bankroll topped up by %.2f USDC (shared by all agents)", ev.amount)
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
        sigma = self._comp_sigma()
        if sigma is None:
            return None
        return max(sigma, self.s.vol_floor_per_sqrt_s) * self.s.vol_inflation

    def _comp_sigma(self) -> float | None:
        """Raw vol over the whole composite lookback, cached until ``comp_log`` changes. The deque only grows
        at the right and shrinks at the left, so its length and end points identify its content."""
        cl = self.comp_log
        key = (len(cl), cl[0], cl[-1]) if cl else None
        cached_key, cached = self._comp_sigma_cache
        if key is not None and key == cached_key:
            return cached
        sigma = self._sigma_of(cl)
        self._comp_sigma_cache = (key, sigma)
        return sigma

    @staticmethod
    def _sigma_of(samples: Any) -> float | None:
        """Raw per-sqrt-second vol of ``(second, log price)`` samples."""
        ss, dt = 0.0, 0
        prev = None
        for sec, lp in samples:
            if prev is not None:
                ss += (lp - prev[1]) ** 2
                dt += sec - prev[0]
            prev = (sec, lp)
        if dt <= 0:
            return None
        return math.sqrt(ss / dt)

    def _recent_comp(self, seconds: int) -> list[tuple[int, float]]:
        cutoff = self.now_ms // 1000 - seconds
        out: list[tuple[int, float]] = []
        for x in reversed(self.comp_log):
            if x[0] < cutoff:
                break
            out.append(x)
        out.reverse()
        return out

    def _comp_before(self, horizons: tuple[int, ...]) -> dict[int, tuple[int, float]]:
        """For each horizon h: the latest composite sample at least h seconds old."""
        now_sec = self.now_ms // 1000
        want = sorted(horizons)
        out: dict[int, tuple[int, float]] = {}
        i = 0
        for sec, lp in reversed(self.comp_log):
            while i < len(want) and sec <= now_sec - want[i]:
                out[want[i]] = (sec, lp)
                i += 1
            if i == len(want):
                break
        return out

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
                self.own_taken.pop(t, None)
                self.last_trade.pop(t, None)
        old = now_s - RL_KEEP_S
        for slug in [s for s, (end, _) in self.rl_experience.items() if end < old]:
            del self.rl_experience[slug]
        for slug in [s for s, end in self.rl_done.items() if end < old]:
            del self.rl_done[slug]
        for ag in self.rl_agents:
            for slug in [s for s, v in ag.virtual.items() if v[2] < old]:
                del ag.virtual[slug]

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

    def predict_band(self, ms: MarketState) -> tuple[Prediction, float, float] | None:
        inp = self.model_input(ms)
        return self._band(inp) if inp is not None else None

    def _band(self, inp: ModelInput) -> tuple[Prediction, float, float]:
        """Central prediction plus the lowest and highest P(Up) with sigma and basis sd scaled by
        1/model_sd_band .. model_sd_band. P(Up) is monotone in the spread, so the endpoints bound it."""
        pred = self.model.predict(inp)
        ps = [pred.p_up]
        k = self.s.model_sd_band
        if k > 1.0:
            for f in (1.0 / k, k):
                ps.append(self.model.predict(dataclasses.replace(inp, sigma=inp.sigma * f,
                                                                 basis_sd=inp.basis_sd * f)).p_up)
        return pred, min(ps), max(ps)

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
            if b.desynced:
                return "book_desync"
        return None

    def available_cash(self) -> float:
        reserved = sum(p.budget for p in self.pending) + sum(p.budget for p, _ in self.resting)
        return self.store.cash(self.s.starting_bankroll) - reserved

    def equity(self) -> float:
        return self.store.cash(self.s.starting_bankroll) + self.store.open_cost()

    def stake_budget(self, p_win: float, cost: float) -> float:
        """USDC to commit when buying at all-in ``cost`` per share with win probability ``p_win``: the
        smaller of ``stake_fraction`` and ``kelly_fraction`` x the Kelly fraction (p - c) / (1 - c)."""
        frac = self.s.stake_fraction
        if self.s.kelly_fraction > 0:
            kelly = (p_win - cost) / (1.0 - cost) if cost < 1.0 else 0.0
            frac = min(frac, self.s.kelly_fraction * max(kelly, 0.0))
        return min(frac * self.equity(), self.s.max_stake_usd, self.available_cash())

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
            inp = self.model_input(ms)
            band = self._band(inp) if inp is not None else None
            bucket = self.now_ms - self.now_ms % self.s.prediction_interval_ms
            if band is not None and bucket > ms.last_pred_ms:
                ms.last_pred_ms = bucket
                self.store.insert_prediction(m.slug, bucket, m.end_ts - now_s, band[0].p_up)
            if self.baseline_on:
                self._consider_entry(ms, band)
            if self.rl_agents:
                self._rl_step(ms, inp, band)

    def _consider_entry(self, ms: MarketState, band: tuple[Prediction, float, float] | None) -> None:
        m = ms.info
        tau = m.end_ts - self.now_ms / 1000.0
        if band is None or not (self.s.min_tau_s <= tau <= self.s.max_tau_s) or not m.accepting_orders:
            return
        if self.store.entries_for(m.slug, self.baseline_name) >= self.s.max_entries_per_market:
            return
        if any(p.slug == m.slug and p.agent == self.baseline_name for p in self.pending):
            return
        if len(self.rtts) < 5:
            self._inc("skip_no_latency_samples")
            return
        stale = self.fresh(m)
        if stale:
            self._inc("skip_" + stale)
            return
        pred, lo, hi = band
        best: tuple[float, str, str, float, float, float, float] | None = None
        # p_win: central estimate (recorded, used for calibration); p_safe: worst case over the sd band
        # (used for every trading decision).
        for side, token, p, p_worst in (("Up", m.token_up, pred.p_up, lo),
                                        ("Down", m.token_down, 1.0 - pred.p_up, 1.0 - hi)):
            p_win, p_safe = self._clip(p), self._clip(p_worst)
            book = self.books[token]
            ask = book.best_ask()
            if ask is None:
                continue
            edge = p_safe - all_in_cost_per_share(ask, m.fee_rate, m.fee_exponent)
            if edge < self.s.min_edge:
                continue
            if self._market_disagrees(p_win, book.best_bid(), ask):
                continue
            tick = book.tick or m.tick_size
            limit = floor_to_tick(ask + self.s.limit_slippage_ticks * tick, tick)
            while limit > ask + 1e-9 and p_safe - all_in_cost_per_share(limit, m.fee_rate, m.fee_exponent) < self.s.min_edge:
                limit = round(limit - tick, 6)
            limit = min(limit, 0.99)
            if best is None or edge > best[0]:
                best = (edge, side, token, p_win, p_safe, ask, limit)
        if best is None:
            self._maybe_maker(ms, band)
            return
        edge, side, token, p_win, p_safe, ask, limit = best
        budget = self.stake_budget(p_safe, all_in_cost_per_share(limit, m.fee_rate, m.fee_exponent))
        pre = simulate_taker_buy(self._asks_for(token), limit, budget, m.min_size, m.fee_rate,
                                 m.fee_exponent, self.s.book_take_fraction, self.s.fee_round_up)
        if not pre.levels:
            self._inc("skip_" + pre.reason)
            return
        self._submit(ms, side, token, "taker", p_win, ask, limit, edge, tau, budget, self.baseline_name)

    def _clip(self, p: float) -> float:
        return min(max(p, 1.0 - self.s.p_clip), self.s.p_clip)

    def _market_disagrees(self, p_win: float, bid: float | None, ask: float,
                          counter: str = "skip_model_market_gap") -> bool:
        """A liquid market far from the model is far more often a model or data error (wrong strike,
        stale oracle, vol regime change) than free money."""
        gap = self.s.max_model_market_gap
        if gap <= 0:
            return False
        mid = (bid + ask) / 2 if bid is not None else ask
        if abs(p_win - mid) > gap:
            self._inc(counter)
            return True
        return False

    def _asks_for(self, token: str) -> list[tuple[float, float]]:
        """Displayed asks minus what our own simulated takers (any agent) already took at each level in this
        market. Our paper fills never reach the exchange, so its book still shows that size; without this,
        two agents (or two entries) would both take ``book_take_fraction`` of the same displayed shares."""
        book = self.books.get(token)
        if book is None:
            return []
        asks = book.asks_ascending()
        taken = self.own_taken.get(token)
        if not taken:
            return asks
        out = []
        for price, size in asks:
            left = size - taken.get(price, 0.0)
            if left > 1e-9:
                out.append((price, left))
        return out

    def _maybe_maker(self, ms: MarketState, band: tuple[Prediction, float, float]) -> None:
        if not self.s.enable_maker:
            return
        m = ms.info
        tau = m.end_ts - self.now_ms / 1000.0
        pred, lo, hi = band
        for side, token, p, p_worst in (("Up", m.token_up, pred.p_up, lo),
                                        ("Down", m.token_down, 1.0 - pred.p_up, 1.0 - hi)):
            p_win, p_safe = self._clip(p), self._clip(p_worst)
            book = self.books[token]
            ask = book.best_ask()
            tick = book.tick or m.tick_size
            price = floor_to_tick(p_safe - self.s.min_edge, tick)  # maker pays no fee
            if ask is None or price <= 0 or price >= ask - 1e-9:
                continue
            if self._market_disagrees(p_win, book.best_bid(), ask):
                continue
            budget = self.stake_budget(p_safe, price)
            if floor_size(budget / price) < m.min_size:
                continue
            self._submit(ms, side, token, "maker", p_win, ask, price, p_safe - price, tau, budget, self.baseline_name)
            return

    # ------------------------------------------------------------------ learning agents (rl.py)
    def rl_features(self, ms: MarketState, inp: ModelInput, pred: Prediction) -> tuple[list[float], float] | None:
        """The learning agents' inputs (``rl.FEATURES`` order) and the market's P(Up), computed only from
        data already received (no look-ahead). None if an input is unavailable."""
        m = ms.info
        up, dn = self.books.get(m.token_up), self.books.get(m.token_down)
        sig = inp.sigma
        if up is None or dn is None or not sig > 0 or not inp.basis_sd > 0:
            return None
        bu, au, bd, ad = up.best_bid(), up.best_ask(), dn.best_bid(), dn.best_ask()
        mids, spreads = [], []
        if bu is not None and au is not None:
            mids.append((bu + au) / 2)
            spreads.append(au - bu)
        if bd is not None and ad is not None:
            mids.append(1.0 - (bd + ad) / 2)
            spreads.append(ad - bd)
        if not mids:
            return None
        p_mkt = sum(mids) / len(mids)
        mkt = rl.logit(p_mkt, 0.01)
        spread = _clamp(10.0 * sum(spreads) / len(spreads), 0.0, 2.0)
        imb_u, dep_u = _book_shape(up, up.tick or m.tick_size)
        imb_d, dep_d = _book_shape(dn, dn.tick or m.tick_size)
        trade: tuple[float, int] | None = None  # latest trade of either token, as P(Up)
        for tok, flip in ((m.token_up, False), (m.token_down, True)):
            tr = self.last_trade.get(tok)
            if tr is not None and self.now_ms - tr[1] <= 60_000 and (trade is None or tr[1] > trade[1]):
                trade = ((1.0 - tr[0]) if flip else tr[0], tr[1])
        s_all = self._comp_sigma()
        recent = self._recent_comp(60)
        s_60 = self._sigma_of(recent) if len(recent) >= 30 else None
        prices = [v.price for v in self.venues.values()]
        if s_all is None or min(prices) <= 0:
            return None
        lp_now = sum(math.log(p) for p in prices) / len(prices)  # the composite, as in comp_log
        now_s = self.now_ms / 1000.0
        past = self._comp_before(MOMENTUM_S)
        mom: dict[str, float] = {}
        for h in MOMENTUM_S:
            pt = past.get(h)
            mom[f"mom_{h}"] = 0.0 if pt is None else _clamp(
                (lp_now - pt[1]) / (sig * math.sqrt(max(now_s - pt[0], 1.0))), -4.0, 4.0)
        ests = [math.log(v.price) + v.basis for v in self.venues.values() if v.basis is not None]
        if len(ests) != 2:
            return None
        lag_s = max(self.now_ms - self.cl_spot_last_ms, 0) / 1000.0
        cl_last = self.cl_spot.get(self.cl_spot_last_ms // 1000)
        gbm = _clamp(rl.logit(pred.p_up, 1e-6), -7.0, 7.0)
        tau = _clamp(inp.tau_s / self.s.window_seconds, 0.0, 1.0)
        f = {
            "bias": 1.0,
            "gbm_logit": gbm,
            "gbm_z": _clamp((pred.mean - inp.log_strike) / pred.sd, -8.0, 8.0),
            "mkt_logit": mkt,
            "tau": tau,
            "known_w": inp.known_weight,
            "gbm_x_tau": gbm * tau,
            "mkt_x_tau": mkt * tau,
            "mkt_x_spread": mkt * spread,
            "spread": spread,
            "overround": _clamp(10.0 * (au + ad - 1.0), -1.0, 2.0) if au is not None and ad is not None else 0.0,
            "book_imb": (imb_u - imb_d) / 2,
            "depth_imb": (dep_u - dep_d) / 2,
            "trade_dev": _clamp(rl.logit(trade[0], 0.01) - mkt, -2.0, 2.0) if trade is not None else 0.0,
            "vol_level": _clamp(math.log(max(s_all, 1e-7) / 5e-5), -3.0, 3.0),
            "vol_ratio": _clamp(math.log(max(s_60, 1e-7) / max(s_all, 1e-7)), -2.0, 2.0) if s_60 is not None else 0.0,
            **mom,
            "venue_gap": _clamp((ests[0] - ests[1]) / inp.basis_sd, -4.0, 4.0),
            "basis_sd": _clamp(math.log(inp.basis_sd / 1e-4), -3.0, 3.0),
            "cl_lag": _clamp(lag_s / 5.0, 0.0, 3.0),
            "cl_gap": (_clamp((inp.log_spot_est - math.log(cl_last)) / (sig * math.sqrt(max(lag_s, 1.0))), -4.0, 4.0)
                       if cl_last is not None and cl_last > 0 else 0.0),
        }
        x = [f[n] for n in rl.FEATURES]
        if not all(math.isfinite(v) for v in x):
            return None
        return x, p_mkt

    def _rl_step(self, ms: MarketState, inp: ModelInput | None, band: tuple[Prediction, float, float] | None) -> None:
        """Once per ``rl_decision_interval_ms`` and live window: compute the inputs, keep a training sample
        every ``rl_sample_interval_ms``, and let every learning agent decide."""
        m = ms.info
        now = self.now_ms
        bucket = now - now % self.s.rl_decision_interval_ms
        if inp is None or band is None or bucket <= ms.last_rl_ms:
            return
        ms.last_rl_ms = bucket
        tau = m.end_ts - now / 1000.0
        if not (self.s.min_tau_s <= tau <= self.s.max_tau_s) or not m.accepting_orders or m.slug in self.rl_done:
            return
        if self.fresh(m) is not None:
            return  # same data requirements as trading, so training samples look like decisions
        pred = band[0]
        feats = self.rl_features(ms, inp, pred)
        if feats is None:
            self._inc("rl_skip_inputs")
            return
        x_all, p_mkt = feats
        sbucket = now - now % self.s.rl_sample_interval_ms
        sample = sbucket > ms.last_rl_sample_ms
        if sample:
            ms.last_rl_sample_ms = sbucket
            self.rl_experience.setdefault(m.slug, (m.end_ts, []))[1].append(x_all)
            # the yardsticks, sampled at the same moments as the agents
            self.store.insert_agent_prediction("market", m.slug, sbucket, tau, p_mkt)
            self.store.insert_agent_prediction(self.baseline_name, m.slug, sbucket, tau, pred.p_up)
        k = self.s.rl_pessimism
        for ag in self.rl_agents:
            mu, var = ag.learner.predict(ag.project(x_all))
            p_up = rl.predictive(mu, var)
            if sample:
                self.store.insert_agent_prediction(ag.name, m.slug, sbucket, tau, p_up)
            sd = math.sqrt(var)
            self._rl_decide(ms, ag, p_up, rl.sigmoid(mu - k * sd), rl.sigmoid(mu + k * sd), tau)

    def _rl_decide(self, ms: MarketState, ag: rl.RLAgent, p_up: float, lo: float, hi: float, tau: float) -> None:
        """The agent's policy: buy the side whose pessimistic P(win) beats the all-in cost by the required
        edge, through the same filters, sizing and fill simulation as the original strategy."""
        m = ms.info
        best: tuple[float, str, str, float, float, float, float] | None = None
        for side, token, p, p_worst in (("Up", m.token_up, p_up, lo), ("Down", m.token_down, 1.0 - p_up, 1.0 - hi)):
            p_win, p_safe = self._clip(p), self._clip(p_worst)
            book = self.books[token]
            ask = book.best_ask()
            if ask is None:
                continue
            edge = p_safe - all_in_cost_per_share(ask, m.fee_rate, m.fee_exponent)
            if edge < self.s.rl_min_edge:
                continue
            if self._market_disagrees(p_win, book.best_bid(), ask, f"{ag.name}:skip_model_market_gap"):
                continue
            if best is None or edge > best[0]:
                best = (edge, side, token, p_win, p_safe, ask, book.tick or m.tick_size)
        if best is None:
            return
        edge, side, token, p_win, p_safe, ask, tick = best
        ag.note_virtual(m.slug, side, p_win, m.end_ts)  # the policy's choice in this window, traded or not
        if ag.learner.windows < self.s.rl_min_windows:
            self._inc(f"{ag.name}:skip_warmup")
            return
        penalty = ag.penalty(self.s.rl_bias_prior_trades)
        need = self.s.rl_min_edge + penalty
        if edge < need:
            self._inc(f"{ag.name}:skip_penalty")
            return
        if self.store.entries_for(m.slug, ag.name) >= self.s.max_entries_per_market:
            return
        if any(po.slug == m.slug and po.agent == ag.name for po in self.pending):
            return
        if len(self.rtts) < 5:
            self._inc(f"{ag.name}:skip_no_latency_samples")
            return
        limit = floor_to_tick(ask + self.s.limit_slippage_ticks * tick, tick)
        while limit > ask + 1e-9 and p_safe - all_in_cost_per_share(limit, m.fee_rate, m.fee_exponent) < need:
            limit = round(limit - tick, 6)
        limit = min(limit, 0.99)
        budget = self.stake_budget(p_safe - penalty, all_in_cost_per_share(limit, m.fee_rate, m.fee_exponent))
        pre = simulate_taker_buy(self._asks_for(token), limit, budget, m.min_size, m.fee_rate, m.fee_exponent,
                                 self.s.book_take_fraction, self.s.fee_round_up)
        if not pre.levels:
            self._inc(f"{ag.name}:skip_{pre.reason}")
            return
        self._submit(ms, side, token, "taker", p_win, ask, limit, edge, tau, budget, ag.name)

    def _rl_learn(self, slug: str, winner: str) -> None:
        """The official outcome is the reward signal: every agent learns from all samples of the window and
        scores the trade its policy chose there (if any). Then the learned state is saved."""
        if not self.rl_agents or winner not in ("Up", "Down") or slug in self.rl_done:
            return
        exp = self.rl_experience.pop(slug, None)
        ms = self.markets.get(slug)
        self.rl_done[slug] = exp[0] if exp is not None else (ms.info.end_ts if ms else self.now_ms // 1000)
        y = 1 if winner == "Up" else 0
        for ag in self.rl_agents:
            learned = False
            if exp is not None and exp[1]:
                learned = ag.learner.update([ag.project(x) for x in exp[1]], y)
                if learned:
                    self._inc(f"{ag.name}:windows_learned")
                else:
                    self._inc(f"{ag.name}:update_rejected")
                    log.warning("%s: update for %s rejected as numerically unsound; kept the old state", ag.name, slug)
            scored = ag.learn_outcome(slug, winner, self.s.rl_memory_windows)
            if learned or scored:
                self.store.set_rl_state(ag.name, ag.to_json(), self.now_ms)

    def sample_latency_ms(self, sid: str) -> float:
        rng = random.Random(_seed_for(self.s.seed, sid))
        base = rng.choice(list(self.rtts))
        jitter = math.exp(rng.gauss(0.0, self.s.latency_jitter_sigma))
        return max(self.s.latency_ms_floor, base * jitter + self.s.latency_extra_ms)

    def _submit(self, ms: MarketState, side: str, token: str, kind: str, p_win: float, ask: float,
                limit: float, edge: float, tau: float, budget: float, agent: str) -> None:
        sid = f"{ms.info.slug}:{self.now_ms}:{side}:{kind}"
        if agent != self.baseline_name:  # the original strategy keeps its ids (and so its latency draws)
            sid += f":{agent}"
        lat = self.sample_latency_ms(sid)
        arrival = self.now_ms + int(math.ceil(lat))
        ok = self.store.insert_signal(dict(
            id=sid, slug=ms.info.slug, ts_ms=self.now_ms, side=side, kind=kind, p_win=p_win, best_ask=ask,
            limit_price=limit, edge=edge, tau_s=tau, latency_ms=lat, arrival_ms=arrival, budget=budget,
            status="pending", detail="{}", agent=agent,
        ))
        if not ok:
            return
        self._inc("signals")
        if agent != self.baseline_name:
            self._inc(f"{agent}:signals")
        self.last_decision = dict(id=sid, agent=agent, side=side, kind=kind, p_win=round(p_win, 4), ask=ask,
                                  limit=limit, edge=round(edge, 4), latency_ms=round(lat, 1))
        log.info("signal %s", self.last_decision)
        self.pending.append(PendingOrder(sid, ms.info.slug, side, token, kind, arrival, limit, budget, p_win, agent))

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
        """The ordering agent's own (central) P(win), re-evaluated when the order reaches the book."""
        ms = self.markets.get(po.slug)
        inp = self.model_input(ms) if ms else None
        if ms is None or inp is None:
            return None
        pred = self.model.predict(inp)
        p_up = pred.p_up
        ag = self.rl_by_name.get(po.agent)
        if ag is not None:
            feats = self.rl_features(ms, inp, pred)
            if feats is None:
                return None
            p_up = rl.predictive(*ag.learner.predict(ag.project(feats[0])))
        p = p_up if po.side == "Up" else 1.0 - p_up
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
        asks = self._asks_for(po.token)
        fill = simulate_taker_buy(asks, po.limit_price, po.budget, m.min_size, m.fee_rate, m.fee_exponent,
                                  self.s.book_take_fraction, self.s.fee_round_up)
        if not fill.levels:
            self.store.update_signal(po.sid, "missed", {"reason": fill.reason, "p_win_arrival": p_arr,
                                                        "best_ask_arrival": book.best_ask() if book else None})
            self._inc("missed")
            return
        taken = self.own_taken.setdefault(po.token, {})
        for lv in fill.levels:
            taken[lv.price] = taken.get(lv.price, 0.0) + lv.shares
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
            p_win_arrival=p_arr, edge_arrival=edge_arr, levels=json.dumps(levels), agent=po.agent,
        )):
            self.open_slugs.add(po.slug)
            self._inc("fills")
            log.info("fill %s agent=%s side=%s net=%.2f cost=%.4f fee=%.5f avg=%.4f", fid, po.agent, po.side, net,
                     cost, fee, avg)

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
        self._rl_learn(r.slug, r.winner)

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
            agents=list(self.agent_names),
            rl={a.name: dict(windows_learned=a.learner.windows, trading=a.learner.windows >= self.s.rl_min_windows,
                             warmup_windows=self.s.rl_min_windows,
                             penalty=round(a.penalty(self.s.rl_bias_prior_trades), 4),
                             judged_trades=round(a.bias_n, 2), unresolved_windows=len(a.virtual))
                for a in self.rl_agents},
            counters=dict(self.counters),
            parse=dict(ok=self.stats.ok, ignored=self.stats.ignored, errors=self.stats.errors,
                       last_error=self.stats.last_error),
            last_decision=self.last_decision,
        )
