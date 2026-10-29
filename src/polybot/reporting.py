"""Honest performance report computed from the SQLite store."""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from .storage import Storage

Z95 = 1.959964
Z80_POWER = 0.841621


def _day(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=UTC).strftime("%Y-%m-%d")


def _brier(pairs: list[tuple[float, int]]) -> float | None:
    return sum((p - o) ** 2 for p, o in pairs) / len(pairs) if pairs else None


def _reliability(pairs: list[tuple[float, int]], bins: int = 10) -> list[dict[str, Any]]:
    acc: dict[int, list[tuple[float, int]]] = defaultdict(list)
    for p, o in pairs:
        acc[min(int(p * bins), bins - 1)].append((p, o))
    out = []
    for b in range(bins):
        rows = acc.get(b, [])
        if not rows:
            continue
        n = len(rows)
        freq = sum(o for _, o in rows) / n
        out.append(dict(bin=f"{b / bins:.1f}-{(b + 1) / bins:.1f}", n=n,
                        mean_pred=round(sum(p for p, _ in rows) / n, 4), actual=round(freq, 4),
                        se=round(math.sqrt(max(freq * (1 - freq), 1e-9) / n), 4)))
    return out


def build_report(store: Storage, starting_bankroll: float, now_ms: int, target_daily: float = 0.005,
                 n_boot: int = 2000) -> dict[str, Any]:
    first_start = int(store.get_meta("first_start_ms") or now_ms)
    elapsed_days = max((now_ms - first_start) / 86_400_000, 1e-9)

    trades = store.query(
        "SELECT f.*, s.won, s.payout, s.pnl, s.settled_by, s.ts_ms AS settle_ms FROM fills f "
        "LEFT JOIN settlements s ON s.id=f.id ORDER BY f.ts_ms"
    )
    settled = [t for t in trades if t["settled_by"] is not None]
    open_ = [t for t in trades if t["settled_by"] is None]

    cash = store.cash(starting_bankroll)
    open_cost = sum(float(t["cost"]) for t in open_)
    equity = cash + open_cost  # open positions valued at cost
    total_pnl = sum(float(t["pnl"]) for t in settled)
    fees = sum(float(t["fee"]) for t in trades)

    growth = equity / starting_bankroll
    pct_day = (growth ** (1 / elapsed_days) - 1) if growth > 0 else -1.0

    # equity curve on settlement order, drawdown
    eq, peak, max_dd = starting_bankroll, starting_bankroll, 0.0
    curve = [(first_start, eq)]
    per_trade_ret: list[float] = []
    for t in sorted(settled, key=lambda r: r["settle_ms"]):
        per_trade_ret.append(float(t["pnl"]) / eq)
        eq += float(t["pnl"])
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak)
        curve.append((int(t["settle_ms"]), round(eq, 4)))

    per_day: dict[str, dict[str, float]] = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "fees": 0.0})
    for t in settled:
        d = per_day[_day(int(t["settle_ms"]))]
        d["pnl"] += float(t["pnl"])
        d["trades"] += 1
        d["fees"] += float(t["fee"])

    # win rate vs predicted
    n = len(settled)
    wins = sum(int(t["won"]) for t in settled)
    pred_dec = [float(t["p_win_decision"]) for t in settled]
    pred_arr = [float(t["p_win_arrival"]) for t in settled if t["p_win_arrival"] is not None]
    traded_pairs = [(float(t["p_win_decision"]), int(t["won"])) for t in settled]

    # calibration on ALL periodic predictions (not just traded ones, which are selected)
    pred_rows = store.query(
        "SELECT p.p_up, m.winner FROM predictions p JOIN markets m ON m.slug=p.slug "
        "WHERE m.winner IN ('Up','Down')"
    )
    all_pairs = [(float(r["p_up"]), 1 if r["winner"] == "Up" else 0) for r in pred_rows]

    # bootstrap of daily return from per-trade returns
    trades_per_day = n / elapsed_days
    boot: dict[str, Any] = {}
    if n >= 2:
        rng = random.Random(0)
        mean_r = statistics.fmean(per_trade_ret)
        sd_r = statistics.stdev(per_trade_ret)
        sims = []
        for _ in range(n_boot):
            sample = [per_trade_ret[rng.randrange(n)] for _ in range(n)]
            sims.append(statistics.fmean(sample) * trades_per_day)
        sims.sort()
        boot = dict(
            method="resample trades; daily return = mean trade return x observed trades/day",
            daily_mean=mean_r * trades_per_day,
            ci95=[sims[int(0.025 * n_boot)], sims[int(0.975 * n_boot) - 1]],
            per_trade_mean=mean_r, per_trade_sd=sd_r,
        )
        if trades_per_day > 0 and sd_r > 0:
            mu_target = target_daily / trades_per_day  # per-trade mean needed for the daily target
            boot["trades_needed_to_detect_target"] = math.ceil((Z95 * sd_r / mu_target) ** 2)
            boot["trades_needed_80pct_power"] = math.ceil(((Z95 + Z80_POWER) * sd_r / mu_target) ** 2)
            boot["days_needed_at_current_rate"] = round(boot["trades_needed_80pct_power"] / trades_per_day, 1)
    days = sorted(per_day)
    if len(days) >= 5:
        rets, e = [], starting_bankroll
        for day in days:
            rets.append(per_day[day]["pnl"] / e)
            e += per_day[day]["pnl"]
        rng = random.Random(1)
        sims = sorted(statistics.fmean(rng.choice(rets) for _ in rets) for _ in range(n_boot))
        boot["day_level"] = dict(days=len(rets), mean=statistics.fmean(rets),
                                 ci95=[sims[int(0.025 * n_boot)], sims[int(0.975 * n_boot) - 1]])

    signals = store.query("SELECT kind, status, detail, latency_ms FROM signals")
    done = [s for s in signals if s["status"] not in ("pending", "resting")]
    vanished = 0
    for s in done:
        d = json.loads(s["detail"] or "{}")
        if s["status"] in ("missed", "expired", "cancelled_restart") or (
            d.get("edge_arrival") is not None and d["edge_arrival"] <= 0
        ):
            vanished += 1
    lats = sorted(float(s["latency_ms"]) for s in signals)
    status_counts: dict[str, int] = defaultdict(int)
    for s in signals:
        status_counts[s["status"]] += 1

    settled_by: dict[str, int] = defaultdict(int)
    for t in settled:
        settled_by[t["settled_by"]] += 1

    mism = store.query(
        "SELECT slug, strike, gamma_strike FROM markets WHERE strike IS NOT NULL AND gamma_strike IS NOT NULL"
    )
    mismatches = [dict(slug=r["slug"], ours=r["strike"], gamma=r["gamma_strike"]) for r in mism
                  if abs(r["strike"] - r["gamma_strike"]) > 1e-6 * r["gamma_strike"]]

    meaningful = n >= int(boot.get("trades_needed_80pct_power", 10**9)) and elapsed_days >= 7
    return dict(
        label=("" if meaningful else
               "STATISTICALLY MEANINGLESS SAMPLE - do not draw conclusions from these numbers"),
        generated_ms=now_ms,
        elapsed_days=round(elapsed_days, 4),
        bankroll=dict(start=starting_bankroll, cash=round(cash, 4), open_cost=round(open_cost, 4),
                      equity=round(equity, 4)),
        pnl=dict(settled_after_fees=round(total_pnl, 4), fees_paid=round(fees, 5),
                 pct_per_day_compounded_wallclock=pct_day, target_pct_per_day=target_daily,
                 max_drawdown=max_dd),
        trades=dict(filled=len(trades), settled=n, open=len(open_), wins=wins,
                    win_rate=(wins / n) if n else None,
                    predicted_win_rate_decision=statistics.fmean(pred_dec) if pred_dec else None,
                    predicted_win_rate_arrival=statistics.fmean(pred_arr) if pred_arr else None,
                    settled_by=dict(settled_by)),
        calibration=dict(
            traded=dict(n=len(traded_pairs), brier=_brier(traded_pairs), table=_reliability(traded_pairs)),
            all_predictions=dict(n=len(all_pairs), brier=_brier(all_pairs), table=_reliability(all_pairs),
                                 note="periodic predictions every 10 s for every window, settled on outcome"),
        ),
        bootstrap=boot,
        signals=dict(total=len(signals), by_status=dict(status_counts),
                     edge_vanished_after_latency=vanished, edge_vanished_share=(vanished / len(done)) if done else None,
                     latency_ms=dict(p50=lats[len(lats) // 2], p90=lats[int(len(lats) * 0.9)], max=lats[-1])
                     if lats else None),
        per_day={d: {k: round(v, 4) for k, v in per_day[d].items()} for d in days},
        equity_curve=curve[-2000:],
        strike_mismatches=mismatches,
    )
