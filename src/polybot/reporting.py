"""Honest performance report computed from the SQLite store.

All agents share one paper account. Top-ups (deposits) add to cash and equity but are not returns: the
%/day, drawdown and per-trade returns are time-weighted, i.e. each settled trade's P&L is divided by the
equity at that moment (deposits included), and those returns are compounded.
"""

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
CALIBRATION_SAMPLE_MS = 10_000  # one prediction per 10 s: neighbours add load, not information
SCORE_SAMPLE_MS = 30_000  # models-vs-market scoreboard: one sample per 30 s of each window ...
SCORE_DAYS = 7  # ... over the last 7 days (the learning agents change; recent scores matter)


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


def _log_loss(pairs: list[tuple[float, int]]) -> float | None:
    if not pairs:
        return None
    eps = 1e-6
    return -sum(math.log(min(max(p if o else 1.0 - p, eps), 1.0)) for p, o in pairs) / len(pairs)


def _agents(trades: list[Any], signals: list[Any]) -> list[dict[str, Any]]:
    """Per-agent results; the money is shared, so these are attributions of one account's P&L."""
    acc: dict[str, dict[str, Any]] = {}

    def row(agent: str) -> dict[str, Any]:
        return acc.setdefault(agent, dict(agent=agent, signals=0, by_status=defaultdict(int), filled=0, settled=0,
                                          open=0, wins=0, pnl=0.0, fees=0.0, staked=0.0, pairs=[]))

    for sg in signals:
        a = row(sg["agent"])
        a["signals"] += 1
        a["by_status"][sg["status"]] += 1
    for t in trades:
        a = row(t["agent"])
        a["filled"] += 1
        a["fees"] += float(t["fee"])
        if t["settled_by"] is None:
            a["open"] += 1
            continue
        a["settled"] += 1
        a["wins"] += int(t["won"])
        a["pnl"] += float(t["pnl"])
        a["staked"] += float(t["cost"])
        a["pairs"].append((float(t["p_win_decision"]), int(t["won"])))
    out = []
    for name in sorted(acc):
        a = acc[name]
        pairs = a.pop("pairs")
        n = a["settled"]
        out.append(dict(a, by_status=dict(a["by_status"]), pnl=round(a["pnl"], 4), fees=round(a["fees"], 5),
                        staked=round(a["staked"], 4), win_rate=(a["wins"] / n) if n else None,
                        predicted_win_rate=statistics.fmean(p for p, _ in pairs) if pairs else None,
                        return_on_staked=(a["pnl"] / a["staked"]) if a["staked"] > 0 else None,
                        brier_traded=_brier(pairs)))
    return out


def _model_scores(store: Storage, now_ms: int) -> dict[str, Any]:
    """Calibration of every predictor sampled together (the market's own P(Up), gbm_twap and each RL agent):
    the same windows at the same moments, so the scores are comparable. Lower is better."""
    rows = store.query(
        "SELECT a.agent, a.p_up, m.winner = 'Up' FROM agent_predictions a JOIN markets m ON m.slug=a.slug "
        f"WHERE m.winner IN ('Up','Down') AND a.ts_ms >= ? AND a.ts_ms % {SCORE_SAMPLE_MS} = 0",
        (now_ms - SCORE_DAYS * 86_400_000,),
    )
    by: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for agent, p_up, up in rows:
        by[agent].append((float(p_up), int(up)))
    return dict(
        note=f"P(Up) of each predictor at the same moments (one per {SCORE_SAMPLE_MS // 1000} s of the live windows "
             f"the agents could trade, last {SCORE_DAYS} days), settled on the official outcome. 'market' is the mid "
             "of the Up and Down books.",
        rows=[dict(model=k, n=len(v), brier=_brier(v), log_loss=_log_loss(v), table=_reliability(v))
              for k, v in sorted(by.items())],
    )


def _rl_status(store: Storage) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in store.query("SELECT agent, state, updated_ms FROM rl_state ORDER BY agent"):
        try:
            st = json.loads(r["state"])
            lr = st["learner"]
            weights = {k: round(float(w), 4) for k, w in zip(lr["names"], lr["mean"], strict=True)}
            out[r["agent"]] = dict(windows_learned=int(lr["windows"]), updated_ms=r["updated_ms"], weights=weights,
                                   judged_trades=round(float(st["bias_n"]), 2),
                                   mean_outcome_minus_predicted=(float(st["bias_sum"]) / float(st["bias_n"])
                                                                 if float(st["bias_n"]) > 0 else None))
        except (ValueError, KeyError, TypeError):
            out[r["agent"]] = dict(error="unreadable state")
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

    deposits = store.query("SELECT id, ts_ms, amount FROM deposits ORDER BY ts_ms, id")
    deposited = sum(float(d["amount"]) for d in deposits)
    cash = store.cash(starting_bankroll)
    open_cost = sum(float(t["cost"]) for t in open_)
    equity = cash + open_cost  # open positions valued at cost; includes deposits
    total_pnl = sum(float(t["pnl"]) for t in settled)
    fees = sum(float(t["fee"]) for t in trades)

    # Account equity curve (deposits show as steps) and the time-weighted growth index (deposits excluded):
    # every settled trade's return is its P&L over the equity at that moment.
    srt = sorted(settled, key=lambda r: r["settle_ms"])
    flows = sorted([(int(d["ts_ms"]), 0, i, float(d["amount"])) for i, d in enumerate(deposits)]
                   + [(int(t["settle_ms"]), 1, i, float(t["pnl"])) for i, t in enumerate(srt)])
    eq, growth, peak, max_dd = starting_bankroll, 1.0, 1.0, 0.0
    curve = [(first_start, eq)]
    per_trade_ret: list[float] = []
    day_growth: dict[str, float] = defaultdict(lambda: 1.0)
    for ts, kind, _, x in flows:
        if kind == 1:
            r = x / eq if eq > 0 else 0.0
            per_trade_ret.append(r)
            growth *= 1.0 + r
            day_growth[_day(ts)] *= 1.0 + r
            peak = max(peak, growth)
            if peak > 0:
                max_dd = max(max_dd, (peak - growth) / peak)
        eq += x
        curve.append((ts, round(eq, 4)))
    try:
        pct_day: float | None = (growth ** (1 / elapsed_days) - 1) if growth > 0 else -1.0
    except OverflowError:  # a gain over a tiny elapsed time (e.g. just after a reset) has no meaningful rate
        pct_day = None

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
        f"WHERE m.winner IN ('Up','Down') AND p.ts_ms % {CALIBRATION_SAMPLE_MS} = 0"
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
        rets = [day_growth[day] - 1.0 for day in days]
        rng = random.Random(1)
        sims = sorted(statistics.fmean(rng.choice(rets) for _ in rets) for _ in range(n_boot))
        boot["day_level"] = dict(days=len(rets), mean=statistics.fmean(rets),
                                 ci95=[sims[int(0.025 * n_boot)], sims[int(0.975 * n_boot) - 1]])

    signals = store.query("SELECT agent, kind, status, detail, latency_ms FROM signals")
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
        bankroll=dict(start=starting_bankroll, deposits=round(deposited, 4), cash=round(cash, 4),
                      open_cost=round(open_cost, 4), equity=round(equity, 4), shared_by_all_agents=True),
        deposits=[dict(ts_ms=int(d["ts_ms"]), amount=float(d["amount"])) for d in deposits][-100:],
        pnl=dict(settled_after_fees=round(total_pnl, 4), fees_paid=round(fees, 5),
                 pct_per_day_compounded_wallclock=pct_day, target_pct_per_day=target_daily,
                 max_drawdown=max_dd, note="time-weighted: deposits are not counted as returns"),
        agents=_agents(list(trades), list(signals)),
        model_scores=_model_scores(store, now_ms),
        rl=_rl_status(store),
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
