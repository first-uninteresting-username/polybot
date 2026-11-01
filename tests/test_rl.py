"""Online-learning agents, the shared bankroll and top-ups."""

import json
import math
import random
import sqlite3
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer
from synth import START, generate, generate_windows

from polybot import rl
from polybot.config import load_settings
from polybot.engine import Engine, PendingOrder
from polybot.events import MAX_DEPOSIT_USDC, ParseStats, parse_raw
from polybot.recorder import read_recording, recording_files
from polybot.replay import replay
from polybot.reporting import build_report
from polybot.storage import Storage
from polybot.web import make_app

MECH = dict(min_vol_samples=60, max_model_market_gap=0)  # settings the single synthetic window trades under


def run(records, store=None, **kw):
    store = store or Storage(":memory:")
    store.set_meta("first_start_ms", str(records[0][0]), only_if_missing=True)  # as replay() does
    e = Engine(load_settings(env={}, **kw), store)
    for t, s, r in records:
        e.on_raw(s, t, r)
    return e


def deposit(t_ms: int, amount: float, dep_id: str = "d1") -> tuple[int, str, str]:
    return (t_ms, "deposit", json.dumps({"id": dep_id, "amount": amount}))


@pytest.fixture(scope="module")
def windows30():
    return generate_windows(30)


@pytest.fixture(scope="module")
def trained(windows30):
    """All three agents over 30 windows; the RL agents may trade after learning from 10."""
    return run(windows30, rl_min_windows=10)


# ---------------------------------------------------------------- learner


def test_linear_algebra() -> None:
    rng = random.Random(1)
    for d in (1, 2, 5, 25):
        a = [[rng.gauss(0, 1) for _ in range(d)] for _ in range(d)]
        spd = [[sum(a[i][k] * a[j][k] for k in range(d)) + (0.5 if i == j else 0.0) for j in range(d)] for i in range(d)]
        low = rl.cholesky(spd)
        inv = rl.chol_inverse(low)
        for i in range(d):
            for j in range(d):
                assert sum(spd[i][k] * inv[k][j] for k in range(d)) == pytest.approx(float(i == j), abs=1e-10)
        b = [rng.gauss(0, 1) for _ in range(d)]
        x = rl.chol_solve(low, b)
        assert [sum(spd[i][k] * x[k] for k in range(d)) for i in range(d)] == pytest.approx(b, abs=1e-10)
    for bad in ([[1.0, 2.0], [2.0, 1.0]], [[0.0]], [[float("nan")]]):
        with pytest.raises(ValueError):
            rl.cholesky(bad)


def test_prior_is_half_model_half_market() -> None:
    lr = rl.OnlineLogit(rl.PRESETS["rl_core"], 0.25, 0)
    x = [1.0, 2.0, -1.0, 0.0, 0.0]  # bias, gbm_logit, mkt_logit, interactions
    mu, var = lr.predict(x)
    assert mu == pytest.approx(0.5 * 2.0 + 0.5 * -1.0)
    assert var == pytest.approx(sum(v * v for v in x) * 0.25**2)


def test_newton_steps_are_converged(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = random.Random(2)
    names = [f"f{i}" for i in range(25)]
    xs = [[1.0] + [rng.gauss(0, 1.5) for _ in range(24)] for _ in range(60)]
    a, b = rl.OnlineLogit(names, 0.25, 0), rl.OnlineLogit(names, 0.25, 0)
    assert a.update(xs, 1)
    monkeypatch.setattr(rl, "NEWTON_STEPS", 50)
    assert b.update(xs, 1)
    assert a.mean == pytest.approx(b.mean, abs=1e-12)


def _exact_posterior_mode(windows: list, d: int, prior_sd: float) -> list[float]:
    """Batch MAP of the same objective the online learner approximates (zero prior mean, windows weighted 1/n)."""
    q0, w = 1 / prior_sd**2, [0.0] * d
    for _ in range(12):
        g = [q0 * wi for wi in w]
        h = [[q0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        for xs, y in windows:
            for x in xs:
                p = rl.sigmoid(sum(a * b for a, b in zip(w, x, strict=True)))
                for i in range(d):
                    g[i] += (p - y) * x[i] / len(xs)
                    for j in range(d):
                        h[i][j] += p * (1 - p) * x[i] * x[j] / len(xs)
        w = [a - b for a, b in zip(w, rl.chol_solve(rl.cholesky(h), g), strict=True)]
    return w


def test_learner_tracks_the_exact_posterior_and_learns_the_signal() -> None:
    rng = random.Random(3)
    true_w = [0.0, 1.2, -0.7, 0.4, 0.0, 0.0]
    lr = rl.OnlineLogit([f"g{i}" for i in range(6)], 0.25, 0)
    windows, loss_prior, loss_learned = [], 0.0, 0.0
    for k in range(900):
        x = [1.0] + [rng.gauss(0, 1) for _ in range(5)]
        y = int(rng.random() < rl.sigmoid(sum(w * v for w, v in zip(true_w, x, strict=True))))
        if k >= 600:
            q = rl.sigmoid(lr.predict(x)[0])
            loss_learned -= math.log(q if y else 1 - q)
            loss_prior -= math.log(0.5)
        # a window's samples: small wiggles around the same state, one outcome
        xs = [[1.0] + [v + rng.gauss(0, 0.05) for v in x[1:]] for _ in range(4)]
        windows.append((xs, y))
        assert lr.update(xs, y)
    assert lr.windows == 900
    assert lr.mean == pytest.approx(_exact_posterior_mode(windows, 6, 0.25), abs=0.06)
    w = lr.mean
    assert w[1] > w[3] > 0.2 and w[2] < -0.4 and max(abs(w[0]), abs(w[4]), abs(w[5])) < 0.15
    assert loss_learned < loss_prior - 0.08 * 300  # out of sample


def test_a_window_counts_once_however_many_samples() -> None:
    x = [1.0, 0.3, -0.2, 0.1, 0.05]
    one, many = rl.OnlineLogit(rl.PRESETS["rl_core"], 0.25, 0), rl.OnlineLogit(rl.PRESETS["rl_core"], 0.25, 0)
    one.update([x], 1)
    many.update([x] * 300, 1)
    assert many.mean == pytest.approx(one.mean, abs=1e-12)
    assert [v for row in many.prec for v in row] == pytest.approx([v for row in one.prec for v in row], abs=1e-9)


def test_update_refuses_bad_input_and_changes_nothing() -> None:
    lr = rl.OnlineLogit(rl.PRESETS["rl_core"], 0.25, 0)
    before = (list(lr.mean), [r[:] for r in lr.prec])
    assert not lr.update([], 1)
    assert not lr.update([[1.0, 0.0]], 1)  # wrong length
    assert not lr.update([[1.0] * 5], 2)  # not an outcome
    assert not lr.update([[1.0, float("nan"), 0.0, 0.0, 0.0]], 1)
    assert (lr.mean, lr.prec, lr.windows) == (before[0], before[1], 0)


def test_forgetting_relaxes_towards_the_prior() -> None:
    x = [1.0, 1.0, 1.0, 0.5, 0.5]
    keep, forget = rl.OnlineLogit(rl.PRESETS["rl_core"], 0.25, 0), rl.OnlineLogit(rl.PRESETS["rl_core"], 0.25, 10)
    for _ in range(50):
        keep.update([x], 1)
        forget.update([x], 1)
    var_keep, var_forget = keep.predict(x)[1], forget.predict(x)[1]
    assert var_forget > var_keep  # it stays readier to learn


def test_state_round_trip_and_rejections() -> None:
    ag = rl.make_agent("rl_wide", 0.25, 2016)
    rng = random.Random(4)
    for _ in range(5):
        ag.learner.update([[1.0] + [rng.gauss(0, 1) for _ in range(len(rl.FEATURES) - 1)] for _ in range(10)], 1)
    ag.note_virtual("s", "Up", 0.7, 0)
    ag.learn_outcome("s", "Down", 2016)
    raw = ag.to_json()
    back = rl.make_agent("rl_wide", 0.25, 2016)
    assert back.load_json(raw)
    x = [rng.gauss(0, 1) for _ in rl.FEATURES]
    assert back.learner.predict(x) == ag.learner.predict(x)  # bit-identical after a restart
    assert (back.bias_sum, back.bias_n, back.learner.windows) == (ag.bias_sum, ag.bias_n, 5)
    other = rl.make_agent("rl_core", 0.25, 2016)  # different inputs: must not adopt it
    assert not other.load_json(raw) and other.learner.windows == 0
    for junk in ("", "[]", "{}", '{"version": 1}', raw.replace('"version": 1', '"version": 99')):
        fresh = rl.make_agent("rl_wide", 0.25, 2016)
        assert not fresh.load_json(junk) and fresh.learner.windows == 0


def test_penalty_learns_from_the_policys_own_trades() -> None:
    ag = rl.make_agent("rl_core", 0.25, 0)
    assert ag.penalty(30) == 0.0
    for k in range(40):  # trades predicted to win 70% of the time win 40%
        ag.note_virtual(f"s{k}", "Up", 0.7, 0)
        ag.note_virtual(f"s{k}", "Down", 0.9, 0)  # only the first choice in a window counts
        ag.learn_outcome(f"s{k}", "Up" if k % 5 < 2 else "Down", 0)
    assert ag.penalty(30) == pytest.approx(0.3 * 40 / 70)
    assert not ag.learn_outcome("never-noted", "Up", 0)
    lucky = rl.make_agent("rl_core", 0.25, 0)
    for k in range(40):
        lucky.note_virtual(f"s{k}", "Up", 0.6, 0)
        lucky.learn_outcome(f"s{k}", "Up", 0)
    assert lucky.penalty(30) == 0.0  # good luck never loosens the rules


# ---------------------------------------------------------------- engine


def test_unknown_or_duplicate_agents_are_rejected() -> None:
    for bad in ("gbm_twap,nope", "rl_core,rl_core"):
        with pytest.raises(ValueError):
            Engine(load_settings(env={}, agents=bad), Storage(":memory:"))
    e = Engine(load_settings(env={}, agents=" rl_core , gbm_twap "), Storage(":memory:"))
    assert e.agent_names == ["rl_core", "gbm_twap"] and e.baseline_on


def test_features_are_finite_and_match_the_books(windows30) -> None:
    e = run([r for r in windows30 if r[0] < (START + 100) * 1000])
    ms = next(m for m in e.markets.values() if m.info.start_ts == START)
    inp = e.model_input(ms)
    x, p_mkt = e.rl_features(ms, inp, e.model.predict(inp))
    assert len(x) == len(rl.FEATURES) and all(math.isfinite(v) for v in x)
    f = dict(zip(rl.FEATURES, x, strict=True))
    up, dn = e.books[ms.info.token_up], e.books[ms.info.token_down]
    mid = ((up.best_bid() + up.best_ask()) / 2 + 1 - (dn.best_bid() + dn.best_ask()) / 2) / 2
    assert p_mkt == pytest.approx(mid) and f["mkt_logit"] == pytest.approx(math.log(mid / (1 - mid)))
    assert f["bias"] == 1.0 and f["tau"] == pytest.approx(inp.tau_s / 300)
    assert f["spread"] == pytest.approx(10 * 0.02)  # the synthetic books are quoted 2 ticks wide
    assert f["gbm_x_tau"] == pytest.approx(f["gbm_logit"] * f["tau"])


def test_rl_agents_learn_trade_and_share_the_money(trained) -> None:
    e = trained
    resolved = e.counters["settled_gamma"]
    for name in ("rl_wide", "rl_core"):
        st = json.loads(e.store.get_rl_state(name))
        assert st["learner"]["windows"] == e.counters[f"{name}:windows_learned"] == resolved
        sigs = e.store.query("SELECT * FROM signals WHERE agent=?", (name,))
        assert sigs and all(s["id"].endswith(":" + name) for s in sigs)
        # no trade before it had learned from rl_min_windows (10) windows
        tenth = sorted(int(r["resolved_ms"]) for r in e.store.query(
            "SELECT resolved_ms FROM markets WHERE settled_by='gamma'"))[9]
        assert min(s["ts_ms"] for s in sigs) >= tenth
    fills = e.store.query("SELECT * FROM fills")
    assert {f["agent"] for f in fills} == {"gbm_twap", "rl_wide", "rl_core"}
    for f in fills:  # every fill is labelled with the agent of its signal
        (sig,) = e.store.query("SELECT agent FROM signals WHERE id=?", (f["id"],))
        assert sig["agent"] == f["agent"]
    # one cash pool for everyone
    paid = sum(r["payout"] for r in e.store.query("SELECT payout FROM settlements"))
    assert e.store.cash(1000) == pytest.approx(1000 - sum(f["cost"] for f in fills) + paid)
    rep = build_report(e.store, 1000, e.now_ms)
    by = {a["agent"]: a for a in rep["agents"]}
    assert set(by) == {"gbm_twap", "rl_wide", "rl_core"}
    assert sum(a["pnl"] for a in by.values()) == pytest.approx(rep["pnl"]["settled_after_fees"], abs=1e-3)
    assert {r["model"] for r in rep["model_scores"]["rows"]} == {"market", "gbm_twap", "rl_wide", "rl_core"}
    assert set(rep["rl"]) == {"rl_wide", "rl_core"} and rep["rl"]["rl_core"]["windows_learned"] == resolved


def test_rl_agents_wait_for_warmup_but_still_learn(windows30) -> None:
    e = run(windows30[: len(windows30) // 2], rl_min_windows=10**6)
    assert e.store.query("SELECT * FROM signals WHERE agent != 'gbm_twap'") == []
    assert e.counters.get("rl_core:skip_warmup", 0) > 0
    assert e.rl_by_name["rl_core"].learner.windows > 0


def test_learning_moves_away_from_a_noisy_market() -> None:
    """The synthetic market quotes the true P(Up) with noise, the model sees it without: both learners must
    lean less on the market than their prior did, and beat it out of sample."""
    e = run(generate_windows(40, market_noise=1.0), rl_min_windows=10**6)
    for name in ("rl_core", "rl_wide"):
        w = dict(zip(e.rl_by_name[name].learner.names, e.rl_by_name[name].learner.mean, strict=True))
        assert w["mkt_logit"] < 0.42 < w["gbm_logit"]  # both priors were 0.5
    scores = {r["model"]: r["log_loss"] for r in build_report(e.store, 1000, e.now_ms)["model_scores"]["rows"]}
    assert scores["rl_core"] < scores["market"] and scores["rl_wide"] < scores["market"]


def test_deterministic_and_restart_keeps_what_was_learned(tmp_path: Path, windows30, trained) -> None:
    again = run(windows30, rl_min_windows=10)
    for q in ("SELECT * FROM rl_state", "SELECT * FROM signals ORDER BY id", "SELECT * FROM fills ORDER BY id",
              "SELECT * FROM agent_predictions ORDER BY agent, slug, ts_ms"):
        assert [tuple(r) for r in again.store.query(q)] == [tuple(r) for r in trained.store.query(q)]
    db = tmp_path / "s.sqlite"
    first = run(windows30[: len(windows30) // 2], Storage(db))
    saved = {n: first.rl_by_name[n].to_json() for n in ("rl_wide", "rl_core")}
    first.store.close()
    restarted = Engine(load_settings(env={}), Storage(db))
    for n in ("rl_wide", "rl_core"):
        assert restarted.rl_by_name[n].to_json() == saved[n]
        assert restarted.rl_by_name[n].learner.windows > 0


def test_untrained_rl_agents_do_not_disturb_the_original_strategy() -> None:
    q = "SELECT * FROM fills ORDER BY id"
    alone = run(generate(), agents="gbm_twap", **MECH)
    together = run(generate(), **MECH)
    assert [tuple(r) for r in together.store.query(q)] == [tuple(r) for r in alone.store.query(q)]
    assert together.store.query("SELECT * FROM signals WHERE agent != 'gbm_twap'") == []


def test_original_strategy_trades_exactly_as_before() -> None:
    """Values the version before agents existed produced on this stream, for the original strategy alone and
    with the (untrained) learning agents listed too."""
    for agents in ("gbm_twap", "gbm_twap,rl_wide,rl_core"):
        e = run(generate(), agents=agents, **MECH)
        (sig,) = e.store.query("SELECT * FROM signals WHERE agent='gbm_twap'")
        assert sig["id"] == "btc-updown-5m-1793304300:1793304315003:Up:taker"
        assert (sig["ts_ms"], sig["arrival_ms"], sig["best_ask"], sig["limit_price"], sig["budget"], sig["status"]) == (
            1793304315003, 1793304315153, 0.55, 0.56, 20.0, "filled")
        assert [sig["p_win"], sig["edge"], sig["tau_s"], sig["latency_ms"]] == pytest.approx(
            [0.7046980061574147, 0.09884467438072264, 284.99699997901917, 150.0], rel=1e-12)
        (f,) = e.store.query("SELECT * FROM fills")
        assert (f["id"], f["ts_ms"], f["levels"]) == (sig["id"], 1793304315253, "[[0.55, 21.5, 0.37249], [0.56, 14.59, 0.25165]]")
        assert [f["gross_shares"], f["net_shares"], f["cost"], f["fee"], f["avg_price"], f["p_win_arrival"]] == pytest.approx(
            [36.09, 34.963370454545455, 19.995400000000004, 0.6241399999999999, 0.5540426711000278, 0.7093524183733964],
            rel=1e-12)
        (st,) = e.store.query("SELECT * FROM settlements")
        assert (st["won"], st["settled_by"]) == (1, "gamma") and st["pnl"] == pytest.approx(14.967970454545451, rel=1e-12)


def test_orders_share_the_displayed_depth() -> None:
    """Our paper fills never reach the exchange, so the next order (any agent) must not take the same
    displayed shares again."""
    recs = generate()
    e = run([r for r in recs if r[0] < (START + 10) * 1000], agents="gbm_twap", **MECH)  # before any trade
    m = e.markets[f"btc-updown-5m-{START}"].info
    displayed = e.books[m.token_up].asks_ascending()
    assert displayed[0] == (0.55, pytest.approx(displayed[0][1])) and e._asks_for(m.token_up) == displayed

    def order(sid: str, agent: str) -> PendingOrder:
        return PendingOrder(sid, m.slug, "Up", m.token_up, "taker", e.now_ms, 0.55, 1000.0, 0.9, agent)

    e._execute(order("a", "gbm_twap"))
    (first,) = e.store.query("SELECT * FROM fills WHERE id='a'")
    size = displayed[0][1]
    assert first["gross_shares"] == pytest.approx(math.floor(size * 0.5 * 100) / 100)
    assert e._asks_for(m.token_up)[0] == (0.55, pytest.approx(size - first["gross_shares"]))
    e._execute(order("b", "rl_core"))
    (second,) = e.store.query("SELECT * FROM fills WHERE id='b'")
    assert second["gross_shares"] == pytest.approx(math.floor((size - first["gross_shares"]) * 0.5 * 100) / 100)
    assert first["gross_shares"] + second["gross_shares"] < size * 0.76  # never 2 x 50% of the same level


# ---------------------------------------------------------------- deposits / top-up


def test_deposit_parser_validates() -> None:
    st = ParseStats()
    (ev,) = parse_raw("deposit", 5, json.dumps({"id": "x", "amount": 250}), st)
    assert (ev.deposit_id, ev.amount, ev.recv_ms) == ("x", 250.0, 5)
    for bad in ({"id": "x", "amount": 0}, {"id": "x", "amount": -5}, {"id": "x", "amount": True},
                {"id": "x", "amount": "100"}, {"id": "x", "amount": MAX_DEPOSIT_USDC * 2}, {"amount": 5},
                {"id": "", "amount": 5}):
        assert parse_raw("deposit", 5, json.dumps(bad), st) == []
    assert parse_raw("deposit", 5, '{"id": "x", "amount": NaN}', st) == []
    assert parse_raw("deposit", 5, '{"id": "x", "amount": Infinity}', st) == []
    assert parse_raw("deposit", 5, '{"id": "x", "amount": 1' + "0" * 400 + '}', st) == []  # too big for a float
    assert st.errors == 10


def test_deposit_adds_money_not_profit() -> None:
    recs = generate()
    base = run(recs, **MECH)
    later = run(recs + [deposit(recs[-1][0] + 1, 500.0), deposit(recs[-1][0] + 2, 500.0)], **MECH)  # same id twice
    assert later.counters["deposits"] == 1
    assert later.store.cash(1000) == pytest.approx(base.store.cash(1000) + 500)
    assert later.equity() == pytest.approx(base.equity() + 500)
    a = build_report(base.store, 1000, recs[-1][0] + 10_000)
    b = build_report(later.store, 1000, recs[-1][0] + 10_000)
    assert b["bankroll"]["deposits"] == 500 and b["bankroll"]["equity"] == pytest.approx(a["bankroll"]["equity"] + 500)
    for k in ("settled_after_fees", "pct_per_day_compounded_wallclock", "max_drawdown"):
        assert b["pnl"][k] == pytest.approx(a["pnl"][k])
    assert b["equity_curve"][-1][1] == pytest.approx(a["equity_curve"][-1][1] + 500)


def test_returns_are_time_weighted_across_top_ups() -> None:
    st = Storage(":memory:")
    st.set_meta("first_start_ms", "0")

    def trade(i: int, ts: int, cost: float, payout: float) -> None:
        st.insert_fill(dict(id=f"f{i}", slug="s", side="Up", ts_ms=ts - 1, kind="taker", gross_shares=1, net_shares=1,
                            cost=cost, fee=0.0, avg_price=0.5, p_win_decision=0.6, p_win_arrival=None,
                            edge_arrival=None, levels="[]", agent="rl_core"))
        st.upsert_settlement(dict(id=f"f{i}", slug="s", ts_ms=ts, won=int(payout > 0), payout=payout,
                                  pnl=payout - cost, settled_by="gamma"))

    trade(1, 1000, 100.0, 0.0)  # 1000 -> 900: -10 %
    st.insert_deposit("d", 2000, 9100.0)  # 900 -> 10000
    trade(2, 3000, 100.0, 1100.0)  # 10000 -> 11000: +10 %
    r = build_report(st, 1000, 86_400_000)
    assert r["bankroll"]["equity"] == pytest.approx(11000) and r["pnl"]["settled_after_fees"] == pytest.approx(900)
    assert r["pnl"]["pct_per_day_compounded_wallclock"] == pytest.approx(0.9 * 1.1 - 1)  # not +1000 %
    assert r["pnl"]["max_drawdown"] == pytest.approx(0.1)
    assert [round(x[1], 4) for x in r["equity_curve"]] == [1000, 900, 10000, 11000]


def test_reset_clears_top_ups_but_keeps_learning(trained) -> None:
    st = Storage(":memory:")
    e = Engine(load_settings(env={}), st)
    t, src, raw = deposit(1, 50.0)
    e.on_raw(src, t, raw)
    st.set_rl_state("rl_core", trained.rl_by_name["rl_core"].to_json(), 1)
    assert st.cash(1000) == 1050
    st.reset_trading(2)
    e.reset_trading()
    assert st.cash(1000) == 1000 and st.query("SELECT * FROM deposits") == []
    assert st.get_rl_state("rl_core") == trained.rl_by_name["rl_core"].to_json()


async def test_topup_endpoint() -> None:
    calls: list[float] = []

    def topup(amount: float) -> dict:
        calls.append(amount)
        return {"ok": True, "amount": amount}

    app = make_app(lambda: {"ok": True}, dict, dict, lambda n: [], admin_token="s3cret", topup=topup)
    hdr = {"X-Admin-Token": "s3cret"}
    async with TestClient(TestServer(app)) as c:
        assert (await c.post("/api/topup", data="amount=5")).status == 415  # plain form post (CSRF)
        assert (await c.post("/api/topup", json={"amount": 5})).status == 401
        for bad in ({}, {"amount": 0}, {"amount": -1}, {"amount": "5"}, {"amount": True}, {"amount": None},
                    {"amount": MAX_DEPOSIT_USDC + 1}, [5]):
            assert (await c.post("/api/topup", json=bad, headers=hdr)).status == 400, bad
        for raw in ('{"amount": NaN}', '{"amount": Infinity}', '{"amount": 1' + "0" * 400 + "}"):
            r = await c.post("/api/topup", data=raw, headers={**hdr, "Content-Type": "application/json"})
            assert r.status == 400, raw
        assert (await c.get("/api/topup")).status == 405
        assert not calls
        r = await c.post("/api/topup", json={"amount": 250}, headers=hdr)
        assert r.status == 200 and (await r.json())["ok"] and calls == [250.0]
    async with TestClient(TestServer(make_app(lambda: {"ok": True}, dict, dict, lambda n: []))) as c:
        assert (await c.post("/api/topup", json={"amount": 5})).status == 404


async def test_live_app_topup_is_recorded_and_replayed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from polybot import app as app_mod

    recs = generate()
    half = len(recs) // 2
    monkeypatch.setattr(app_mod, "now_ms", lambda: recs[half][0])  # the button is pressed mid-stream
    app = app_mod.LiveApp(load_settings(env={}, data_dir=str(tmp_path), **MECH))
    for t, s, r in recs[:half]:
        app.pipeline.process(s, t, r)
    out = app.topup(750.0)
    assert out["ok"] and out["deposits_total"] == 750 and out["equity"] == pytest.approx(app.engine.equity())
    for t, s, r in recs[half:]:
        app.pipeline.process(s, t, r)
    rep = app.report()
    assert rep["bankroll"]["deposits"] == 750
    app.recorder.close()
    recorded = [x for f in recording_files(tmp_path / "recordings") for x in read_recording(f)]
    assert sum(1 for _, s, _ in recorded if s == "deposit") == 1
    eng = replay(load_settings(env={}, **MECH), recorded)
    q = "SELECT * FROM deposits"
    assert [tuple(r) for r in eng.store.query(q)] == [tuple(r) for r in app.store.query(q)]
    assert eng.store.cash(1000) == pytest.approx(app.store.cash(1000))
    app.store.close()


# ---------------------------------------------------------------- migration


OLD_SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE signals (
  id TEXT PRIMARY KEY, slug TEXT, ts_ms INTEGER, side TEXT, kind TEXT,
  p_win REAL, best_ask REAL, limit_price REAL, edge REAL, tau_s REAL,
  latency_ms REAL, arrival_ms INTEGER, budget REAL, status TEXT, detail TEXT
);
CREATE TABLE fills (
  id TEXT PRIMARY KEY, slug TEXT, side TEXT, ts_ms INTEGER, kind TEXT,
  gross_shares REAL, net_shares REAL, cost REAL, fee REAL, avg_price REAL,
  p_win_decision REAL, p_win_arrival REAL, edge_arrival REAL, levels TEXT
);
INSERT INTO signals VALUES ('sig1', 'm', 1, 'Up', 'taker', 0.6, 0.5, 0.51, 0.04, 100, 150, 151, 20, 'filled', '{}');
INSERT INTO fills VALUES ('sig1', 'm', 'Up', 2, 'taker', 40, 39, 20, 0.7, 0.5, 0.6, 0.6, 0.05, '[]');
"""


def test_old_database_is_migrated(tmp_path: Path) -> None:
    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    con.executescript(OLD_SCHEMA)
    con.commit()
    con.close()
    st = Storage(db)
    assert st.query("SELECT agent FROM signals")[0][0] == "gbm_twap"
    assert st.query("SELECT agent FROM fills")[0][0] == "gbm_twap"
    assert st.entries_for("m", "gbm_twap") == 1 and st.entries_for("m", "rl_core") == 0
    assert st.cash(1000) == 980
    st.close()
    st = Storage(db)  # idempotent
    fresh = Storage(tmp_path / "new.sqlite")
    for table in ("signals", "fills"):
        cols = [r[1] for r in st.query(f"PRAGMA table_info({table})")]
        assert cols == [r[1] for r in fresh.query(f"PRAGMA table_info({table})")] and cols[-1] == "agent"
    engine = Engine(load_settings(env={}), st)  # an existing open position is still tracked after the upgrade
    assert engine.open_slugs == {"m"}
