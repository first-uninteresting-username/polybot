import math

import pytest

from polybot.fills import (
    RestingBuy,
    all_in_cost_per_share,
    floor_to_tick,
    simulate_taker_buy,
    taker_fee_usdc,
)

# docs.polymarket.com/trading/fees, crypto table (100 shares), rounded to cents in the docs
DOC_TABLE = [(0.01, 0.07), (0.05, 0.33), (0.10, 0.63), (0.25, 1.31), (0.50, 1.75), (0.75, 1.31), (0.90, 0.63),
             (0.99, 0.07)]


@pytest.mark.parametrize("price,fee", DOC_TABLE)
def test_fee_matches_documented_table(price: float, fee: float) -> None:
    assert round(taker_fee_usdc(100, price, 0.07), 2) == pytest.approx(fee, abs=0.006)


def test_fee_symmetric_and_rounded_up_to_5dp() -> None:
    assert taker_fee_usdc(100, 0.3, 0.07) == taker_fee_usdc(100, 0.7, 0.07)
    f = taker_fee_usdc(1.23, 0.37, 0.07)
    raw = 1.23 * 0.07 * 0.37 * 0.63
    assert f >= raw and f - raw < 1e-5
    assert round(f, 5) == f


def test_fee_zero_when_disabled() -> None:
    assert taker_fee_usdc(100, 0.5, 0.0) == 0.0


def test_all_in_cost_fee_in_shares() -> None:
    # buying 1 gross share at 0.5 costs 0.5 and the fee 0.07*0.25=0.0175 USDC is taken as 0.035 shares
    c = all_in_cost_per_share(0.5, 0.07)
    assert c == pytest.approx(0.5 / (1 - 0.035))
    assert all_in_cost_per_share(0.5, 0.0) == 0.5


def test_walk_book_levels_limit_and_fees() -> None:
    asks = [(0.50, 10), (0.51, 10), (0.53, 100)]
    r = simulate_taker_buy(asks, limit_price=0.51, max_usdc=100, min_size=5, fee_rate=0.07)
    assert [lv.price for lv in r.levels] == [0.50, 0.51]
    assert r.gross_shares == 20
    assert r.reason == "partial"  # liquidity at/below limit exhausted before budget
    assert r.fee_usdc == pytest.approx(taker_fee_usdc(10, 0.5, 0.07) + taker_fee_usdc(10, 0.51, 0.07))
    assert r.net_shares == pytest.approx(20 - taker_fee_usdc(10, 0.5, 0.07) / 0.5 - taker_fee_usdc(10, 0.51, 0.07) / 0.51)
    assert r.cost_usdc == pytest.approx(10 * 0.5 + 10 * 0.51)


def test_budget_limits_fill() -> None:
    r = simulate_taker_buy([(0.40, 1000)], 0.40, 10.0, 5, 0.07)
    assert r.gross_shares == 25 and r.reason == "filled"
    assert r.cost_usdc == pytest.approx(10.0)


def test_take_fraction_haircut() -> None:
    r = simulate_taker_buy([(0.40, 20), (0.41, 20)], 0.41, 100, 5, 0.07, take_fraction=0.5)
    assert r.gross_shares == 20


def test_min_size_and_no_liquidity() -> None:
    assert simulate_taker_buy([(0.5, 100)], 0.5, 2.0, 5, 0.07).reason == "below_min_size"
    assert simulate_taker_buy([(0.6, 100)], 0.5, 50, 5, 0.07).reason == "no_liquidity_within_limit"
    assert simulate_taker_buy([], 0.5, 50, 5, 0.07).levels == []


def test_tick_rounding() -> None:
    assert floor_to_tick(0.567, 0.01) == pytest.approx(0.56)
    assert floor_to_tick(0.57, 0.01) == pytest.approx(0.57)


def test_maker_fills_only_when_traded_through() -> None:
    rb = RestingBuy("o", "a", price=0.40, shares=10, placed_ms=0, expires_ms=10**9)
    assert rb.on_trade(0.40, 100) == 0  # touching our price is not enough (back of queue)
    assert rb.on_trade(0.41, 100) == 0
    assert rb.on_trade(0.39, 4) == 4
    assert rb.on_trade(0.38, 100) == 6
    assert rb.on_trade(0.30, 100) == 0
    assert math.isclose(rb.filled, 10)
