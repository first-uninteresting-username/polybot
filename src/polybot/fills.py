"""Fee math and the fill simulator.

Fees (docs.polymarket.com/trading/fees, checked 2026-10-29): ``fee = C * feeRate * p * (1 - p)`` in USDC,
taker only, rounded to 5 decimals. Per-market parameters come from Gamma ``feeSchedule``
(e.g. ``{"exponent": 1, "rate": 0.07, "takerOnly": true}``). We apply ``(p*(1-p))**exponent``; with the
observed exponent of 1 this is exactly the documented formula.

On buys the fee is collected in shares: net shares = C - fee_usdc / p. Given that we only buy when we
believe P(win) > p, charging in shares costs at least as much in expectation as charging USDC, so it is
also the pessimistic choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

FEE_DECIMALS = 5
SIZE_DECIMALS = 2  # share quantities are expressed to 0.01


def taker_fee_usdc(shares: float, price: float, rate: float, exponent: float = 1.0, round_up: bool = True) -> float:
    if shares <= 0 or rate <= 0:
        return 0.0
    raw = shares * rate * (price * (1.0 - price)) ** exponent
    scale = 10**FEE_DECIMALS
    # Exchange rounds to 5 dp; rounding up is the pessimistic choice (differs by < 0.00001 USDC).
    return (math.ceil(raw * scale - 1e-9) if round_up else round(raw * scale)) / scale


def floor_to_tick(price: float, tick: float) -> float:
    return math.floor(price / tick + 1e-9) * tick


def floor_size(x: float) -> float:
    s = 10**SIZE_DECIMALS
    return math.floor(x * s + 1e-9) / s


@dataclass
class LevelFill:
    price: float
    shares: float  # gross shares matched at this level
    fee_usdc: float


@dataclass
class FillResult:
    levels: list[LevelFill] = field(default_factory=list)
    reason: str = ""  # why it stopped / did not fill

    @property
    def gross_shares(self) -> float:
        return sum(lv.shares for lv in self.levels)

    @property
    def cost_usdc(self) -> float:
        return sum(lv.shares * lv.price for lv in self.levels)

    @property
    def fee_usdc(self) -> float:
        return sum(lv.fee_usdc for lv in self.levels)

    @property
    def net_shares(self) -> float:
        """Shares actually held after the fee is taken in shares."""
        return sum(lv.shares - (lv.fee_usdc / lv.price if lv.price > 0 else 0.0) for lv in self.levels)

    @property
    def avg_price(self) -> float:
        g = self.gross_shares
        return self.cost_usdc / g if g > 0 else 0.0

    @property
    def cost_per_net_share(self) -> float:
        n = self.net_shares
        return self.cost_usdc / n if n > 0 else math.inf


def all_in_cost_per_share(price: float, rate: float, exponent: float = 1.0) -> float:
    """USDC paid per share actually held when buying one share at ``price`` (fee taken in shares)."""
    fee_shares = rate * (price * (1.0 - price)) ** exponent / price if price > 0 else 0.0
    net = 1.0 - fee_shares
    return price / net if net > 0 else math.inf


def simulate_taker_buy(
    asks: list[tuple[float, float]],
    limit_price: float,
    max_usdc: float,
    min_size: float,
    fee_rate: float,
    fee_exponent: float = 1.0,
    take_fraction: float = 1.0,
    fee_round_up: bool = True,
) -> FillResult:
    """Fill-and-kill buy: walk asks (ascending) up to ``limit_price``, spending at most ``max_usdc``.

    ``take_fraction`` < 1 assumes competing takers consume part of every displayed level first.
    The order itself must be at least ``min_size`` shares (checked against what the budget buys at the
    limit price); partial fills of a valid order are allowed.
    """
    res = FillResult()
    if max_usdc <= 0:
        res.reason = "no_budget"
        return res
    if floor_size(max_usdc / limit_price) < min_size:
        res.reason = "below_min_size"
        return res
    budget = max_usdc
    for price, size in asks:
        if price > limit_price + 1e-9:
            break
        avail = floor_size(size * take_fraction)
        if avail <= 0:
            continue
        take = floor_size(min(avail, budget / price))
        if take <= 0:
            break
        fee = taker_fee_usdc(take, price, fee_rate, fee_exponent, fee_round_up)
        res.levels.append(LevelFill(price, take, fee))
        budget -= take * price
        if budget < price * 0.01:
            break
    if not res.levels:
        res.reason = "no_liquidity_within_limit"
    elif budget >= limit_price * 0.01:
        res.reason = "partial"  # ran out of liquidity at or below the limit before the budget
    else:
        res.reason = "filled"
    return res


@dataclass
class RestingBuy:
    """A simulated maker bid. Assumed to sit at the BACK of the queue at its price.

    It is only filled when a trade prints strictly *below* its price (the whole level, including us,
    must have been consumed for that to happen). A trade *at* our price is never counted.
    Fill size per print is min(remaining, print size). Maker fee is 0; rebates are ignored.
    """

    order_id: str
    asset_id: str
    price: float
    shares: float
    placed_ms: int
    expires_ms: int
    filled: float = 0.0

    def on_trade(self, price: float, size: float) -> float:
        if price < self.price - 1e-9 and self.filled < self.shares:
            q = floor_size(min(self.shares - self.filled, size))
            self.filled += q
            return q
        return 0.0
