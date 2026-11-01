"""Online-learning ("RL") trading agents.

Each agent faces, in every 5-minute window, the choice the original strategy faces: skip, buy Up or buy Down
through the same fill simulator (latency, book walk, depth haircut, fees). The value of buying a side is

    P(side wins | inputs) - all-in cost per share

so learning that value means learning P(Up | inputs). The agent learns it from the environment's feedback:
every window's official outcome is the reward signal for every decision the agent looked at in that window.
Because the outcome of a window is observed whether or not the agent traded, the counterfactual reward of
every action is known, so no random exploration (no deliberate losing trades) is needed (a contextual bandit
with full-information feedback).

Model: Bayesian logistic regression ``P(Up) = sigmoid(w . x)`` with a Gaussian posterior over ``w`` (full
covariance), updated once per resolved window by a Laplace step (Newton on the log posterior). All samples of
a window share one outcome, so together they count as one observation. The precision decays slowly towards
the prior, so the agent keeps adapting when markets change. The prior mean is half the ``gbm_twap`` logit plus
half the market's logit; every other weight starts at 0.

Policy: an agent trades only after it has learned from ``rl_min_windows`` windows, and then on the
pessimistic end of its posterior: ``P_lo = sigmoid(mean - k sd)`` for Up and ``1 - sigmoid(mean + k sd)`` for
Down, so an agent that knows little does not trade (and, unlike widening the distribution, this is
pessimistic for both sides). It also learns from the rewards of its own policy: for the first trade the
policy would make in each window (taken or not), it compares the outcome with the predicted P(win). If those
selected trades win less often than predicted (winner's curse, adverse selection), the shortfall is added to
the required edge.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Every input the engine computes for the learning agents, in this order (Engine.rl_features).
FEATURES: tuple[str, ...] = (
    "bias",  # 1
    "gbm_logit",  # logit of the gbm_twap model's P(Up)
    "gbm_z",  # gbm_twap z-score (mean - log strike) / sd: lets the agent reshape the model's tails
    "mkt_logit",  # logit of the market's P(Up): mean of the Up mid and 1 - the Down mid
    "tau",  # time left / window length
    "known_w",  # share of the final TWAP already observed
    "gbm_x_tau",  # time-dependent trust in the model ...
    "mkt_x_tau",  # ... and in the market
    "mkt_x_spread",  # a wide market is a less informative market
    "spread",  # mean bid-ask spread of both tokens (x10)
    "overround",  # best ask Up + best ask Down - 1 (x10)
    "book_imb",  # top-of-book size imbalance, oriented to Up
    "depth_imb",  # log depth ratio within 3 ticks of the touch, oriented to Up
    "trade_dev",  # last trade (as P(Up)) vs market mid, in logits
    "vol_level",  # log realized vol (15 min) vs 5e-5 per sqrt s
    "vol_ratio",  # log realized vol last 60 s vs 15 min (regime change)
    "mom_5",  # exchange momentum over 5 ... 120 s, in sd units
    "mom_15",
    "mom_30",
    "mom_60",
    "mom_120",
    "venue_gap",  # Binance vs Coinbase estimate of Chainlink spot, in basis-error sd
    "basis_sd",  # log basis-error sd vs 1e-4
    "cl_lag",  # age of the latest Chainlink print (/5 s)
    "cl_gap",  # move of the spot estimate since the latest Chainlink print, in sd units
)

PRESETS: dict[str, tuple[str, ...]] = {
    "rl_wide": FEATURES,
    "rl_core": ("bias", "gbm_logit", "mkt_logit", "gbm_x_tau", "mkt_x_tau"),
}

PRIOR_MEAN: dict[str, float] = {"gbm_logit": 0.5, "mkt_logit": 0.5}

STATE_VERSION = 1
NEWTON_STEPS = 2  # the prior dominates a single window's likelihood, so this is converged (see tests)
LOGIT_CAP = 30.0


def sigmoid(z: float) -> float:
    z = min(max(z, -LOGIT_CAP), LOGIT_CAP)
    return 1.0 / (1.0 + math.exp(-z))


def logit(p: float, eps: float) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return math.log(p / (1.0 - p))


def predictive(mu: float, var: float) -> float:
    """Posterior predictive P(Up) for a logit ~ N(mu, var) (probit approximation)."""
    return sigmoid(mu / math.sqrt(1.0 + math.pi * var / 8.0))


# ---------------------------------------------------------------- small dense linear algebra (d <= ~30)


def cholesky(a: Sequence[Sequence[float]]) -> list[list[float]]:
    """Lower-triangular ``L`` with ``L L^T = a`` for a symmetric positive-definite ``a``; ValueError otherwise."""
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        li = low[i]
        for j in range(i + 1):
            lj = low[j]
            s = a[i][j]
            for k in range(j):
                s -= li[k] * lj[k]
            if i == j:
                if not s > 0.0:  # also rejects NaN
                    raise ValueError("matrix is not positive definite")
                li[i] = math.sqrt(s)
            else:
                li[j] = s / lj[j]
    return low


def chol_solve(low: Sequence[Sequence[float]], b: Sequence[float]) -> list[float]:
    """Solve ``(L L^T) x = b``."""
    n = len(low)
    y = [0.0] * n
    for i in range(n):
        s = b[i]
        li = low[i]
        for k in range(i):
            s -= li[k] * y[k]
        y[i] = s / li[i]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = y[i]
        for k in range(i + 1, n):
            s -= low[k][i] * x[k]
        x[i] = s / low[i][i]
    return x


def chol_inverse(low: Sequence[Sequence[float]]) -> list[list[float]]:
    n = len(low)
    cols = [chol_solve(low, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)]
    return [[0.5 * (cols[j][i] + cols[i][j]) for j in range(n)] for i in range(n)]


# ---------------------------------------------------------------- learner


class OnlineLogit:
    """Bayesian logistic regression learned online; posterior ``N(mean, prec^-1)`` over the weights."""

    def __init__(self, names: Sequence[str], prior_sd: float, memory_windows: float):
        if not prior_sd > 0:
            raise ValueError("rl_prior_sd must be > 0")
        self.names = tuple(names)
        d = len(self.names)
        self.q0 = 1.0 / prior_sd**2
        self.forget = 1.0 / memory_windows if memory_windows > 0 else 0.0
        self.prior_mean = [PRIOR_MEAN.get(n, 0.0) for n in self.names]
        self.mean = list(self.prior_mean)
        self.prec = [[self.q0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        self.cov = [[1.0 / self.q0 if i == j else 0.0 for j in range(d)] for i in range(d)]
        self.windows = 0  # resolved windows learned from

    def predict(self, x: Sequence[float]) -> tuple[float, float]:
        """Posterior mean and variance of the logit ``w . x``."""
        mu = 0.0
        for m, xi in zip(self.mean, x, strict=True):
            mu += m * xi
        var = 0.0
        for i, xi in enumerate(x):
            if xi == 0.0:
                continue
            row = self.cov[i]
            s = 0.0
            for j, xj in enumerate(x):
                s += row[j] * xj
            var += xi * s
        return mu, max(var, 0.0)

    def _curvature(self, base: list[list[float]], w: list[float], xs: Sequence[Sequence[float]], v: float,
                   y: int) -> tuple[list[list[float]], list[float]]:
        """Hessian and gradient of the negative log posterior at ``w`` (prior N(mean, base^-1), samples
        weighted ``v``)."""
        d = len(w)
        h = [row[:] for row in base]
        dw = [wi - mi for wi, mi in zip(w, self.mean, strict=True)]
        g = [0.0] * d
        for i in range(d):
            row = base[i]
            s = 0.0
            for j in range(d):
                s += row[j] * dw[j]
            g[i] = s
        for x in xs:
            z = 0.0
            for wi, xi in zip(w, x, strict=True):
                z += wi * xi
            p = sigmoid(z)
            r = v * (p - y)
            for i in range(d):
                g[i] += r * x[i]
            s = v * p * (1.0 - p)
            if s <= 0.0:
                continue
            for i in range(d):
                sxi = s * x[i]
                if sxi == 0.0:
                    continue
                hi = h[i]
                for j in range(i + 1):
                    hi[j] += sxi * x[j]
        for i in range(d):  # mirror the updated lower triangle
            for j in range(i):
                h[j][i] = h[i][j]
        return h, g

    def update(self, xs: Sequence[Sequence[float]], y: int) -> bool:
        """Learn from one resolved window: ``xs`` are its samples, ``y`` its outcome (1 = Up). The samples
        share one outcome, so they are weighted 1/n and count as one observation together. Returns False and
        changes nothing if there is nothing to learn or the update is not numerically sound."""
        n = len(xs)
        d = len(self.names)
        if n == 0 or y not in (0, 1) or any(len(x) != d for x in xs):
            return False
        v = 1.0 / n
        lam = self.forget
        base = [[(1.0 - lam) * self.prec[i][j] + (lam * self.q0 if i == j else 0.0) for j in range(d)]
                for i in range(d)]
        w = list(self.mean)
        try:
            for _ in range(NEWTON_STEPS):
                h, g = self._curvature(base, w, xs, v, y)
                step = chol_solve(cholesky(h), g)
                w = [wi - si for wi, si in zip(w, step, strict=True)]
            h, _ = self._curvature(base, w, xs, v, y)
            cov = chol_inverse(cholesky(h))
        except (ValueError, ZeroDivisionError, OverflowError):
            return False
        if not all(math.isfinite(t) for t in w) or not all(math.isfinite(t) for row in cov for t in row):
            return False
        self.mean, self.prec, self.cov = w, h, cov
        self.windows += 1
        return True

    def to_state(self) -> dict[str, Any]:
        return {"names": list(self.names), "mean": self.mean, "prec": self.prec, "windows": self.windows}

    def load_state(self, st: dict[str, Any]) -> bool:
        """Adopt a saved posterior if it was learned on exactly these inputs; False (unchanged) otherwise."""
        try:
            if list(st["names"]) != list(self.names):
                return False
            d = len(self.names)
            mean = [float(t) for t in st["mean"]]
            prec = [[float(t) for t in row] for row in st["prec"]]
            if len(mean) != d or len(prec) != d or any(len(r) != d for r in prec):
                return False
            cov = chol_inverse(cholesky(prec))
            windows = int(st["windows"])
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return False
        if not all(math.isfinite(t) for t in mean) or not all(math.isfinite(t) for row in cov for t in row):
            return False
        self.mean, self.prec, self.cov, self.windows = mean, prec, cov, windows
        return True


# ---------------------------------------------------------------- agent


@dataclass
class RLAgent:
    name: str
    learner: OnlineLogit
    index: tuple[int, ...]  # positions of this agent's inputs in FEATURES
    bias_sum: float = 0.0  # decayed sum of (won - predicted P(win)) over the policy's would-be trades
    bias_n: float = 0.0  # decayed count of those trades
    # slug -> (side, central P(win), window end) of the first trade the policy would make in that window
    virtual: dict[str, tuple[str, float, int]] = field(default_factory=dict)

    def project(self, x_all: Sequence[float]) -> list[float]:
        return [x_all[i] for i in self.index]

    def penalty(self, prior_trades: float) -> float:
        """Extra edge the policy must clear because its selected trades have won less often than predicted
        (winner's curse, adverse selection). Never negative: good luck never loosens the rules."""
        den = self.bias_n + prior_trades
        return max(0.0, -self.bias_sum / den) if den > 0 else 0.0

    def note_virtual(self, slug: str, side: str, p_win: float, end_ts: int) -> None:
        self.virtual.setdefault(slug, (side, p_win, end_ts))

    def learn_outcome(self, slug: str, winner: str, memory_windows: float) -> bool:
        """Reward feedback for the window's would-be trade, if the policy wanted one."""
        vt = self.virtual.pop(slug, None)
        if vt is None:
            return False
        side, p_win, _ = vt
        decay = 1.0 - 1.0 / memory_windows if memory_windows > 0 else 1.0
        self.bias_sum = decay * self.bias_sum + ((1.0 if side == winner else 0.0) - p_win)
        self.bias_n = decay * self.bias_n + 1.0
        return True

    def to_json(self) -> str:
        return json.dumps({"version": STATE_VERSION, "learner": self.learner.to_state(),
                           "bias_sum": self.bias_sum, "bias_n": self.bias_n})

    def load_json(self, raw: str) -> bool:
        """Adopt a saved state; False and unchanged if it is unreadable or was learned on other inputs."""
        try:
            st = json.loads(raw)
            if st.get("version") != STATE_VERSION:
                return False
            bias_sum, bias_n = float(st["bias_sum"]), float(st["bias_n"])
            learner_state = st["learner"]
        except (ValueError, TypeError, KeyError, AttributeError):
            return False
        if not (math.isfinite(bias_sum) and math.isfinite(bias_n) and bias_n >= 0):
            return False
        if not isinstance(learner_state, dict) or not self.learner.load_state(learner_state):
            return False
        self.bias_sum, self.bias_n = bias_sum, bias_n
        return True


def make_agent(name: str, prior_sd: float, memory_windows: float) -> RLAgent:
    feats = PRESETS[name]
    return RLAgent(name, OnlineLogit(feats, prior_sd, memory_windows), tuple(FEATURES.index(f) for f in feats))
