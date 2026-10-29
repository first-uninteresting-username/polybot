"""Pluggable probability models for P(Up).

Resolution (Gamma rules text + ``cryptoMarketConfig.twapLookbackSeconds = 60``, checked 2026-10-29):
"Up" if the Chainlink BTC/USD 60s-TWAP at the window end is >= the value at the window start
(``eventMetadata.priceToBeat`` / ``finalPrice`` on closed events). Ties go to Up (``>=``).

Default model ``gbm_twap``: log-price is a driftless Brownian motion with per-sqrt-second vol
sigma (realized, floored, inflated). The final TWAP is approximated by the average of Chainlink spot over
the last W=60 s, so with tau seconds left:

* tau >= W:  log F - x_now ~ N(0, s^2 * ((tau - W) + W/3))
* tau <  W:  log F = k * A_known + (1 - k) * avg_rest, where A_known is the average of the Chainlink spot
  prints we actually observed inside the TWAP window and k = observed seconds / W;
  Var = (1-k)^2 * s^2 * max(tau, (1-k) * W) / 3 (unobserved past seconds are treated like future ones)

Extra variance terms (all pessimistic):
* basis error: our estimate of Chainlink spot from exchange prices has error sd ``basis_sd``
  (measured residual, floored, inflated), weighted by the not-yet-observed share of the TWAP.
* TWAP approximation error ``twap_err_sd``: the published TWAP is not exactly the mean of the
  1 s spot prints we see (measured ~3-4e-5 in log terms on 2026-10-29).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ModelInput:
    tau_s: float  # seconds until window end
    twap_window_s: float  # W (60)
    log_strike: float  # log(priceToBeat)
    log_spot_est: float  # log of current Chainlink-spot estimate
    known_avg_log: float | None  # average log Chainlink spot already inside the final TWAP window
    known_weight: float  # fraction of the TWAP window already observed (0..1)
    sigma: float  # per-sqrt-second vol of log price (already floored and inflated)
    basis_sd: float  # sd of log basis error (already floored and inflated)
    twap_err_sd: float


@dataclass(frozen=True)
class Prediction:
    p_up: float
    mean: float
    sd: float


class Model(Protocol):
    name: str

    def predict(self, inp: ModelInput) -> Prediction: ...


def norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class GbmTwapModel:
    name = "gbm_twap"

    def predict(self, inp: ModelInput) -> Prediction:
        w = inp.twap_window_s
        tau = max(inp.tau_s, 0.0)
        s2 = inp.sigma**2
        if tau >= w or inp.known_avg_log is None:
            # The whole TWAP window is still ahead.
            mean = inp.log_spot_est
            future_w = 1.0
            var = s2 * (max(tau - w, 0.0) + min(tau, w) / 3.0)
        else:
            kw = min(max(inp.known_weight, 0.0), 1.0)
            future_w = 1.0 - kw
            mean = kw * inp.known_avg_log + future_w * inp.log_spot_est
            # Unobserved seconds (feed lag) are treated like future ones: horizon = max(tau, unknown share * W).
            var = future_w**2 * s2 * max(tau, future_w * w) / 3.0
        var += (future_w * inp.basis_sd) ** 2 + inp.twap_err_sd**2
        sd = math.sqrt(max(var, 1e-18))
        z = (mean - inp.log_strike) / sd
        return Prediction(norm_cdf(z), mean, sd)


MODELS: dict[str, Callable[[], Model]] = {"gbm_twap": GbmTwapModel}


def get_model(name: str) -> Model:
    try:
        return MODELS[name]()
    except KeyError:
        raise ValueError(f"unknown model {name!r}; available: {sorted(MODELS)}") from None
