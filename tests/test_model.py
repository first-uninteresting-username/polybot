import math

import pytest

from polybot.model import GbmTwapModel, ModelInput, get_model

M = GbmTwapModel()


def inp(**kw: float) -> ModelInput:
    base = dict(tau_s=200.0, twap_window_s=60.0, log_strike=math.log(80000), log_spot_est=math.log(80000),
                known_avg_log=None, known_weight=0.0, sigma=0.0001, basis_sd=0.0001, twap_err_sd=5e-5)
    base.update(kw)
    return ModelInput(**base)  # type: ignore[arg-type]


def test_at_the_money_is_half() -> None:
    assert M.predict(inp()).p_up == pytest.approx(0.5)


def test_monotonic_in_spot() -> None:
    ps = [M.predict(inp(log_spot_est=math.log(80000 + d))).p_up for d in (-50, -10, 0, 10, 50)]
    assert ps == sorted(ps) and ps[0] < 0.5 < ps[-1]


def test_more_time_means_less_certain() -> None:
    above = math.log(80040)
    assert M.predict(inp(log_spot_est=above, tau_s=30, known_avg_log=above, known_weight=0.5)).p_up > \
        M.predict(inp(log_spot_est=above, tau_s=250)).p_up > 0.5


def test_variance_formula_full_window_ahead() -> None:
    p = M.predict(inp(tau_s=200, basis_sd=0, twap_err_sd=0))
    assert p.sd == pytest.approx(0.0001 * math.sqrt(140 + 60 / 3))


def test_inside_twap_window_uses_known_average() -> None:
    # 50 of 60 s already observed well above the strike, spot now slightly below: still likely Up
    p = M.predict(inp(tau_s=10, known_avg_log=math.log(80100), known_weight=50 / 60, log_spot_est=math.log(79990)))
    assert p.p_up > 0.95
    w = 10 / 60
    exp_var = (w**2) * 0.0001**2 * 10 / 3 + (w * 0.0001) ** 2 + 5e-5**2
    assert p.sd == pytest.approx(math.sqrt(exp_var))


def test_registry() -> None:
    assert get_model("gbm_twap").name == "gbm_twap"
    with pytest.raises(ValueError):
        get_model("nope")
