"""Configuration: defaults < optional TOML file (POLYBOT_CONFIG) < environment variables.

Every field can be set with an env var named ``POLYBOT_<FIELD_NAME_UPPER>``.
"""

from __future__ import annotations

import dataclasses
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Settings:
    # --- storage / service ---
    data_dir: str = "./data"
    http_host: str = "0.0.0.0"
    http_port: int = 8080
    log_level: str = "INFO"
    record_retention_days: int = 14  # 0 = keep forever (~3 GB/day compressed)

    # --- endpoints (all public, unauthenticated) ---
    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    rtds_ws_url: str = "wss://ws-live-data.polymarket.com"
    binance_ws_url: str = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    coinbase_ws_url: str = "wss://ws-feed.exchange.coinbase.com"

    # --- market ---
    slug_prefix: str = "btc-updown-5m-"
    window_seconds: int = 300
    discovery_interval_s: float = 20.0
    discovery_lookahead_windows: int = 2

    # --- staleness: skip trading if any required feed is older than this (event time) ---
    stale_exchange_ms: int = 2000
    stale_book_ms: int = 5000  # Polymarket books only change when someone acts
    stale_oracle_ms: int = 12000  # by Chainlink timestamp; RTDS may be in 5 s snapshot-polling mode
    rtds_resnapshot_s: float = 5.0

    # --- model ---
    model: str = "gbm_twap"
    vol_lookback_s: int = 900
    vol_floor_per_sqrt_s: float = 0.00003  # ~ 0.17%/sqrt(hour); floor on realized vol
    vol_inflation: float = 1.0  # centre estimate; uncertainty is handled by model_sd_band, not by bias
    basis_halflife_s: float = 120.0
    basis_err_floor: float = 0.00005  # min stdev (log) of exchange->Chainlink basis error
    basis_err_inflation: float = 1.5
    min_vol_samples: int = 120
    twap_err_sd: float = 5e-5  # log-sd: published TWAP vs mean of 1 s spot prints (measured ~3-4e-5)

    # --- strategy ---
    eval_interval_ms: int = 500
    min_edge: float = 0.03  # required (p_win - all-in cost per share)
    min_tau_s: float = 15.0
    max_tau_s: float = 285.0
    max_entries_per_market: int = 1
    stake_fraction: float = 0.02
    max_stake_usd: float = 50.0
    starting_bankroll: float = 1000.0
    p_clip: float = 0.98  # never believe the model beyond this
    # Inflating vol is not pessimistic for a two-sided strategy: it pulls P toward 0.5 and invents edge on
    # the underdog. Instead, trade on the worst P(win) with sigma and basis sd scaled by 1/band .. band.
    model_sd_band: float = 1.25  # 1 = off
    max_model_market_gap: float = 0.20  # skip if |P(win) - token mid| is larger (likely model/data error); 0 = off
    kelly_fraction: float = 0.25  # stake = min(stake_fraction, kelly_fraction * Kelly f*) x equity; 0 = flat

    # --- fill model ---
    latency_ms_floor: float = 150.0  # min simulated decision->match latency
    latency_extra_ms: float = 50.0  # added to every sampled RTT (processing / signing time)
    latency_jitter_sigma: float = 0.35  # lognormal sigma around the measured RTT median
    book_take_fraction: float = 0.5  # assume other takers race us for displayed size
    limit_slippage_ticks: int = 1  # limit = decision ask + N ticks (bounded by min_edge)
    fee_round_up: bool = True
    enable_maker: bool = False
    maker_ttl_s: float = 30.0

    # --- settlement ---
    resolution_poll_s: float = 30.0
    fallback_after_s: float = 7200.0  # after this long past window end, settle on our own TWAP

    # --- misc ---
    seed: int = 1337
    latency_probe_interval_s: float = 15.0
    admin_token: str = ""  # if set, POST /api/reset needs header X-Admin-Token with this value

    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


def _coerce(value: str, typ: Any) -> Any:
    if typ in (bool, "bool"):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if typ in (int, "int"):
        return int(value)
    if typ in (float, "float"):
        return float(value)
    return value


def load_settings(env: dict[str, str] | None = None, **overrides: Any) -> Settings:
    env = dict(os.environ) if env is None else env
    values: dict[str, Any] = {}
    cfg_file = env.get("POLYBOT_CONFIG")
    if cfg_file:
        with open(cfg_file, "rb") as fh:
            values.update(tomllib.load(fh))
    for f in dataclasses.fields(Settings):
        if f.name == "extra":
            continue
        key = "POLYBOT_" + f.name.upper()
        if key in env:
            values[f.name] = _coerce(env[key], f.type)
    values.update(overrides)
    known = {f.name for f in dataclasses.fields(Settings)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")
    return Settings(**values)
