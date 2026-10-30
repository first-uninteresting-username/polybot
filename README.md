# polybot: paper trader for Polymarket 5-minute BTC Up/Down markets

**Paper trading only.** This service never places, signs or cancels an order. It has no wallet, no
private keys, no API credentials and calls no authenticated endpoint. Every "trade" is a simulation
against the live public order book.

It measures, as honestly as it can, whether a simple model-driven strategy on Polymarket's
`btc-updown-5m-<start>` markets would make money after fees, latency and slippage. Read the results
with the sections [How fills are simulated](#how-fills-are-simulated) and
[Known optimistic biases](#known-optimistic-biases) in mind.

## What it does

```
Binance trade ws ─┐
Coinbase ticker ws ┤                       ┌─> recorder (zstd JSONL, daily files)
Polymarket RTDS ───┼─> asyncio queue ──────┤
CLOB market ws ────┤   (recv_ms, src, raw) └─> Engine (deterministic, event time)
Gamma discovery ───┤                             ├─ model: P(Up)
Gamma resolution ──┤                             ├─ strategy: edge after fees
CLOB RTT probe ────┘                             ├─ fill simulator (after latency, book walk)
                                                 ├─ settlement on Polymarket's outcome
                                                 └─ SQLite (signals, fills, settlements, predictions)
web: /  /health  /report  /api/state  /api/signals  /logs  POST /api/reset
```

* Every input, including market metadata, resolutions and latency samples, goes through one queue,
  is recorded raw with its local receive timestamp, and is then handed to the engine.
* The engine uses only event time (receive timestamps), never the wall clock. Replay feeds the
  recording through the same engine, so live and replay make identical decisions on identical data.
  `tests/test_equivalence.py` proves it on a synthetic window and on a real live capture.

## Market mechanics (verified 2026-10-29)

Each claim below was checked against live responses, which are saved in `tests/fixtures/`, and against
the docs.

| Item | Finding |
| --- | --- |
| Slug | `btc-updown-5m-<unix start>`, start aligned to 300 s. `GET gamma-api.polymarket.com/markets?slug=…` |
| Tokens | `clobTokenIds` (JSON string) zipped with `outcomes` (`["Up","Down"]`) |
| **Resolution** | **Chainlink BTC/USD 60-second TWAP** stream (`resolutionSource: data.chain.link/streams/btc-usd-twap-60s-streams`, `cryptoMarketConfig.twapLookbackSeconds: 60`). "Up" if TWAP(end) **>=** TWAP(start), so ties go to Up. This is *not* the spot price. |
| Strike | `eventMetadata.priceToBeat` equals, to the last digit, the RTDS `crypto_prices_twap_sixty` value timestamped at the window start, and `finalPrice` equals the value at the window end. Gamma only publishes both after the window closes, so the bot takes the strike from RTDS at the start. |
| Outcome | Closed event: `closed: true`, `umaResolutionStatus: "resolved"`, `outcomePrices` `["1","0"]` or `["0","1"]`. About 80 s after the window ends. |
| Fees | `feesEnabled: true`, `feeType: crypto_fees_v2`, `feeSchedule: {rate: 0.07, exponent: 1, takerOnly: true}`. `fee = C × 0.07 × p × (1−p)` USDC, taker only, 5 dp ([docs](https://docs.polymarket.com/trading/fees)). Read per market, never hardcoded. CLOB `GET /fee-rate` returns `base_fee: 1000`, which is not used. |
| Tick / min size | `orderPriceMinTickSize: 0.01`, `orderMinSize: 5` (shares). Also `GET /tick-size`. `tick_size_change` events are honoured. |
| CLOB ws | `wss://ws-subscriptions-clob.polymarket.com/ws/market`, `{"assets_ids":[…],"type":"market","custom_feature_enabled":true}`. Events: `book`, `price_change` (new aggregate size per level), `last_trade_price`, `best_bid_ask`, `new_market`. Text `PING` every 10 s. |
| RTDS | `wss://ws-live-data.polymarket.com`, topics `crypto_prices_chainlink` and `crypto_prices_twap_sixty` with filter `{"symbol":"btc/usd"}`. Values are E18 fixed-point in `full_accuracy_value`. The Chainlink snapshot frame arrives under topic `crypto_prices`. |

### RTDS status (important)

RTDS price topics are deprecated in favour of **PolyBolt** (`wss://ws-live-v2.polymarket.com/ws`).
PolyBolt requires CLOB API credentials, which come from a wallet signature, so this project must not
use it. Since about 2026-10-29 20:21 UTC, new RTDS connections receive the 60 s history snapshot but
**no live updates**. The RTDS feed detects this and switches to *snapshot polling*: it reconnects every
`POLYBOT_RTDS_RESNAPSHOT_S` (3 s) and uses the fresh snapshot. Chainlink data then lags by about
1–4 s. `/api/state` shows the feed mode as `streaming` or `snapshot_polling`. If RTDS goes away
entirely, the bot cannot know the strike and **stops trading** rather than guessing.

## Model (`src/polybot/model.py`, pluggable)

`gbm_twap`: driftless Brownian log-price with realized vol, floored (and multiplied by
`vol_inflation`, 1.0 by default: the central estimate is not deliberately biased). The final TWAP is approximated as the mean of Chainlink spot over the last 60 s:

* With more than 60 s left: `log F − x_now ~ N(0, σ²((τ−60) + 60/3))`.
* Inside the final minute: the Chainlink prints already observed in the averaging window are known
  and only the rest is random. Seconds the feed has not delivered yet count as unknown.
* `x_now` is the Chainlink spot *estimated* from Binance and Coinbase, each with an EWMA basis to
  Chainlink. The basis error variance (measured residual, floored, inflated) and a TWAP
  approximation error (`twap_err_sd`; the published TWAP differs from the mean of 1 s prints by
  about $3) are added.
* Probabilities are clipped to `[1−p_clip, p_clip]`.

Calibration is tracked two ways: on traded signals, and on unselected periodic predictions (stored
every `prediction_interval_ms`, 1 s; the report samples one per 10 s for every window). The second is less biased. The report shows Brier scores and a reliability
table for both, plus predicted vs actual win rate.

## Strategy

Every `eval_interval_ms` of event time, for each live window with a known strike:

* **Filters:** `min_tau_s ≤ τ ≤ max_tau_s`, every feed fresh, at least 5 latency samples, fewer than
  `max_entries_per_market` entries in this window.
* **Worst-case P(win):** the model is also run with sigma and basis sd scaled by `1/model_sd_band`
  and `model_sd_band` (1.25). Each outcome uses the lowest of the three P(win). Inflating vol is *not*
  a pessimistic choice for a strategy that can buy either side: it pulls P toward 0.5 and invents edge
  on the underdog. The band is pessimistic for both sides. The signal records the central P(win)
  (used for calibration) and the worst-case edge.
* **Edge:** for each outcome, `edge = worst-case P(win) − all-in cost per share at the best ask`, with
  the fee taken in shares.
* **Model vs market:** an outcome is skipped (`skip_model_market_gap`) if the central P(win) is more than
  `max_model_market_gap` (0.20) from the token's mid. A liquid market that far from the model is far
  more often a model or data error (wrong strike, stale oracle) than free money.
* **Signal:** if `edge ≥ min_edge` and the book can fill at least `orderMinSize` within the limit, a
  simulated fill-and-kill buy is sent with `limit = min(ask + limit_slippage_ticks, highest price
  still meeting min_edge)`.
* **Stake:** `min(f × equity, max_stake_usd, free cash)` with
  `f = min(stake_fraction, kelly_fraction × (p − c)/(1 − c))`, the Kelly fraction for worst-case
  P(win) `p` and all-in cost `c` at the limit. Thin edges get small stakes. `kelly_fraction=0` gives a
  flat `stake_fraction`.

Optional maker mode (`enable_maker`, off by default) rests a bid at `worst-case P(win) − min_edge`.

## How fills are simulated

1. **Latency.** At signal time, a latency is drawn from the bot's own measured HTTPS round-trip times
   to `clob.polymarket.com` (kept-alive connection, every 15 s, recorded so replay uses the same
   samples). The draw is a random measured sample × lognormal jitter (σ 0.35) + 50 ms processing, with
   a floor of 150 ms. The draw is seeded from the signal id, so it is deterministic.
2. **Book at arrival.** The order executes at the first event at or after `decision + latency`,
   against the book as it was at that moment, before that event is applied. The book is rebuilt from
   the CLOB websocket `book` and `price_change` events. Each `price_change` also carries the exchange's
   best bid/ask; levels better than that are phantom liquidity from a missed update (trades remove
   levels without a `price_change`) and are dropped. If we lack the reported level itself, the book is
   flagged (`skip_book_desync`) until the next snapshot. On the saved live capture this makes the
   reconstructed top of book match the exchange on all of about 17,000 updates (it was 12 off before).
3. **Book walk.** Asks are walked from the best price up to the limit, level by level, with partial
   fills. Only `book_take_fraction` (0.5) of each displayed level is assumed available, because
   other takers race for it. Sizes are floored to 0.01 shares, the order must be at least
   `orderMinSize`, and the unfilled rest is cancelled.
4. **Fees per level.** `C × rate × p(1−p)`, rounded **up** to 5 dp, collected in shares:
   net shares = C − fee/p.
5. **Maker orders** (optional): assumed to sit at the back of the queue, filled only when a trade
   prints strictly *through* the price (never on a touch), with fill size limited by the print.
   No maker rebates.
6. **Settlement** uses Polymarket's resolved outcome (Gamma). Only after `fallback_after_s` (2 h) with
   no official result is a trade settled on the bot's own TWAP, flagged `fallback`. If the bot cannot
   tell either, the trade counts as a loss (`fallback_loss`). A later official result overrides it.
7. **Edge vanished** (reported): the share of signals that, after latency, did not fill or filled with
   non-positive edge under the model re-evaluated at arrival.

## Reporting (`/report`, dashboard `/`)

* P&L after fees and fees paid.
* %/day compounded over total wall-clock time since first start, including idle and down time,
  with open positions valued at cost.
* Win rate vs predicted, trade count, max drawdown, per-day P&L.
* Bootstrap 95% CI of the daily return (trade-level resample × trades/day; day-level once there
  are 5 or more days).
* Trades and days needed for the 0.5%/day target to be distinguishable from zero (two-sided 5%, and
  with 80% power).
* Brier score and reliability tables, and the share of signals whose edge vanished after latency.
* A `STATISTICALLY MEANINGLESS SAMPLE` label until both the trade count reaches the 80%-power
  requirement and 7 days have passed.

### Resetting the paper account

The dashboard's **Reset database** button (`POST /api/reset` with JSON body `{"confirm": "RESET"}`)
deletes all signals, fills, settlements and predictions, puts the bankroll back to
`starting_bankroll` and restarts the %/day clock. Market metadata and outcomes are kept, because live
windows still need their strikes. Recordings are untouched, so a replay still covers the full
history. The endpoint only accepts `application/json`, which a cross-site form cannot send. **If the
dashboard is reachable from the internet, set `POLYBOT_ADMIN_TOKEN`**: the endpoint then requires it
in the `X-Admin-Token` header, and the button asks for it.

## Known optimistic biases

The simulation still flatters results in these ways:

1. **No market impact or information leakage.** Our order never moves the book or the other
   participants. The 50% depth haircut is a guess, not a measurement.
2. **Latency is an HTTPS GET round trip,** not a real order POST with EIP-712 signing and matching
   engine time. The 50 ms extra and 150 ms floor are guesses. A real bot also needs the order *ack*
   before it knows it filled.
3. **Book reconstruction:** if we miss a websocket message, depth *behind* the best level can still be
   wrong until the next `book` snapshot (only the top of book is checked against the exchange). The
   book we see is also already one feed delay old.
4. **Queue position for maker orders** is modelled conservatively, but cancellations ahead of us and
   hidden size are ignored. Maker mode is off by default.
5. **Fee assumptions:** `exponent` other than 1 is applied as `(p(1−p))^exponent`, which is
   unverified. Taker rebates are ignored (pessimistic) but so are possible future fee increases.
6. **Unlimited availability:** we assume the market accepts orders right up to its end (we stop at
   `min_tau_s` = 15 s). Geographic restrictions (`restricted: true`) are ignored: you may not legally
   be able to trade these markets at all.
7. **Selection in the model:** vol floor, inflation and basis-error parameters were set by hand, not
   fitted out of sample. Any later tuning on the same recordings overfits.
8. **Small sample:** 5-minute binary outcomes are very noisy. See `trades_needed_80pct_power`.
9. **Restarts:** simulated orders in flight at a restart are cancelled (neutral), and the current
   window may be skipped because the strike is unknown (conservative).
10. **Snapshot-polling mode** may itself be blocked or rate-limited by Polymarket at any time.

## Configuration

All settings live in `src/polybot/config.py`. Each one can be set as `POLYBOT_<NAME>` in the
environment, or in a TOML file named by `POLYBOT_CONFIG`. The most relevant:

| Variable | Default | Meaning |
| --- | --- | --- |
| `POLYBOT_DATA_DIR` | `./data` (`/data` in the image) | SQLite, recordings, logs |
| `POLYBOT_HTTP_PORT` | 8080 | web/health port |
| `POLYBOT_MIN_EDGE` | 0.03 | required P(win) − all-in cost |
| `POLYBOT_STAKE_FRACTION` / `POLYBOT_MAX_STAKE_USD` | 0.02 / 50 | sizing |
| `POLYBOT_STARTING_BANKROLL` | 1000 | paper bankroll (USDC) |
| `POLYBOT_BOOK_TAKE_FRACTION` | 0.5 | share of displayed depth we may take |
| `POLYBOT_LATENCY_EXTRA_MS` / `_FLOOR` | 50 / 150 | latency model |
| `POLYBOT_VOL_INFLATION` / `POLYBOT_BASIS_ERR_INFLATION` | 1.0 / 1.5 | vol bias (keep at 1) / basis error floor |
| `POLYBOT_MODEL_SD_BAND` | 1.25 | trade on the worst P(win) over sd × [1/band, band] |
| `POLYBOT_MAX_MODEL_MARKET_GAP` | 0.20 | skip if the model is this far from the mid (0 = off) |
| `POLYBOT_KELLY_FRACTION` | 0.25 | stake cap as a fraction of Kelly (0 = flat stake) |
| `POLYBOT_ADMIN_TOKEN` | empty | required by the reset button if set |
| `POLYBOT_STALE_EXCHANGE_MS` / `_BOOK_MS` / `_ORACLE_MS` | 2000 / 5000 / 12000 | skip trading if older |
| `POLYBOT_ENABLE_MAKER` | false | simulate resting bids |
| `POLYBOT_RECORD_RETENTION_DAYS` | 14 | about 3 GB/day compressed |

## Running

```sh
uv venv -p 3.12 && uv pip install -r requirements.lock -e '.[dev]'
.venv/bin/pytest                      # unit, fixture and live-vs-replay tests
.venv/bin/polybot run                 # live paper trading, http://localhost:8080
.venv/bin/polybot report              # print the report from data/polybot.sqlite
.venv/bin/polybot replay --from 2026-10-29 --to 2026-10-30 --set min_edge=0.05
```

Docker: `docker run -p 8080:8080 -v polybot-data:/data ghcr.io/first-uninteresting-username/polybot`.

## Deployment (Orchard)

The service runs on a self-hosted [Orchard](https://docs.orchard.my) (k3s PaaS). Orchard builds this
repo's `Dockerfile` itself through its GitHub App, and **auto-deploys on every push to `main`**. No
credentials live in this repo or in GitHub secrets. GitHub Actions runs lint, type checks and tests
on every push and publishes an image to GHCR. The Orchard app:

* `polybot`: 1 replica, `recreate` strategy (SQLite needs a single writer), port 8080.
* PVC `polybot-data` mounted at `/data`. It survives restarts, redeploys and rebuilds.
* HTTP liveness/readiness on `/health`. Kubernetes restarts the pod on crashes, and the node's
  k3s brings it back after reboots.
* Logs: `/logs` on the web page, Orchard's log view (stdout JSON), and daily rotated files in
  `/data/logs`.
