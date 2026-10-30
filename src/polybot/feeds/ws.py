"""Websocket feeds with reconnect/backoff. Each pushes ``(source, recv_ms, raw)`` onto a shared queue.

Subscription formats (checked against live servers 2026-10-29, samples in tests/fixtures):
* Binance:   wss://stream.binance.com:9443/ws/btcusdt@trade (no subscribe frame)
* Coinbase:  {"type":"subscribe","product_ids":["BTC-USD"],"channels":["ticker","heartbeat"]}
* CLOB:      {"assets_ids":[...],"type":"market","custom_feature_enabled":true}; text "PING" every 10 s;
             later {"assets_ids":[...],"operation":"subscribe"|"unsubscribe"}
* RTDS:      {"action":"subscribe","subscriptions":[{"topic":"crypto_prices_chainlink","type":"*",
             "filters":"{\\"symbol\\":\\"btc/usd\\"}"}, ...]}; text "PING" every 5 s
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import websockets

log = logging.getLogger("polybot.feeds")

Queue = asyncio.Queue[tuple[str, int, str]]

BACKOFF_RESET_AFTER_S = 30.0


def now_ms() -> int:
    return time.time_ns() // 1_000_000


@dataclass
class FeedStatus:
    name: str
    connected: bool = False
    connects: int = 0
    messages: int = 0
    last_msg_wall_ms: int = 0
    last_error: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class WsFeed:
    def __init__(self, name: str, source: str, url: str, queue: Queue,
                 subscribe: Callable[[], list[str]] | None = None,
                 ping_text: str | None = None, ping_interval_s: float = 10.0,
                 max_silence_s: float = 30.0, update_marker: str | None = None,
                 resnapshot_after_s: float | None = None):
        self.name = name
        self.update_marker = update_marker
        self.resnapshot_after_s = resnapshot_after_s
        self.source = source
        self.url = url
        self.queue = queue
        self.subscribe = subscribe
        self.ping_text = ping_text
        self.ping_interval_s = ping_interval_s
        self.max_silence_s = max_silence_s
        self.status = FeedStatus(name)
        self._ws: Any = None
        self._stop = asyncio.Event()

    async def run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            started = time.monotonic()
            try:
                if await self._session():
                    self.status.connected = False
                    continue  # deliberate re-snapshot: reconnect right away, no backoff
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # network errors of all kinds: reconnect
                self.status.last_error = f"{type(exc).__name__}: {exc}"[:300]
                log.warning("feed %s disconnected: %s", self.name, self.status.last_error)
            # A session that ran for a while was a real connection, so this is a new outage, not a retry
            # of the last one. Without this every disconnect waits the full 15-45 s of the capped backoff.
            if time.monotonic() - started >= BACKOFF_RESET_AFTER_S:
                backoff = 1.0
            self.status.connected = False
            self._ws = None
            delay = min(backoff, 30.0) * (0.5 + random.random())
            backoff = min(backoff * 2, 30.0)
            await asyncio.sleep(delay)

    async def ready_to_connect(self) -> None:
        """Hook: wait until there is something to subscribe to."""

    async def _session(self) -> bool:
        await self.ready_to_connect()
        async with websockets.connect(self.url, open_timeout=15, ping_interval=20, ping_timeout=20,
                                      max_size=8 * 1024 * 1024,
                                      close_timeout=1 if self.resnapshot_after_s else 5) as ws:
            self._ws = ws
            self.status.connected = True
            self.status.connects += 1
            # first connect at INFO; deliberate re-snapshot reconnects would flood the log
            (log.info if self.status.connects == 1 or not self.resnapshot_after_s else log.debug)(
                "feed %s connected", self.name)
            for frame in (self.subscribe() if self.subscribe else []):
                await ws.send(frame)
            await self.on_connected()
            last_ping = time.monotonic()
            last_msg = time.monotonic()
            connected_at = last_update = time.monotonic()
            while True:
                timeout = self.ping_interval_s if self.ping_text else self.max_silence_s
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=min(timeout, 1.0))
                except TimeoutError:
                    msg = None
                mono = time.monotonic()
                if msg is not None:
                    raw = msg if isinstance(msg, str) else msg.decode("utf-8", "replace")
                    t = now_ms()
                    last_msg = mono
                    self.status.messages += 1
                    self.status.last_msg_wall_ms = t
                    if raw and raw != "PONG":
                        await self.queue.put((self.source, t, raw))
                        if self.update_marker and self.update_marker in raw:
                            last_update = mono
                            self.status.extra["mode"] = "streaming"
                if self.ping_text and mono - last_ping >= self.ping_interval_s:
                    await ws.send(self.ping_text)
                    last_ping = mono
                if mono - last_msg > self.max_silence_s:
                    raise TimeoutError(f"no message for {self.max_silence_s}s")
                if (self.resnapshot_after_s and mono - last_update >= self.resnapshot_after_s
                        and mono - connected_at >= self.resnapshot_after_s):
                    # Server sends the snapshot but no live updates: reconnect for a fresh snapshot.
                    self.status.extra["mode"] = "snapshot_polling"
                    self.status.extra["resnapshots"] = self.status.extra.get("resnapshots", 0) + 1
                    return True

    async def on_connected(self) -> None:
        pass

    async def send(self, frame: str) -> bool:
        ws = self._ws
        if ws is None:
            return False
        try:
            await ws.send(frame)
            return True
        except Exception as exc:
            log.warning("feed %s send failed: %s", self.name, exc)
            return False

    def stop(self) -> None:
        self._stop.set()


class ClobMarketFeed(WsFeed):
    """Polymarket CLOB market channel with a changing set of outcome tokens."""

    def __init__(self, url: str, queue: Queue):
        super().__init__("clob", "clob", url, queue, subscribe=self._initial, ping_text="PING",
                         ping_interval_s=10.0, max_silence_s=60.0)
        self.wanted: set[str] = set()
        self.subscribed: set[str] = set()
        self._have_tokens = asyncio.Event()

    def _initial(self) -> list[str]:
        self.subscribed = set(self.wanted)
        return [json.dumps({"assets_ids": sorted(self.subscribed), "type": "market",
                            "custom_feature_enabled": True})]

    async def ready_to_connect(self) -> None:
        await self._have_tokens.wait()

    async def set_tokens(self, tokens: set[str]) -> None:
        self.wanted = set(tokens)
        if tokens:
            self._have_tokens.set()
        if self._ws is None:
            return
        add, remove = self.wanted - self.subscribed, self.subscribed - self.wanted
        if add and await self.send(json.dumps({"assets_ids": sorted(add), "operation": "subscribe",
                                               "custom_feature_enabled": True})):
            self.subscribed |= add
        if remove and await self.send(json.dumps({"assets_ids": sorted(remove), "operation": "unsubscribe"})):
            self.subscribed -= remove
        self.status.extra["subscribed"] = len(self.subscribed)


def binance_feed(url: str, queue: Queue) -> WsFeed:
    return WsFeed("binance", "binance", url, queue, max_silence_s=30.0)


def coinbase_feed(url: str, queue: Queue) -> WsFeed:
    sub = json.dumps({"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["ticker", "heartbeat"]})
    return WsFeed("coinbase", "coinbase", url, queue, subscribe=lambda: [sub], max_silence_s=30.0)


def rtds_feed(url: str, queue: Queue, resnapshot_after_s: float = 5.0) -> WsFeed:
    flt = json.dumps({"symbol": "btc/usd"})
    sub = json.dumps({"action": "subscribe", "subscriptions": [
        {"topic": "crypto_prices_chainlink", "type": "*", "filters": flt},
        {"topic": "crypto_prices_twap_sixty", "type": "*", "filters": flt},
    ]})
    # Since 2026-10-29 ~20:21 UTC new RTDS connections get the 60 s history snapshot but no live updates
    # (RTDS price topics are deprecated in favour of the authenticated PolyBolt feed, which we must not
    # use). If no update frame arrives for ``resnapshot_after_s`` we reconnect to get a fresh snapshot.
    return WsFeed("rtds", "rtds", url, queue, subscribe=lambda: [sub], ping_text="PING",
                  ping_interval_s=5.0, max_silence_s=30.0, update_marker='"type":"update"',
                  resnapshot_after_s=resnapshot_after_s)
