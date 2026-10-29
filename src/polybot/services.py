"""HTTP-polling services: market discovery, resolution polling and CLOB latency probing.

All of them push their results onto the same queue as the websocket feeds, so the results are recorded
and replayed like any other input. Only public, unauthenticated GET endpoints are used:
* Gamma  GET /markets?slug=btc-updown-5m-<ts>   (market metadata, token ids, fee schedule)
* Gamma  GET /events?slug=btc-updown-5m-<ts>    (resolution: closed, outcomePrices, eventMetadata)
* CLOB   GET /time                               (round-trip latency probe)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable

import aiohttp

from .config import Settings
from .events import SRC_GAMMA_MARKET, SRC_LATENCY, SRC_RESOLUTION, parse_resolution
from .feeds.ws import Queue, now_ms

log = logging.getLogger("polybot.services")
HEADERS = {"User-Agent": "polybot-paper-trader/0.1 (paper trading only)"}


async def _get_json(session: aiohttp.ClientSession, url: str) -> object:
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
        r.raise_for_status()
        return await r.json(content_type=None)


class Discovery:
    def __init__(self, s: Settings, queue: Queue, session: aiohttp.ClientSession):
        self.s, self.queue, self.session = s, queue, session
        self.last_ok_wall_ms = 0
        self.last_error = ""
        self._seen_accepting: dict[str, bool] = {}

    async def run(self) -> None:
        while True:
            try:
                await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = repr(exc)[:300]
                log.warning("discovery failed: %s", self.last_error)
            await asyncio.sleep(self.s.discovery_interval_s)

    async def poll_once(self) -> None:
        w = self.s.window_seconds
        now = int(time.time())
        base = now - now % w
        for k in range(0, self.s.discovery_lookahead_windows + 1):
            slug = f"{self.s.slug_prefix}{base + k * w}"
            data = await _get_json(self.session, f"{self.s.gamma_url}/markets?slug={slug}")
            if not isinstance(data, list) or not data:
                continue
            m = data[0]
            # Record each market once, plus again whenever acceptingOrders changes.
            key = f"{slug}:{m.get('acceptingOrders')}:{m.get('feeSchedule')}"
            if key in self._seen_accepting:
                continue
            self._seen_accepting[key] = True
            await self.queue.put((SRC_GAMMA_MARKET, now_ms(), json.dumps(m, separators=(",", ":"))))
        self.last_ok_wall_ms = now_ms()
        if len(self._seen_accepting) > 500:
            self._seen_accepting = dict(list(self._seen_accepting.items())[-100:])


class Resolver:
    """Polls Gamma for finished windows until Polymarket reports a final outcome."""

    def __init__(self, s: Settings, queue: Queue, session: aiohttp.ClientSession,
                 pending_slugs: Callable[[], list[str]]):
        self.s, self.queue, self.session = s, queue, session
        self.pending_slugs = pending_slugs
        self.resolved: set[str] = set()
        self.last_error = ""
        self.polls = 0

    async def run(self) -> None:
        while True:
            try:
                for slug in self.pending_slugs():
                    if slug in self.resolved:
                        continue
                    data = await _get_json(self.session, f"{self.s.gamma_url}/events?slug={slug}")
                    self.polls += 1
                    if isinstance(data, list) and data:
                        raw = json.dumps(data[0], separators=(",", ":"))
                        if parse_resolution(0, raw):
                            self.resolved.add(slug)
                            await self.queue.put((SRC_RESOLUTION, now_ms(), raw))
                    await asyncio.sleep(0.2)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = repr(exc)[:300]
                log.warning("resolver failed: %s", self.last_error)
            await asyncio.sleep(self.s.resolution_poll_s)


class LatencyProbe:
    """Measures HTTPS round-trip time to the CLOB on a kept-alive connection (as an order POST would use)."""

    def __init__(self, s: Settings, queue: Queue, session: aiohttp.ClientSession):
        self.s, self.queue, self.session = s, queue, session
        self.samples: list[float] = []
        self.last_error = ""

    async def run(self) -> None:
        while True:
            try:
                t0 = time.perf_counter()
                async with self.session.get(f"{self.s.clob_url}/time",
                                            timeout=aiohttp.ClientTimeout(total=10)) as r:
                    await r.read()
                    r.raise_for_status()
                rtt = (time.perf_counter() - t0) * 1000
                self.samples = (self.samples + [rtt])[-1000:]
                await self.queue.put((SRC_LATENCY, now_ms(), json.dumps({"rtt_ms": round(rtt, 3)})))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.last_error = repr(exc)[:300]
                log.warning("latency probe failed: %s", self.last_error)
            await asyncio.sleep(self.s.latency_probe_interval_s)
