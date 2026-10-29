"""Live paper-trading service: feeds -> queue -> (recorder, engine) + web server."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
import time
from typing import Any

import aiohttp
from aiohttp import web

from .config import Settings
from .engine import Engine
from .feeds.ws import ClobMarketFeed, Queue, WsFeed, binance_feed, coinbase_feed, now_ms, rtds_feed
from .recorder import Recorder
from .reporting import build_report
from .services import HEADERS, Discovery, LatencyProbe, Resolver
from .storage import Storage
from .web import make_app

log = logging.getLogger("polybot.app")


class Pipeline:
    """The single code path every live message goes through. Replay calls ``engine.on_raw`` on the
    recorded tuples in the same order, so both modes see identical engine inputs."""

    def __init__(self, settings: Settings, store: Storage, recorder: Recorder | None):
        self.s = settings
        self.store = store
        self.recorder = recorder
        self.engine = Engine(settings, store)
        self.processed = 0
        self.last_recv_ms = 0

    def process(self, source: str, recv_ms: int, raw: str) -> None:
        if self.recorder is not None:
            self.recorder.write(recv_ms, source, raw)
        self.engine.on_raw(source, recv_ms, raw)
        self.processed += 1
        self.last_recv_ms = recv_ms


class LiveApp:
    def __init__(self, settings: Settings):
        self.s = settings
        data = settings.data_path
        data.mkdir(parents=True, exist_ok=True)
        self.store = Storage(data / "polybot.sqlite")
        self.store.set_meta("first_start_ms", str(now_ms()), only_if_missing=True)
        self.store.set_meta("last_start_ms", str(now_ms()))
        self.recorder = Recorder(data / "recordings", retention_days=settings.record_retention_days)
        self.pipeline = Pipeline(settings, self.store, self.recorder)
        self.queue: Queue = asyncio.Queue(maxsize=200_000)
        self.clob = ClobMarketFeed(settings.clob_ws_url, self.queue)
        self.feeds: list[WsFeed] = [
            binance_feed(settings.binance_ws_url, self.queue),
            coinbase_feed(settings.coinbase_ws_url, self.queue),
            rtds_feed(settings.rtds_ws_url, self.queue, settings.rtds_resnapshot_s),
            self.clob,
        ]
        self.started_wall = time.time()
        self.discovery: Discovery | None = None
        self.resolver: Resolver | None = None
        self.probe: LatencyProbe | None = None

    @property
    def engine(self) -> Engine:
        return self.pipeline.engine

    def _pending_resolution(self) -> list[str]:
        now = int(time.time())
        return self.store.unresolved_slugs(before_end_ts=now, after_end_ts=now - 3 * 86400)

    async def consume(self) -> None:
        tokens: set[str] = set()
        while True:
            source, recv_ms, raw = await self.queue.get()
            try:
                self.pipeline.process(source, recv_ms, raw)
            except Exception:  # never let one bad message kill the service; it is in the recording
                log.exception("engine error on %s message", source)
            wanted = self.engine.wanted_tokens()
            if wanted != tokens:
                tokens = wanted
                await self.clob.set_tokens(wanted)

    def health(self) -> dict[str, Any]:
        wall = now_ms()
        problems = []
        feeds = {}
        for f in self.feeds:
            age = wall - f.status.last_msg_wall_ms if f.status.last_msg_wall_ms else None
            feeds[f.name] = dict(connected=f.status.connected, messages=f.status.messages, age_ms=age,
                                 connects=f.status.connects, last_error=f.status.last_error, **f.status.extra)
            # Judge by data age, not the instantaneous socket state: RTDS reconnects on purpose in
            # snapshot-polling mode, and a reconnect is not an outage.
            limit = 120_000 if f.name == "clob" else 30_000
            if age is None or age > limit:
                problems.append(f"feed_{f.name}")
        if self.queue.qsize() > 50_000:
            problems.append("queue_backlog")
        lag = wall - self.pipeline.last_recv_ms if self.pipeline.last_recv_ms else None
        if lag is None or lag > 30_000:
            problems.append("engine_stalled")
        warmup = time.time() - self.started_wall < 90
        return dict(
            ok=not problems or warmup, warming_up=warmup, problems=problems, paper_only=True,
            uptime_s=round(time.time() - self.started_wall, 1), queue=self.queue.qsize(),
            engine_lag_ms=lag, processed=self.pipeline.processed,
            recorder=dict(lines=self.recorder.lines, bytes_uncompressed=self.recorder.bytes_in),
            feeds=feeds,
            discovery=dict(last_ok_ms=self.discovery.last_ok_wall_ms if self.discovery else None,
                           last_error=self.discovery.last_error if self.discovery else ""),
            resolver=dict(polls=self.resolver.polls if self.resolver else 0,
                          last_error=self.resolver.last_error if self.resolver else ""),
            latency_probe=dict(samples=len(self.probe.samples) if self.probe else 0,
                               last_ms=self.probe.samples[-1] if self.probe and self.probe.samples else None,
                               last_error=self.probe.last_error if self.probe else ""),
            counters=self.engine.counters,
        )

    def state(self) -> dict[str, Any]:
        h = self.health()
        return dict(feeds=h["feeds"], engine=self.engine.snapshot())

    def report(self) -> dict[str, Any]:
        return build_report(self.store, self.s.starting_bankroll, now_ms())

    def signals(self, limit: int) -> list[dict[str, Any]]:
        rows = self.store.query(
            "SELECT s.*, f.net_shares, f.cost, f.fee, st.pnl, st.settled_by FROM signals s "
            "LEFT JOIN fills f ON f.id=s.id LEFT JOIN settlements st ON st.id=s.id "
            "ORDER BY s.ts_ms DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    async def run(self) -> None:
        log.info("polybot starting (PAPER TRADING ONLY - no orders are ever sent)")
        runner = web.AppRunner(make_app(self.health, self.report, self.state, self.signals))
        await runner.setup()
        await web.TCPSite(runner, self.s.http_host, self.s.http_port).start()
        log.info("http listening on %s:%s", self.s.http_host, self.s.http_port)
        async with aiohttp.ClientSession(headers=HEADERS) as session:
            self.discovery = Discovery(self.s, self.queue, session)
            self.resolver = Resolver(self.s, self.queue, session, self._pending_resolution)
            self.probe = LatencyProbe(self.s, self.queue, session)
            tasks = [asyncio.create_task(f.run(), name=f.name) for f in self.feeds]
            tasks += [asyncio.create_task(x.run(), name=type(x).__name__)
                      for x in (self.discovery, self.resolver, self.probe)]
            tasks.append(asyncio.create_task(self.consume(), name="consume"))
            stop = asyncio.Event()
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, stop.set)
            stopper = asyncio.create_task(stop.wait(), name="stop")
            try:
                done, _ = await asyncio.wait([*tasks, stopper], return_when=asyncio.FIRST_COMPLETED)
                if stopper in done:
                    log.info("shutdown requested; flushing recorder and database")
                for t in done:
                    if t is not stopper and not t.cancelled() and t.exception():
                        log.error("task %s died: %r", t.get_name(), t.exception())
                        raise t.exception()  # type: ignore[misc]
            finally:
                stopper.cancel()
                for t in tasks:
                    t.cancel()
                for t in tasks:
                    with contextlib.suppress(BaseException):
                        await t
                self.recorder.close()
                self.store.close()
                await runner.cleanup()
                log.info("stopped cleanly")
