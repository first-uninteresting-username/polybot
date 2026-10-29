"""Read-only HTTP endpoints: /health, /report, /api/state, /api/signals, /logs and a small dashboard at /."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from importlib import resources
from typing import Any

from aiohttp import web

from .logging_setup import RING


def make_app(health: Callable[[], dict[str, Any]], report: Callable[[], dict[str, Any]],
             state: Callable[[], dict[str, Any]], signals: Callable[[int], list[dict[str, Any]]]) -> web.Application:
    app = web.Application()
    page = resources.files("polybot").joinpath("dashboard.html").read_text()

    def js(obj: Any, status: int = 200) -> web.Response:
        return web.Response(text=json.dumps(obj, default=str, indent=1), status=status,
                            content_type="application/json")

    async def h_health(_: web.Request) -> web.Response:
        h = health()
        return js(h, 200 if h.get("ok") else 503)

    async def h_report(_: web.Request) -> web.Response:
        return js(report())

    async def h_state(_: web.Request) -> web.Response:
        return js(state())

    async def h_signals(req: web.Request) -> web.Response:
        return js(signals(min(int(req.query.get("limit", "100")), 1000)))

    async def h_logs(req: web.Request) -> web.Response:
        n = min(int(req.query.get("n", "300")), len(RING.lines) or 1)
        level = req.query.get("level", "").upper()
        lines = [ln for ln in list(RING.lines)[-5000:] if not level or f'"level": "{level}"' in ln][-n:]
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    async def h_index(_: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html")

    app.router.add_get("/", h_index)
    app.router.add_get("/health", h_health)
    app.router.add_get("/report", h_report)
    app.router.add_get("/api/state", h_state)
    app.router.add_get("/api/signals", h_signals)
    app.router.add_get("/logs", h_logs)
    app["started"] = time.time()
    return app
