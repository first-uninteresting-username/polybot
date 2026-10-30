"""HTTP endpoints: /health, /report, /api/state, /api/signals, /logs, a small dashboard at /, and
POST /api/reset (the only one that changes anything: it wipes the paper account)."""

from __future__ import annotations

import asyncio
import hmac
import json
import time
from collections.abc import Callable
from importlib import resources
from typing import Any

from aiohttp import web

from .logging_setup import RING


def make_app(health: Callable[[], dict[str, Any]], report: Callable[[], dict[str, Any]],
             state: Callable[[], dict[str, Any]], signals: Callable[[int], list[dict[str, Any]]],
             reset: Callable[[], dict[str, Any]] | None = None, admin_token: str = "") -> web.Application:
    app = web.Application()
    page = resources.files("polybot").joinpath("dashboard.html").read_text()

    def js(obj: Any, status: int = 200) -> web.Response:
        return web.Response(text=json.dumps(obj, default=str, indent=1), status=status,
                            content_type="application/json")

    async def h_health(_: web.Request) -> web.Response:
        h = health()
        return js(h, 200 if h.get("ok") else 503)

    async def h_report(_: web.Request) -> web.Response:
        return js(await asyncio.to_thread(report))  # takes ~0.5 s: must not block the feeds

    async def h_state(_: web.Request) -> web.Response:
        return js(state())

    async def h_signals(req: web.Request) -> web.Response:
        return js(signals(min(int(req.query.get("limit", "100")), 1000)))

    async def h_logs(req: web.Request) -> web.Response:
        n = min(int(req.query.get("n", "300")), len(RING.lines) or 1)
        level = req.query.get("level", "").upper()
        lines = [ln for ln in list(RING.lines)[-5000:] if not level or f'"level": "{level}"' in ln][-n:]
        return web.Response(text="\n".join(lines) + "\n", content_type="text/plain")

    async def h_reset(req: web.Request) -> web.Response:
        if reset is None:
            return js({"ok": False, "error": "reset not available"}, 404)
        # A JSON body cannot be sent cross-site without a CORS preflight, so this also stops CSRF.
        if req.content_type != "application/json":
            return js({"ok": False, "error": "expected application/json"}, 415)
        if admin_token and not hmac.compare_digest(req.headers.get("X-Admin-Token", ""), admin_token):
            return js({"ok": False, "error": "admin token required", "token_required": True}, 401)
        try:
            body = await req.json()
        except ValueError:
            body = None
        if not isinstance(body, dict) or body.get("confirm") != "RESET":
            return js({"ok": False, "error": 'body must be {"confirm": "RESET"}'}, 400)
        return js(reset())

    async def h_index(_: web.Request) -> web.Response:
        return web.Response(text=page, content_type="text/html")

    app.router.add_get("/", h_index)
    app.router.add_get("/health", h_health)
    app.router.add_get("/report", h_report)
    app.router.add_get("/api/state", h_state)
    app.router.add_get("/api/signals", h_signals)
    app.router.add_get("/logs", h_logs)
    app.router.add_post("/api/reset", h_reset)
    app["started"] = time.time()
    return app
