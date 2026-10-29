"""CLI: ``polybot run`` (live paper trading), ``polybot replay`` (backtest recordings), ``polybot report``."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import sys
from typing import Any

from .config import Settings, _coerce, load_settings
from .logging_setup import setup_logging


def _overrides(pairs: list[str]) -> dict[str, Any]:
    types = {f.name: f.type for f in dataclasses.fields(Settings)}
    out: dict[str, Any] = {}
    for p in pairs:
        k, _, v = p.partition("=")
        if k not in types:
            raise SystemExit(f"unknown setting {k}")
        out[k] = _coerce(v, types[k])
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="polybot")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("run", help="live paper trading service")
    rp = sub.add_parser("replay", help="run the strategy over recordings")
    rp.add_argument("--from", dest="start", help="first day YYYY-MM-DD")
    rp.add_argument("--to", dest="end", help="last day YYYY-MM-DD")
    rp.add_argument("--db", default=":memory:", help="sqlite output (default in-memory)")
    rp.add_argument("--set", action="append", default=[], help="override setting, e.g. min_edge=0.05")
    sub.add_parser("report", help="print the report from the live database")
    args = ap.parse_args(argv)

    if args.cmd in (None, "run"):
        s = load_settings()
        setup_logging(s.log_level, s.data_path / "logs")
        from .app import LiveApp

        asyncio.run(LiveApp(s).run())
    elif args.cmd == "replay":
        s = load_settings(**_overrides(args.set))
        setup_logging("WARNING")
        from .recorder import recording_files
        from .replay import iter_files, replay
        from .reporting import build_report

        files = recording_files(s.data_path / "recordings", args.start, args.end)
        if not files:
            raise SystemExit("no recordings found")
        eng = replay(s, iter_files(files), args.db)
        json.dump(build_report(eng.store, s.starting_bankroll, eng.now_ms), sys.stdout, indent=1, default=str)
        print()
    elif args.cmd == "report":
        s = load_settings()
        import time

        from .reporting import build_report
        from .storage import Storage

        json.dump(build_report(Storage(s.data_path / "polybot.sqlite"), s.starting_bankroll,
                               int(time.time() * 1000)), sys.stdout, indent=1, default=str)
        print()


if __name__ == "__main__":
    main()
