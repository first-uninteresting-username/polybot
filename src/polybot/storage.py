"""SQLite persistence for markets, signals, fills, settlements and model predictions.

Writes are idempotent (deterministic primary keys + INSERT OR IGNORE / guarded UPDATEs), so a restart
cannot double-count a fill or a settlement.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS markets (
  slug TEXT PRIMARY KEY, start_ts INTEGER, end_ts INTEGER, token_up TEXT, token_down TEXT,
  fee_rate REAL, fee_exponent REAL, tick REAL, min_size REAL, twap_window_s INTEGER,
  strike REAL, strike_source TEXT,
  winner TEXT, final_price REAL, gamma_strike REAL, settled_by TEXT, resolved_ms INTEGER
);
CREATE TABLE IF NOT EXISTS signals (
  id TEXT PRIMARY KEY, slug TEXT, ts_ms INTEGER, side TEXT, kind TEXT,
  p_win REAL, best_ask REAL, limit_price REAL, edge REAL, tau_s REAL,
  latency_ms REAL, arrival_ms INTEGER, budget REAL, status TEXT, detail TEXT
);
CREATE TABLE IF NOT EXISTS fills (
  id TEXT PRIMARY KEY, slug TEXT, side TEXT, ts_ms INTEGER, kind TEXT,
  gross_shares REAL, net_shares REAL, cost REAL, fee REAL, avg_price REAL,
  p_win_decision REAL, p_win_arrival REAL, edge_arrival REAL, levels TEXT
);
CREATE TABLE IF NOT EXISTS settlements (
  id TEXT PRIMARY KEY, slug TEXT, ts_ms INTEGER, won INTEGER, payout REAL, pnl REAL, settled_by TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
  slug TEXT, ts_ms INTEGER, tau_s REAL, p_up REAL, PRIMARY KEY (slug, ts_ms)
);
CREATE INDEX IF NOT EXISTS idx_signals_slug ON signals(slug);
CREATE INDEX IF NOT EXISTS idx_fills_slug ON fills(slug);
"""


class Storage:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)

    # --- meta ---
    def get_meta(self, key: str) -> str | None:
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else None

    def set_meta(self, key: str, value: str, only_if_missing: bool = False) -> None:
        verb = "INSERT OR IGNORE" if only_if_missing else "INSERT OR REPLACE"
        self.db.execute(f"{verb} INTO meta(key, value) VALUES (?, ?)", (key, value))

    # --- markets ---
    def upsert_market(self, row: dict[str, Any]) -> None:
        cols = ["slug", "start_ts", "end_ts", "token_up", "token_down", "fee_rate", "fee_exponent", "tick",
                "min_size", "twap_window_s"]
        self.db.execute(
            f"INSERT INTO markets({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
            "ON CONFLICT(slug) DO UPDATE SET fee_rate=excluded.fee_rate, fee_exponent=excluded.fee_exponent, "
            "tick=excluded.tick, min_size=excluded.min_size",
            [row[c] for c in cols],
        )

    def set_strike(self, slug: str, strike: float, source: str) -> None:
        self.db.execute("UPDATE markets SET strike=?, strike_source=? WHERE slug=? AND strike IS NULL",
                        (strike, source, slug))

    def set_resolution(self, slug: str, winner: str, final_price: float | None, gamma_strike: float | None,
                       settled_by: str, ts_ms: int) -> None:
        self.db.execute(
            "UPDATE markets SET winner=?, final_price=?, gamma_strike=?, settled_by=?, resolved_ms=? WHERE slug=?",
            (winner, final_price, gamma_strike, settled_by, ts_ms, slug),
        )

    def market(self, slug: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM markets WHERE slug=?", (slug,)).fetchone()

    def unresolved_slugs(self, before_end_ts: int, after_end_ts: int = 0) -> list[str]:
        """Markets past their end that we care about (have fills or predictions) and lack an official result."""
        rows = self.db.execute(
            "SELECT m.slug FROM markets m WHERE m.end_ts <= ? AND m.end_ts >= ? AND (m.settled_by IS NULL OR m.settled_by != 'gamma') "
            "AND (EXISTS (SELECT 1 FROM fills f WHERE f.slug=m.slug) OR "
            "EXISTS (SELECT 1 FROM predictions p WHERE p.slug=m.slug)) ORDER BY m.end_ts",
            (before_end_ts, after_end_ts),
        ).fetchall()
        return [r[0] for r in rows]

    # --- signals / fills ---
    def insert_signal(self, row: dict[str, Any]) -> bool:
        cols = list(row)
        cur = self.db.execute(
            f"INSERT OR IGNORE INTO signals({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [row[c] for c in cols],
        )
        return cur.rowcount == 1

    def update_signal(self, sid: str, status: str, detail: dict[str, Any] | None = None) -> None:
        self.db.execute("UPDATE signals SET status=?, detail=? WHERE id=?",
                        (status, json.dumps(detail or {}), sid))

    def pending_signals(self) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM signals WHERE status IN ('pending','resting')").fetchall()

    def entries_for(self, slug: str) -> int:
        r = self.db.execute(
            "SELECT COUNT(*) FROM signals WHERE slug=? AND status IN ('pending','resting','filled','partial')",
            (slug,),
        ).fetchone()
        return int(r[0])

    def insert_fill(self, row: dict[str, Any]) -> bool:
        cols = list(row)
        cur = self.db.execute(
            f"INSERT OR IGNORE INTO fills({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [row[c] for c in cols],
        )
        return cur.rowcount == 1

    def fills_for(self, slug: str) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM fills WHERE slug=?", (slug,)).fetchall()

    def upsert_settlement(self, row: dict[str, Any]) -> None:
        """Insert, or overwrite a provisional (fallback) settlement with the official one."""
        self.db.execute(
            "INSERT INTO settlements(id, slug, ts_ms, won, payout, pnl, settled_by) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET won=excluded.won, payout=excluded.payout, pnl=excluded.pnl, "
            "settled_by=excluded.settled_by, ts_ms=excluded.ts_ms "
            "WHERE settlements.settled_by != 'gamma'",
            (row["id"], row["slug"], row["ts_ms"], row["won"], row["payout"], row["pnl"], row["settled_by"]),
        )

    # --- predictions ---
    def insert_prediction(self, slug: str, ts_ms: int, tau_s: float, p_up: float) -> None:
        self.db.execute("INSERT OR IGNORE INTO predictions VALUES (?,?,?,?)", (slug, ts_ms, tau_s, p_up))

    # --- money ---
    def cash(self, starting: float) -> float:
        spent = self.db.execute("SELECT COALESCE(SUM(cost),0) FROM fills").fetchone()[0]
        paid = self.db.execute("SELECT COALESCE(SUM(payout),0) FROM settlements").fetchone()[0]
        return starting - float(spent) + float(paid)

    def open_cost(self) -> float:
        r = self.db.execute(
            "SELECT COALESCE(SUM(f.cost),0) FROM fills f LEFT JOIN settlements s ON s.id=f.id WHERE s.id IS NULL"
        ).fetchone()
        return float(r[0])

    def reset_trading(self, now_ms: int) -> None:
        """Start the paper account over: drop all signals, fills, settlements and predictions and restart the
        performance clock. Market metadata and outcomes stay (live windows still need their strikes)."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for table in ("signals", "fills", "settlements", "predictions"):
                self.db.execute(f"DELETE FROM {table}")
            self.set_meta("first_start_ms", str(now_ms))
            self.set_meta("last_reset_ms", str(now_ms))
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def query(self, sql: str, args: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        return self.db.execute(sql, args).fetchall()

    def close(self) -> None:
        self.db.close()
