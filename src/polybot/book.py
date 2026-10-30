"""Local L2 order book for one outcome token, maintained from CLOB websocket events."""

from __future__ import annotations

from dataclasses import dataclass, field

from .events import BookDelta, BookSnapshot


def _key(price: float) -> float:
    return round(price, 6)


@dataclass
class OrderBook:
    asset_id: str
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    last_update_ms: int = 0  # local receive time of the last applied event
    has_snapshot: bool = False
    desynced: bool = False  # we lack a level the exchange reports as best; unreliable until next snapshot
    tick: float = 0.01

    def apply_snapshot(self, ev: BookSnapshot) -> None:
        self.bids = {_key(p): s for p, s in ev.bids if s > 0}
        self.asks = {_key(p): s for p, s in ev.asks if s > 0}
        self.last_update_ms = ev.recv_ms
        self.has_snapshot = True
        self.desynced = False

    def apply_delta(self, ev: BookDelta) -> None:
        side = self.bids if ev.side == "BUY" else self.asks
        k = _key(ev.price)
        if ev.size <= 0:
            side.pop(k, None)
        else:
            side[k] = ev.size
        self.last_update_ms = ev.recv_ms
        self._reconcile(ev.best_bid, ev.best_ask)

    def _reconcile(self, best_bid: float | None, best_ask: float | None) -> None:
        """``price_change`` reports the exchange's best bid/ask after the change. Levels better than that
        are phantom liquidity left over from a message we never got (trades remove levels without a
        ``price_change``), so drop them. If we are missing the reported level itself, we cannot invent its
        size: flag the book until the next snapshot."""
        if best_ask is not None:
            for p in [p for p in self.asks if p < best_ask - 1e-9]:
                del self.asks[p]
            mine = self.best_ask()
            if (mine is None and best_ask < 1.0 - 1e-9) or (mine is not None and abs(mine - best_ask) > 1e-9):
                self.desynced = True
        if best_bid is not None:
            for p in [p for p in self.bids if p > best_bid + 1e-9]:
                del self.bids[p]
            mine = self.best_bid()
            if (mine is None and best_bid > 1e-9) or (mine is not None and abs(mine - best_bid) > 1e-9):
                self.desynced = True

    def asks_ascending(self) -> list[tuple[float, float]]:
        return sorted(self.asks.items())

    def bids_descending(self) -> list[tuple[float, float]]:
        return sorted(self.bids.items(), reverse=True)

    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    def copy_levels(self) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
        return self.bids_descending(), self.asks_ascending()
