"""Order orchestration (spec sections 14, 18, 31).

The Executor is the ONLY component allowed to send orders.  It:

1. Re-checks idempotency (no duplicate orders after reconnection - the same
   setup id or an identical pending order within the duplicate window is
   refused).
2. Sends via the configured BrokerClient with retries.
3. Verifies the result (ticket must come back; a "sent" order without a
   ticket is treated as UNKNOWN and recorded, never claimed as success).
4. Records everything in the journal and the state store.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, Optional

from ..config import Config
from ..constants import Direction
from ..journal.journal import Journal
from ..state_store import StateStore, utc_now_ms
from .broker_client import BrokerClient, BrokerError, OrderResult


class Executor:
    def __init__(self, cfg: Config, broker: BrokerClient, store: StateStore,
                 journal: Journal):
        self.cfg = cfg
        self.broker = broker
        self.store = store
        self.journal = journal

    # ------------------------------------------------------------------
    def _duplicate_guard(self, key: str) -> bool:
        """True if a duplicate within the window (order must NOT be sent)."""
        return self.store.idempotency_check_and_set(
            key, ttl_seconds=self.cfg.execution.duplicate_window_seconds) is not None

    # ------------------------------------------------------------------
    def place_market(self, setup_id: str, symbol: str, direction: Direction,
                     lots: float, sl: float, tp: float,
                     risk_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not sl or sl <= 0 or lots <= 0:
            self.journal.log_security("ERROR",
                                      "refused order without SL or with invalid size",
                                      {"setup_id": setup_id, "symbol": symbol})
            return {"ok": False, "reason": "stop loss is mandatory"}
        key = f"mkt:{setup_id}:{symbol}:{direction.value}"
        if self._duplicate_guard(key):
            return {"ok": False, "duplicate": True, "reason": "duplicate order suppressed"}
        if self._similar_pending(symbol, direction):
            return {"ok": False, "duplicate": True,
                    "reason": "an equivalent pending order already exists"}

        side = direction.value
        comment = f"{self.cfg.execution.order_comment_prefix}:{setup_id[:18]}"
        magic = self.cfg.execution.magic
        result: Optional[OrderResult] = None
        last_err: Optional[str] = None

        for attempt in range(1, self.cfg.execution.order_retry_attempts + 1):
            try:
                result = self.broker.market_order(
                    symbol, side, lots, sl, tp, comment=comment, magic=magic,
                    deviation_points=self.cfg.execution.execution_slippage_points,
                )
                break
            except BrokerError as exc:
                last_err = str(exc)
                self.journal.log_risk("WARN", f"order attempt {attempt} failed: {exc}",
                                      {"setup_id": setup_id, "symbol": symbol})
                if attempt < self.cfg.execution.order_retry_attempts:
                    time.sleep(self.cfg.execution.order_retry_delay_seconds)

        if result is None:
            self.journal.log_trade_event("ERROR", "order failed (no result)", {
                "setup_id": setup_id, "symbol": symbol, "error": last_err,
            })
            return {"ok": False, "reason": f"broker unreachable: {last_err}"}

        if not result.ok or not result.ticket:
            # UNKNOWN state (e.g. timeout after send) must be recorded loudly.
            self.journal.log_trade_event("ERROR", "order rejected or unknown state", {
                "setup_id": setup_id, "symbol": symbol, "retcode": result.retcode,
                "comment": result.comment,
            })
            return {"ok": False, "reason": result.comment or "broker rejected order",
                    "retcode": result.retcode}

        self.store.idempotency_store_result(key, {"ticket": result.ticket})
        self.journal.log_trade_event("INFO", "market order executed", {
            "setup_id": setup_id, "symbol": symbol, "side": side, "lots": lots,
            "sl": sl, "tp": tp, "ticket": result.ticket, "price": result.price,
        })
        return {"ok": True, "ticket": result.ticket, "price": result.price}

    # ------------------------------------------------------------------
    def place_limit(self, setup_id: str, symbol: str, direction: Direction,
                    lots: float, price: float, sl: float, tp: float,
                    expiry_ms: Optional[int] = None) -> Dict[str, Any]:
        if not sl or sl <= 0 or lots <= 0 or price <= 0:
            self.journal.log_security("ERROR",
                                      "refused pending order without SL or invalid price/size",
                                      {"setup_id": setup_id, "symbol": symbol})
            return {"ok": False, "reason": "stop loss is mandatory"}
        key = f"lim:{setup_id}:{symbol}:{direction.value}:{price}"
        if self._duplicate_guard(key):
            return {"ok": False, "duplicate": True, "reason": "duplicate order suppressed"}
        if self._similar_pending(symbol, direction):
            return {"ok": False, "duplicate": True,
                    "reason": "an equivalent pending order already exists"}

        side = direction.value
        comment = f"{self.cfg.execution.order_comment_prefix}:{setup_id[:18]}"
        magic = self.cfg.execution.magic
        try:
            result = self.broker.limit_order(
                symbol, side, lots, price, sl, tp, comment=comment, magic=magic,
                expiration_ms=expiry_ms,
            )
        except BrokerError as exc:
            self.journal.log_trade_event("ERROR", "limit order failed", {
                "setup_id": setup_id, "symbol": symbol, "error": str(exc),
            })
            return {"ok": False, "reason": str(exc)}
        if not result.ok or not result.ticket:
            return {"ok": False, "reason": result.comment or "broker rejected order",
                    "retcode": result.retcode}
        self.store.idempotency_store_result(key, {"ticket": result.ticket})
        self.journal.log_trade_event("INFO", "limit order placed", {
            "setup_id": setup_id, "symbol": symbol, "side": side, "lots": lots,
            "price": price, "sl": sl, "tp": tp, "ticket": result.ticket,
            "expires_at": expiry_ms,
        })
        return {"ok": True, "ticket": result.ticket, "price": price}

    # ------------------------------------------------------------------
    def modify_sl_tp(self, ticket: str, sl: Optional[float],
                     tp: Optional[float]) -> bool:
        try:
            res = self.broker.modify_position(ticket, sl, tp)
        except BrokerError as exc:
            self.journal.log_trade_event("ERROR", f"modify failed for {ticket}: {exc}", {})
            return False
        if res.ok:
            self.journal.log_trade_event("INFO", f"position {ticket} modified",
                                         {"sl": sl, "tp": tp})
        return res.ok

    def close_position(self, ticket: str, lots: Optional[float] = None,
                       reason: str = "") -> Dict[str, Any]:
        try:
            res = self.broker.close_position(ticket, lots)
        except BrokerError as exc:
            self.journal.log_trade_event("ERROR", f"close failed for {ticket}: {exc}",
                                         {"reason": reason})
            return {"ok": False, "reason": str(exc)}
        detail = res.raw or {}
        self.journal.log_trade_event(
            "INFO", f"position {ticket} closed ({reason or 'manual'})",
            {"profit": detail.get("profit"), "lots": detail.get("lots")})
        return {"ok": res.ok, "profit": detail.get("profit", 0.0),
                "lots": detail.get("lots"), "reason": reason}

    def cancel_pending(self, ticket: str, reason: str = "") -> bool:
        try:
            res = self.broker.cancel_order(ticket)
        except BrokerError as exc:
            self.journal.log_trade_event("ERROR", f"cancel failed for {ticket}: {exc}",
                                         {"reason": reason})
            return False
        if res.ok:
            self.journal.log_trade_event("INFO",
                                         f"pending {ticket} cancelled ({reason or 'manual'})", {})
        return res.ok

    # ------------------------------------------------------------------
    def _similar_pending(self, symbol: str, direction: Direction) -> bool:
        try:
            pendings = self.broker.pending_orders()
        except BrokerError:
            return False
        prefix = self.cfg.execution.order_comment_prefix
        for o in pendings:
            if o.symbol == symbol and o.magic == self.cfg.execution.magic:
                wanted = f"{direction.value}_limit"
                if o.kind == wanted:
                    return True
                _ = prefix  # magic match is sufficient
        return False


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}-{utc_now_ms() % 100000}"
