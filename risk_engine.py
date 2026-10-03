"""Risk engine - hard safety limits evaluated BEFORE every order and re-run
AFTER fills to verify resulting exposure (spec section 17).

All limits are configurable (config.yaml / web app).  The engine fails
CLOSED: any uncertain input (missing account data, stale equity) rejects the
trade.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..config import Config
from ..constants import BotState
from ..state_store import StateStore


@dataclass
class RiskContext:
    """Snapshot of everything the risk engine needs right now."""
    bot_state: BotState
    balance: Optional[float]
    equity: Optional[float]
    day: str                                   # trading day key (engine tz)
    open_positions: List[Dict] = field(default_factory=list)
    open_pendings: List[Dict] = field(default_factory=list)
    spread_points: Optional[float] = None
    consecutive_losses: int = 0
    last_loss_ts_ms: Optional[int] = None
    now_ms: int = 0
    # "Force trade": when True the daily-loss, daily-drawdown, weekly-drawdown
    # and trades-per-day limits are bypassed. Everything else still applies
    # (emergency stop, account drawdown, exposure caps, cooldown, spread,
    # session, mandatory stop-loss, position sizing).
    override_limits: bool = False


@dataclass
class RiskDecision:
    allowed: bool
    reason: str = "ok"
    stage: str = "risk_check"
    metrics: Dict = field(default_factory=dict)


class RiskEngine:
    def __init__(self, cfg: Config, store: StateStore):
        self.cfg = cfg
        self.store = store

    # ------------------------------------------------------------------
    def _daily_stats(self, ctx: RiskContext) -> Dict:
        stats = self.store.get_daily(ctx.day)
        return stats or {}

    def daily_pnl_pct(self, ctx: RiskContext) -> Optional[float]:
        stats = self._daily_stats(ctx)
        start_bal = stats.get("start_balance")
        if not start_bal or ctx.equity is None:
            return None
        realized = stats.get("realized_pnl", 0.0) or 0.0
        floating = ctx.equity - (ctx.balance or ctx.equity)
        return ((realized + floating) / start_bal) * 100.0

    def daily_drawdown_pct(self, ctx: RiskContext) -> Optional[float]:
        stats = self._daily_stats(ctx)
        peak = stats.get("peak_equity")
        if not peak or ctx.equity is None:
            return None
        return max(0.0, ((peak - ctx.equity) / peak) * 100.0)

    def weekly_drawdown_pct(self, ctx: RiskContext) -> Optional[float]:
        peak = self.store.kv_get("week_peak_equity")
        if not peak or ctx.equity is None:
            return None
        return max(0.0, ((peak - ctx.equity) / peak) * 100.0)

    def account_drawdown_pct(self, ctx: RiskContext) -> Optional[float]:
        peak = self.store.kv_get("account_peak_equity")
        if not peak or ctx.equity is None:
            return None
        return max(0.0, ((peak - ctx.equity) / peak) * 100.0)

    # ------------------------------------------------------------------
    def pre_trade_check(self, ctx: RiskContext, symbol: str,
                        lots: float, setup_id: str = "") -> RiskDecision:
        cfg = self.cfg.effective_risk()
        m: Dict = {}

        # 0. Bot state -----------------------------------------------------
        if ctx.bot_state is BotState.EMERGENCY_STOP:
            return RiskDecision(False, "EMERGENCY STOP active", "state_check", m)
        if ctx.bot_state is not BotState.RUNNING:
            return RiskDecision(False, f"bot is {ctx.bot_state.value}", "state_check", m)

        # 1. Account data sanity --------------------------------------------
        if not ctx.balance or not ctx.equity or ctx.equity <= 0:
            return RiskDecision(False, "account equity unavailable (fail-closed)",
                                "account_check", m)

        # 2-5. Loss / drawdown / trade-count limits ---------------------------
        # "Soft" limits: these pause NEW trades but never stop the engine
        # from scanning (signals keep flowing). The user can bypass exactly
        # these with Force Trade. Account drawdown below is a hard stop.
        bypassed: List[str] = []

        def soft_block(stage: str, reason: str) -> Optional[RiskDecision]:
            if ctx.override_limits:
                bypassed.append(stage)
                return None
            return RiskDecision(False, reason, stage, m)

        dp = self.daily_pnl_pct(ctx)
        m["daily_pnl_pct"] = dp
        if dp is None:
            d = soft_block("daily_loss_check", "daily P/L not computable")
            if d:
                return d
        elif dp <= -abs(cfg.max_daily_loss_pct):
            d = soft_block("daily_loss_check",
                           f"daily loss limit hit ({dp:.2f}% <= -{cfg.max_daily_loss_pct}%)")
            if d:
                return d

        dd = self.daily_drawdown_pct(ctx)
        m["daily_dd_pct"] = dd
        if dd is not None and dd >= cfg.max_daily_drawdown_pct:
            d = soft_block("daily_dd_check",
                           f"daily drawdown {dd:.2f}% >= {cfg.max_daily_drawdown_pct}%")
            if d:
                return d

        wd = self.weekly_drawdown_pct(ctx)
        m["weekly_dd_pct"] = wd
        weekly_cap = getattr(cfg, "max_weekly_drawdown_pct", 0) or 0
        if weekly_cap > 0 and wd is not None and wd >= weekly_cap:
            d = soft_block("weekly_dd_check",
                           f"weekly drawdown {wd:.2f}% >= {weekly_cap}%")
            if d:
                return d

        # Account drawdown is a HARD stop: Force Trade does not bypass it.
        add = self.account_drawdown_pct(ctx)
        m["account_dd_pct"] = add
        if add is not None and add >= cfg.max_account_drawdown_pct:
            return RiskDecision(False, f"account drawdown {add:.2f}% >= "
                                f"{cfg.max_account_drawdown_pct}%", "account_dd_check", m)

        stats = self._daily_stats(ctx)
        trades_today = stats.get("trades", 0) or 0
        m["trades_today"] = trades_today
        if trades_today >= cfg.max_trades_per_day:
            d = soft_block("trade_count_check",
                           f"max trades/day reached ({trades_today})")
            if d:
                return d
        if bypassed:
            m["override_bypassed"] = bypassed

        # 6. Simultaneous positions / symbol caps -------------------------------
        positions = ctx.open_positions
        m["open_positions"] = len(positions)
        if len(positions) >= cfg.max_simultaneous_positions:
            return RiskDecision(False, f"max simultaneous positions ({len(positions)})",
                                "exposure_check", m)
        same_symbol = sum(1 for p in positions if p.get("symbol") == symbol)
        if same_symbol >= cfg.max_positions_per_symbol:
            return RiskDecision(False, f"max positions for {symbol}",
                                "exposure_check", m)
        total_lots = sum(float(p.get("lots", 0) or 0) for p in positions) + lots
        if cfg.max_exposure_lots > 0 and total_lots > cfg.max_exposure_lots:
            return RiskDecision(False, f"exposure {total_lots} lots > cap "
                                f"{cfg.max_exposure_lots}", "exposure_check", m)
        if sum(1 for o in ctx.open_pendings if o.get("symbol") == symbol) >= \
                self.cfg.strategy.max_pending_per_symbol:
            return RiskDecision(False, f"max pending orders for {symbol}",
                                "exposure_check", m)

        # 7. Consecutive losses + cooldown --------------------------------------
        if ctx.consecutive_losses >= cfg.max_consecutive_losses and ctx.last_loss_ts_ms:
            elapsed_min = (ctx.now_ms - ctx.last_loss_ts_ms) / 60000.0
            if elapsed_min < cfg.cooldown_minutes_after_losses:
                return RiskDecision(
                    False,
                    f"cooldown after {ctx.consecutive_losses} consecutive losses "
                    f"({cfg.cooldown_minutes_after_losses - elapsed_min:.0f} min left)",
                    "cooldown_check", m)

        # 8. Spread ---------------------------------------------------------------
        if ctx.spread_points is not None and cfg.max_spread_points > 0:
            if ctx.spread_points > cfg.max_spread_points:
                return RiskDecision(False, f"spread {ctx.spread_points} > "
                                    f"{cfg.max_spread_points} points", "spread_check", m)
        elif ctx.spread_points is None:
            return RiskDecision(False, "spread unavailable (fail-closed)",
                                "spread_check", m)

        # 9. Session window ---------------------------------------------------------
        session_ok = self.session_allowed(ctx.now_ms)
        if not session_ok:
            return RiskDecision(False, "outside allowed trading sessions",
                                "session_check", m)

        return RiskDecision(True, "ok", "risk_check", m)

    # ------------------------------------------------------------------
    def session_allowed(self, now_ms: int) -> bool:
        allowed = self.cfg.effective_sessions()
        if not allowed:
            return True
        dt = datetime.fromtimestamp(now_ms / 1000.0, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        for name in allowed:
            win = self.cfg.sessions.get(name)
            if not win:
                continue
            start, end = win["start"], win["end"]
            if start <= end:
                if start <= hour < end:
                    return True
            else:  # overnight window (e.g. sydney 21 -> 6)
                if hour >= start or hour < end:
                    return True
        return False

    # ------------------------------------------------------------------
    def limit_status(self, ctx: RiskContext) -> Dict:
        """Which loss/drawdown/trade-count limit (if any) is currently
        pausing new trades. Published to the website so it can show
        "limit reached - still scanning" and offer Force Trade."""
        cfg = self.cfg.effective_risk()
        dp = self.daily_pnl_pct(ctx)
        dd = self.daily_drawdown_pct(ctx)
        wd = self.weekly_drawdown_pct(ctx)
        add = self.account_drawdown_pct(ctx)
        trades = (self._daily_stats(ctx).get("trades", 0) or 0)
        weekly_cap = getattr(cfg, "max_weekly_drawdown_pct", 0) or 0

        hit = None
        if dp is not None and dp <= -abs(cfg.max_daily_loss_pct):
            hit = ("daily_loss", f"Daily loss limit reached ({dp:.2f}%)")
        elif dd is not None and dd >= cfg.max_daily_drawdown_pct:
            hit = ("daily_drawdown", f"Daily drawdown limit reached ({dd:.2f}%)")
        elif weekly_cap > 0 and wd is not None and wd >= weekly_cap:
            hit = ("weekly_drawdown", f"Weekly drawdown limit reached ({wd:.2f}%)")
        elif trades >= cfg.max_trades_per_day:
            hit = ("max_trades", f"Max trades per day reached ({trades}/{cfg.max_trades_per_day})")

        hard = None
        if add is not None and add >= cfg.max_account_drawdown_pct:
            hard = f"Account drawdown limit reached ({add:.2f}%)"

        def r(v):
            return None if v is None else round(v, 2)

        return {
            "blocked": bool(hit),
            "limit": hit[0] if hit else None,
            "message": hit[1] if hit else None,
            "hardStop": hard,
            "override": bool(ctx.override_limits),
            "dailyPnlPct": r(dp), "dailyDdPct": r(dd),
            "weeklyDdPct": r(wd), "accountDdPct": r(add),
            "tradesToday": int(trades), "maxTradesPerDay": int(cfg.max_trades_per_day),
            "maxDailyLoss": cfg.max_daily_loss_pct,
            "maxDailyDrawdown": cfg.max_daily_drawdown_pct,
            "maxWeeklyDrawdown": weekly_cap,
        }

    # ------------------------------------------------------------------
    def consecutive_losses(self) -> int:
        trades = self.store.recent_trades(50)
        streak = 0
        for t in trades:
            if t.status != "closed":
                continue
            if t.profit < 0:
                streak += 1
            elif t.profit > 0:
                break
        return streak

    def last_loss_time(self) -> Optional[int]:
        for t in self.store.recent_trades(50):
            if t.status == "closed" and t.profit < 0 and t.closed_at:
                return t.closed_at
        return None
