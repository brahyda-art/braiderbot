"""Trade validation pipeline (spec section 18).

Every candidate setup travels through the exact stages advertised:

    DATA -> HTF CONTEXT -> ZONES -> SR/TRENDLINE -> LIQUIDITY ->
    CONFIRMATION -> ENTRY -> STOP LOSS -> POSITION SIZE -> RISK ->
    SPREAD -> EXPOSURE -> FINAL APPROVAL

The confluence engine has already merged the analysis stages into a scored
setup; this pipeline re-verifies the hard, non-negotiable stages (SL, sizing,
risk, spread, exposure) with FRESH live data and produces a final verdict.
The first failing stage rejects the trade and is journaled with its reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict

from ..config import Config
from ..constants import Direction
from ..risk.position_sizing import SymbolSpec, compute_lot_size, fixed_lot_size, sl_respects_broker_stops
from ..risk.risk_engine import RiskContext, RiskEngine
from ..risk.stop_loss import validate_stop_loss
from ..strategy.confluence import Setup

STAGES = [
    "data_validation", "htf_context", "structure", "supply_demand",
    "support_resistance", "trendline", "liquidity", "pullback_confirmation",
    "entry_validation", "stop_loss_validation", "position_size", "risk_check",
    "spread_check", "exposure_check", "final_approval",
]


@dataclass
class PipelineResult:
    approved: bool
    stage: str
    reason: str
    lots: float = 0.0
    metrics: Dict = field(default_factory=dict)


class TradePipeline:
    def __init__(self, cfg: Config, risk_engine: RiskEngine,
                 get_symbol_spec: Callable[[str], SymbolSpec]):
        self.cfg = cfg
        self.risk = risk_engine
        self.get_spec = get_symbol_spec

    # ------------------------------------------------------------------
    def run(self, setup: Setup, ctx: RiskContext, spread_points: float,
            tick_size: float, spec: SymbolSpec) -> PipelineResult:
        risk_cfg = self.cfg.effective_risk()

        # -- entry validation -------------------------------------------------
        if setup.entry <= 0 or setup.direction is Direction.NONE:
            return PipelineResult(False, "entry_validation", "entry price invalid")

        # -- stop loss validation ----------------------------------------------
        point = spec.point or tick_size or 0.00001
        min_stop_price = risk_cfg.min_stop_points * point if risk_cfg.min_stop_points else 0.0
        max_stop_price = risk_cfg.max_stop_points * point if risk_cfg.max_stop_points else 0.0
        logical = []
        thesis = setup.thesis or {}
        z = thesis.get("zone") or {}
        if z:
            logical += [z.get("lower"), z.get("upper")]
        if thesis.get("invalidation_level"):
            logical.append(thesis["invalidation_level"])
        if thesis.get("sweep"):
            logical.append(thesis["sweep"].get("extreme"))
        logical = [float(v) for v in logical if v]

        sl_check = validate_stop_loss(
            setup.direction.value, setup.entry, setup.sl, logical,
            min_stop_price=min_stop_price, max_stop_price=max_stop_price,
            broker_min_distance=(spec.stops_level_points * point
                                 if spec.stops_level_points and point else 0.0),
        )
        if not sl_check.ok:
            return PipelineResult(False, "stop_loss_validation", sl_check.reason,
                                  metrics={"sl_distance": sl_check.distance})
        if not sl_respects_broker_stops(setup.entry, setup.sl, spec,
                                        setup.direction.value):
            return PipelineResult(False, "stop_loss_validation",
                                  "stop violates broker stops_level")

        # -- position size ---------------------------------------------------------
        equity = ctx.equity or 0.0
        mode = str(getattr(risk_cfg, "lot_sizing_mode", "risk_percent")).lower()
        if mode == "fixed_lot":
            requested_lots = float(getattr(risk_cfg, "fixed_lot_size", 0.01))
            sizing = fixed_lot_size(requested_lots, spec)
        elif mode == "per_symbol":
            table = getattr(risk_cfg, "fixed_lot_sizes", {}) or {}
            requested_lots = float(table.get(setup.symbol, getattr(risk_cfg, "fixed_lot_size", 0.01)))
            sizing = fixed_lot_size(requested_lots, spec)
        else:
            sizing = compute_lot_size(equity, risk_cfg.per_trade_risk_pct,
                                      setup.entry, setup.sl, spec)
        if not sizing.ok:
            return PipelineResult(False, "position_size", sizing.reason,
                                  metrics={"risk_amount": sizing.risk_amount, "lot_sizing_mode": mode})
        lots = sizing.lots

        # -- risk check ---------------------------------------------------------------
        risk_decision = self.risk.pre_trade_check(ctx, setup.symbol, lots, setup.id)
        if not risk_decision.allowed:
            return PipelineResult(False, risk_decision.stage, risk_decision.reason,
                                  metrics=risk_decision.metrics)

        # -- spread ------------------------------------------------------------------
        if spread_points is None:
            return PipelineResult(False, "spread_check", "spread unavailable")
        if spread_points > risk_cfg.max_spread_points:
            return PipelineResult(False, "spread_check",
                                  f"spread {spread_points} > {risk_cfg.max_spread_points}")

        # -- final approval -------------------------------------------------------------
        return PipelineResult(True, "final_approval", "approved", lots=lots,
                              metrics={
                                  "risk_amount": round(sizing.risk_amount, 2),
                                  "actual_risk_amount": round(sizing.actual_risk_amount, 2),
                                  "loss_per_lot": sizing.loss_per_lot,
                                  "sl_distance": sl_check.distance,
                                  "score": setup.score,
                                  "spread_points": spread_points,
                              })
