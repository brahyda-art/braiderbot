"""One-time repair: fill in the REAL profit / close price / exit reason of trades that were
closed before the engine learned to read MT5 deal history (they were stored with profit 0).

    python scripts/backfill_closed_profits.py           # dry run - shows what would change
    python scripts/backfill_closed_profits.py --apply   # write to the local DB and the website

Needs the MT5 bridge running (it only READS history; no orders are sent).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv                                   # noqa: E402
load_dotenv(ROOT / ".env")

from app.config import load_config                               # noqa: E402
from app.execution.broker_client import BridgeBroker             # noqa: E402
from app.state_store import StateStore, TradeRecord             # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    token = os.environ.get("BRIDGE_TOKEN", "")
    broker = BridgeBroker(cfg.execution.bridge_url, token, cfg.execution.bridge_timeout_seconds)
    store = StateStore(ROOT / "data" / "braider.db")
    sink = None
    if a.apply:
        from app.services.firebase_sync import build_sink_from_env
        sink = build_sink_from_env()

    rows = store.query("SELECT * FROM trades WHERE status='closed' AND ticket IS NOT NULL")
    fixed = skipped = 0
    for r in rows:
        t = TradeRecord.from_row(r)
        try:
            res = broker.position_result(t.ticket)
        except Exception as exc:
            print(f"  {t.symbol} {t.ticket}: bridge error {exc}")
            skipped += 1
            continue
        if not res.get("found"):
            skipped += 1
            continue
        new_profit = float(res["profit"])
        if abs(new_profit - (t.profit or 0.0)) < 0.005 and t.close_price:
            continue
        print(f"  {t.symbol:8} #{t.ticket}: profit {t.profit:+.2f} -> {new_profit:+.2f}  ({res.get('reason')})")
        fixed += 1
        if a.apply:
            t.profit = new_profit
            t.close_price = res.get("close_price") or t.close_price
            if res.get("closed_at"):
                t.closed_at = int(res["closed_at"])
            t.exit_reason = res.get("reason") or t.exit_reason
            store.upsert_trade(t)
            if sink is not None:
                try:
                    sink.push_trade(t.to_web())
                except Exception as exc:
                    print(f"    website sync failed: {exc}")
    print(f"\n{'Updated' if a.apply else 'Would update'} {fixed} trade(s); {skipped} had no MT5 history.")
    if not a.apply and fixed:
        print("Run again with --apply to save.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
