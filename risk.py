"""Staci - order-volume forecast and inventory-shortfall early warning.

Reads the SKU forecast written by forecast.py plus data/, for the model chosen in .env
(FORECAST_MODEL), and writes to outputs/<model>/:
  order_volume_forecast.csv  weekly units and orders (P50 / P90) per client / warehouse / channel
  inventory_risk.csv         one row per SKU: days of cover, stockout dates, RED/AMBER/GREEN/BLUE, reorder qty
  risk_summary.json          count of SKUs in each risk state

The risk logic does not depend on which model produced the forecast.
"""
import json
import os

import numpy as np
import pandas as pd

from config import DATA_DIR, OUT_DIR

FORECAST_FILE = f"{OUT_DIR}/sku_forecast.csv"

ORDER_RATIO_DAYS = 56          # recent history used for the orders-per-unit ratio
DEFAULT_ORDERS_PER_UNIT = 0.5  # used when a SKU shipped nothing recently
USAGE_DAYS = 28                # "average daily usage" = mean P50 demand over the next 4 weeks
EXCESS_COVER_DAYS = 90         # more cover than this = BLUE (excess)
REVIEW_DAYS = 14               # reorder covers lead time + safety stock + a 2-week review cycle
NO_DEMAND_COVER = 999          # days of cover reported when there is no forecast demand
RISK_ORDER = {"RED": 0, "AMBER": 1, "GREEN": 2, "BLUE": 3}


def load_inputs():
    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    demand = pd.read_csv(f"{DATA_DIR}/daily_demand.csv", parse_dates=["date"])
    inventory = pd.read_csv(f"{DATA_DIR}/inventory_snapshot.csv", parse_dates=["snapshot_date"])
    asn = pd.read_csv(f"{DATA_DIR}/inbound_asn.csv", parse_dates=["expected_arrival_date"])
    fc = pd.read_csv(FORECAST_FILE, parse_dates=["week_start"])
    return items, demand, inventory, asn, fc


# ============ order-volume forecast by client / warehouse / channel ============
def order_volume(fc, demand, asof):
    """Convert SKU unit forecasts to order counts using each SKU's recent orders-per-unit ratio."""
    recent = demand[(demand.date > asof - pd.Timedelta(days=ORDER_RATIO_DAYS)) & (demand.date <= asof)]
    totals = recent.groupby("sku")[["orders", "units_shipped"]].sum()
    orders_per_unit = ((totals.orders / totals.units_shipped.replace(0, np.nan))
                       .fillna(DEFAULT_ORDERS_PER_UNIT).rename("orders_per_unit"))
    ov = fc.merge(orders_per_unit, left_on="sku", right_index=True)
    ov["orders_p50"] = ov.forecast_p50 * ov.orders_per_unit
    ov["orders_p90"] = ov.forecast_p90 * ov.orders_per_unit
    return (ov.groupby(["client_id", "warehouse", "channel", "week_start"])
              [["forecast_p50", "forecast_p90", "orders_p50", "orders_p90"]]
              .sum().round(0).reset_index()
              .rename(columns={"forecast_p50": "units_p50", "forecast_p90": "units_p90"}))


# ============ inventory shortfall early warning (daily projection) ============
def first_stockout(available, inbound, daily_demand, days):
    """First day the projected balance goes below zero, or NaT if it never does."""
    balance = available + np.cumsum(inbound - daily_demand)
    below_zero = np.where(balance < 0)[0]
    return days[below_zero[0]] if len(below_zero) else pd.NaT


def classify(item, stockout_p50, stockout_p90, cover, next_asn, need_by):
    """Return (risk_state, reason). Checked in order: RED, AMBER, BLUE, GREEN."""
    if pd.notna(stockout_p50) and stockout_p50 <= need_by:
        if pd.notna(next_asn) and next_asn > stockout_p50:
            gap = (next_asn - stockout_p50).days
            return "RED", (f"Runs out {stockout_p50.date()}, {gap} days before ASN arrives "
                           f"{next_asn.date()} - expedite inbound")
        return "RED", f"Runs out {stockout_p50.date()}, before a new order placed today could arrive"
    if (pd.notna(stockout_p50)
            or cover < item.supplier_lead_time_days + item.safety_stock_days
            or (pd.notna(stockout_p90) and stockout_p90 <= need_by)):
        return "AMBER", "Cover below lead time + safety stock (or short in a busy-week scenario) - reorder now"
    if cover > EXCESS_COVER_DAYS:
        return "BLUE", f"Excess stock (>{EXCESS_COVER_DAYS} days cover)"
    return "GREEN", "Covered through lead time"


def assess_item(item, snapshot, sku_fc, sku_asn, asof, method_version):
    available = snapshot.on_hand_qty - snapshot.allocated_qty - snapshot.damaged_hold_qty
    sku_fc = sku_fc.sort_values("week_start")
    demand_p50 = np.repeat(sku_fc.forecast_p50.values / 7, 7)   # weekly forecast spread evenly over 7 days
    demand_p90 = np.repeat(sku_fc.forecast_p90.values / 7, 7)
    days = pd.date_range(asof + pd.Timedelta(days=1), periods=len(demand_p50))

    inbound = np.zeros(len(days))
    for _, shipment in sku_asn.iterrows():
        i = (shipment.expected_arrival_date - days[0]).days
        if 0 <= i < len(days):
            inbound[i] += shipment.expected_qty

    stockout_p50 = first_stockout(available, inbound, demand_p50, days)
    stockout_p90 = first_stockout(available, inbound, demand_p90, days)
    usage = demand_p50[:USAGE_DAYS].mean()
    cover = available / usage if usage > 0 else NO_DEMAND_COVER
    lead_time = item.supplier_lead_time_days
    next_asn = sku_asn.expected_arrival_date.min() if len(sku_asn) else pd.NaT
    need_by = asof + pd.Timedelta(days=lead_time + item.lead_time_std_days)   # earliest a new PO could land
    state, reason = classify(item, stockout_p50, stockout_p90, cover, next_asn, need_by)

    target_days = lead_time + item.safety_stock_days + REVIEW_DAYS
    reorder = max(0, int(np.ceil(demand_p90[:target_days].sum() - available - inbound[:target_days].sum())))

    as_date = lambda d: d.date() if pd.notna(d) else None
    return dict(sku=item.sku, client_id=item.client_id, warehouse=item.warehouse,
                available_qty=int(available), avg_daily_usage_next4w=round(usage, 1),
                days_of_cover=round(cover, 1), lead_time_days=lead_time,
                next_asn_date=as_date(next_asn),
                projected_stockout_p50=as_date(stockout_p50), projected_stockout_p90=as_date(stockout_p90),
                risk_state=state, reason=reason,
                suggested_reorder_qty=reorder if state in ("RED", "AMBER") else 0,
                created_at=asof.date(), method_version=method_version)


def inventory_risk(items, inventory, asn, fc, asof, method_version):
    rows = [assess_item(item,
                        inventory[inventory.sku == item.sku].iloc[0],
                        fc[fc.sku == item.sku],
                        asn[asn.sku == item.sku],
                        asof, method_version)
            for item in items.itertuples(index=False)]
    # most urgent first: by risk state, then lowest cover
    return pd.DataFrame(rows).sort_values(
        ["risk_state", "days_of_cover"],
        key=lambda col: col.map(RISK_ORDER) if col.name == "risk_state" else col)


def main():
    if not os.path.exists(FORECAST_FILE):
        raise SystemExit(f"{FORECAST_FILE} not found - run forecast.py first with the same FORECAST_MODEL")
    items, demand, inventory, asn, fc = load_inputs()
    asof = inventory.snapshot_date.max()
    method_version = fc.method_version.iloc[0]

    ov = order_volume(fc, demand, asof)
    ov["created_at"], ov["method_version"] = asof.date(), method_version
    ov.to_csv(f"{OUT_DIR}/order_volume_forecast.csv", index=False)

    risk = inventory_risk(items, inventory, asn, fc, asof, method_version)
    risk.to_csv(f"{OUT_DIR}/inventory_risk.csv", index=False)

    summary = dict(as_of=str(asof.date()), method_version=method_version, forecast_file=FORECAST_FILE,
                   risk_counts=risk.risk_state.value_counts().to_dict())
    with open(f"{OUT_DIR}/risk_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
