"""Staci - order-volume forecast and inventory-shortfall early warning.

Reads the SKU forecast written by forecast.py plus data/, for the model chosen in .env
(FORECAST_MODEL), and writes to outputs/<model>/:
  order_volume_forecast.csv  weekly units and orders (P50 / P90) per client / warehouse / channel
  inventory_risk.csv         one row per SKU: days of cover, stockout dates, risk level, reorder qty
  forecast_vs_actual.csv     daily orders received vs forecast, last 14 days, per client / warehouse
  forecast_accuracy_tracking.csv  past live forecasts (forecast_ledger.csv) scored against actuals as they arrive
  risk_summary.json          count of SKUs per risk level, SKUs at shortfall risk (under 30 days), forecast-vs-actual status

The risk logic does not depend on which model produced the forecast.
"""
import json
import os

import numpy as np
import pandas as pd

from config import DATA_DIR, OUT_DIR, RISK_METHOD

FORECAST_FILE = f"{OUT_DIR}/sku_forecast.csv"
ORDER_FORECAST_FILE = f"{OUT_DIR}/order_forecast.csv"

ORDER_RATIO_DAYS = 56          # recent history used for the orders-per-unit ratio
DEFAULT_ORDERS_PER_UNIT = 0.5  # used when a SKU shipped nothing recently
USAGE_DAYS = 28                # "average daily usage" = mean P50 demand over the next 4 weeks
# Risk levels follow the Staci Customer Portal mockup (Forecasting tab), by days of cover:
HIGH_RISK_DAYS = 14            # under 14 days of stock  = High risk
WATCH_DAYS = 30                # under 30 days           = Watch   (High risk + Watch = "SKUs at shortfall risk")
MONITOR_DAYS = 60              # under 60 days           = Monitor (the mockup's at-risk list stops here)
EXCESS_COVER_DAYS = 120        # 120 days or more        = Possible overstock; in between = Healthy
REVIEW_DAYS = 14               # reorder covers lead time + safety stock + a 2-week review cycle
NO_DEMAND_COVER = 999          # days of cover reported when there is no forecast demand
FVA_DAYS = 14                  # forecast-vs-actual window (mockup: "last 14 days of the period")
FVA_WEEKS = 4                  # fallback only (no fva_forecast.csv): average of the same weekday in the 4 weeks before
FVA_FORECAST_FILE = f"{OUT_DIR}/fva_forecast.csv"
FVA_OK, FVA_WARN = 10, 20      # variance bands: within +/-10% on track, 10-20% watch, over 20% off
DOW_WEEKS = 13                 # weekday profile (share of a week's demand per weekday) from the last 13 weeks
DOW_FLAT_SHARE = 0.2           # blend 20% of a flat profile in, so a weekday with no recent sales isn't set to zero
Z90 = 1.2816                   # standard normal 90th percentile (lead-time variability buffer)
RISK_ORDER = {"High risk": 0, "Watch": 1, "Monitor": 2, "Healthy": 3, "Possible overstock": 4}
AT_RISK = ("High risk", "Watch", "Monitor")   # states that get a suggested reorder qty


def load_inputs():
    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    demand = pd.read_csv(f"{DATA_DIR}/daily_demand.csv", parse_dates=["date"])
    inventory = pd.read_csv(f"{DATA_DIR}/inventory_snapshot.csv", parse_dates=["snapshot_date"])
    asn = pd.read_csv(f"{DATA_DIR}/inbound_asn.csv", parse_dates=["expected_arrival_date"])
    fc = pd.read_csv(FORECAST_FILE, parse_dates=["week_start"])
    return items, demand, inventory, asn, fc


# ============ order-volume forecast by client / warehouse / channel ============
def order_volume(fc, demand, asof, order_fc=None):
    """Convert SKU unit forecasts to order counts using each SKU's recent orders-per-unit ratio (P50).
    With order_fc (forecast.py's direct orders model), orders_p90 comes from that model instead of the sum of
    SKU P90s, which is far too wide (every SKU running high at once)."""
    recent = demand[(demand.date > asof - pd.Timedelta(days=ORDER_RATIO_DAYS)) & (demand.date <= asof)]
    totals = recent.groupby("sku")[["orders", "units_shipped"]].sum()
    orders_per_unit = ((totals.orders / totals.units_shipped.replace(0, np.nan))
                       .fillna(DEFAULT_ORDERS_PER_UNIT).rename("orders_per_unit"))
    ov = fc.merge(orders_per_unit, left_on="sku", right_index=True)
    ov["orders_p50"] = ov.forecast_p50 * ov.orders_per_unit
    ov["orders_p90"] = ov.forecast_p90 * ov.orders_per_unit
    keys = ["client_id", "warehouse", "channel", "week_start"]
    out = (ov.groupby(keys)[["forecast_p50", "forecast_p90", "orders_p50", "orders_p90"]].sum().reset_index()
             .rename(columns={"forecast_p50": "units_p50", "forecast_p90": "units_p90"}))
    if order_fc is not None:
        direct = order_fc[keys + ["orders_p90"]].rename(columns={"orders_p90": "direct_p90"})
        out = out.merge(direct, on=keys, how="left")
        out["orders_p90"] = np.maximum(out.direct_p90.fillna(out.orders_p90), out.orders_p50)
        out = out.drop(columns="direct_p90")
    return out.round(0)


# ============ forecast vs actual (mockup KPI "Forecast vs actual") ============
def fva_status(variance_pct):
    v = abs(variance_pct)
    return "On track" if v <= FVA_OK else "Watch" if v <= FVA_WARN else "Off track"


def forecast_vs_actual(demand, asof):
    """Orders received per day vs the ML forecast for those days (fva_forecast.csv: what the model forecast
    using only data from before the window). Without that file, forecast = mean of the same weekday over the
    4 weeks before. Variance % = (received - forecast) / forecast; status On track <=10%, Watch <=20%, Off track >20%."""
    if os.path.exists(FVA_FORECAST_FILE):
        fc = pd.read_csv(FVA_FORECAST_FILE, parse_dates=["date"])
        received = demand.groupby(["client_id", "warehouse", "date"]).orders.sum().rename("orders_received")
        out = fc.join(received, on=["client_id", "warehouse", "date"]).fillna({"orders_received": 0})
        out = out[(out.date > asof - pd.Timedelta(days=FVA_DAYS)) & (out.orders_forecast > 0)].copy()
        out["orders_received"] = out.orders_received.astype(int)
        out["variance_pct"] = ((out.orders_received - out.orders_forecast) / out.orders_forecast * 100).round(1)
        out["status"] = out.variance_pct.map(fva_status)
        return out[["client_id", "warehouse", "date", "orders_received", "orders_forecast", "variance_pct",
                    "status", "forecast_data_to"]].reset_index(drop=True)
    print(f"WARNING: {FVA_FORECAST_FILE} not found - forecast vs actual uses a 4-week weekday average. "
          "Re-run forecast.py.")
    daily = demand.groupby(["client_id", "warehouse", "date"]).orders.sum().rename("orders_received")
    idx = pd.MultiIndex.from_product([daily.index.levels[0], daily.index.levels[1],
                                      pd.date_range(daily.index.levels[2].min(), asof)],
                                     names=["client_id", "warehouse", "date"])
    daily = daily.reindex(idx, fill_value=0).reset_index()
    daily = daily[daily.groupby(["client_id", "warehouse"]).orders_received.transform("sum") > 0]
    daily = daily.sort_values(["client_id", "warehouse", "date"])
    g = daily.groupby(["client_id", "warehouse"]).orders_received
    daily["orders_forecast"] = sum(g.shift(7 * k) for k in range(1, FVA_WEEKS + 1)) / FVA_WEEKS
    out = daily[daily.date > asof - pd.Timedelta(days=FVA_DAYS)].copy()
    out = out[out.orders_forecast > 0]
    out["variance_pct"] = ((out.orders_received - out.orders_forecast) / out.orders_forecast * 100).round(1)
    out["status"] = out.variance_pct.map(fva_status)
    out["orders_forecast"] = out.orders_forecast.round(1)
    return out.reset_index(drop=True)


# ============ inventory shortfall early warning (daily projection) ============
def first_stockout(available, inbound, daily_demand, days):
    """First day the projected balance goes below zero, or NaT if it never does."""
    balance = available + np.cumsum(inbound - daily_demand)
    below_zero = np.where(balance < 0)[0]
    return days[below_zero[0]] if len(below_zero) else pd.NaT


def classify(cover, stockout_p50, next_asn):
    """Return (risk_level, reason). Level is set by days of cover alone, as in the mockup;
    the reason adds the projected stockout date and whether an inbound ASN arrives in time."""
    if cover >= EXCESS_COVER_DAYS:
        return "Possible overstock", f"{EXCESS_COVER_DAYS}+ days of stock"
    if cover >= MONITOR_DAYS:
        return "Healthy", "60-120 days of stock"
    level = "High risk" if cover < HIGH_RISK_DAYS else "Watch" if cover < WATCH_DAYS else "Monitor"
    if pd.isna(stockout_p50):
        return level, f"{cover:.0f} days of stock; covered by forecast and inbound ASNs"
    if pd.isna(next_asn):
        return level, f"Runs out {stockout_p50.date()}; no inbound ASN"
    if next_asn > stockout_p50:
        return level, (f"Runs out {stockout_p50.date()}, {(next_asn - stockout_p50).days} days before "
                       f"ASN arrives {next_asn.date()}")
    return level, f"Runs out {stockout_p50.date()}; ASN arrives {next_asn.date()}"


def weekday_profile(demand, items, asof):
    """Share of a week's demand that falls on each weekday (Mon=0), per SKU, from recent sales."""
    recent = demand[demand.date > asof - pd.Timedelta(weeks=DOW_WEEKS)]
    by_day = (recent.groupby(["sku", recent.date.dt.dayofweek]).units_shipped.sum().unstack(fill_value=0)
                    .reindex(index=items.sku, columns=range(7), fill_value=0))
    share = by_day.div(by_day.sum(axis=1), axis=0).fillna(1 / 7)
    return (1 - DOW_FLAT_SHARE) * share + DOW_FLAT_SHARE / 7


def supplier_delays(items, demand, asof):
    """Expected extra days late per SKU: the supplier's average delay over its last 3 deliveries.
    Taken from the same (simulated) delivery history the ML model uses; replace with real ASN receipts."""
    from stock_sim import supplier_state
    state = supplier_state(items, demand, asof)
    return {sku: max(0, int(round(np.nan_to_num(s["supplier_delay_mean_last3"])))) for sku, s in state.items()}


def assess_item(item, snapshot, sku_fc, sku_asn, asof, method_version, weekday_share, delay_days):
    available = snapshot.on_hand_qty - snapshot.allocated_qty - snapshot.damaged_hold_qty
    sku_fc = sku_fc.sort_values("week_start")
    # Weekly forecast -> days, following the SKU's weekday pattern. The forecast starts on the Monday after the
    # last FULL week, which can be before the as-of date; those days have already happened (and are already
    # reflected in the stock snapshot), so the projection starts the day after the as-of date.
    all_days = pd.date_range(sku_fc.week_start.iloc[0], periods=7 * len(sku_fc))
    share = weekday_share[all_days.dayofweek]
    future = all_days > asof
    days = all_days[future]
    demand_p50 = (np.repeat(sku_fc.forecast_p50.values, 7) * share)[future]
    demand_p90 = (np.repeat(sku_fc.forecast_p90.values, 7) * share)[future]

    # inbound lands on the ASN date plus the supplier's usual lateness
    open_asn = sku_asn[sku_asn.status != "Received"].sort_values("expected_arrival_date")
    likely_arrival = open_asn.expected_arrival_date + pd.Timedelta(days=delay_days)
    inbound = np.zeros(len(days))
    for when, qty in zip(likely_arrival, open_asn.expected_qty):
        i = (when - days[0]).days
        if 0 <= i < len(days):
            inbound[i] += qty

    stockout_p50 = first_stockout(available, inbound, demand_p50, days)
    stockout_p90 = first_stockout(available, inbound, demand_p90, days)
    usage = demand_p50[:USAGE_DAYS].mean()
    cover = available / usage if usage > 0 else NO_DEMAND_COVER
    lead_time = item.supplier_lead_time_days
    next_asn = open_asn.expected_arrival_date.iloc[0] if len(open_asn) else pd.NaT
    next_likely = likely_arrival.iloc[0] if len(open_asn) else pd.NaT
    asn_id = open_asn.asn_id.iloc[0] if len(open_asn) else None
    # arrives in time = no projected stockout, or the ASN (allowing for supplier delay) lands by the stockout date
    asn_in_time = bool(pd.notna(next_likely) and (pd.isna(stockout_p50) or next_likely <= stockout_p50))
    state, reason = classify(cover, stockout_p50, next_likely)

    # Reorder up to the P90 of demand over lead time + safety stock + review cycle. Weekly P90s are not added up
    # (every week running high at once is far rarer than 1 in 10): the weekly uncertainties are combined as
    # independent, plus a buffer for the supplier's lead time running long.
    target_days = lead_time + item.safety_stock_days + REVIEW_DAYS
    spread = (demand_p90 - demand_p50)[:target_days]
    by_week = np.add.reduceat(spread, np.arange(0, len(spread), 7)) if len(spread) else spread
    lead_time_buffer = Z90 * item.lead_time_std_days * demand_p50[:target_days].mean()
    demand_lt_p90 = demand_p50[:target_days].sum() + np.sqrt((by_week ** 2).sum() + lead_time_buffer ** 2)
    reorder = max(0, int(np.ceil(demand_lt_p90 - available - inbound[:target_days].sum())))

    as_date = lambda d: d.date() if pd.notna(d) else None
    return dict(sku=item.sku, client_id=item.client_id, warehouse=item.warehouse,
                available_qty=int(available), avg_daily_usage_next4w=round(usage, 1),
                days_of_cover=round(cover, 1), lead_time_days=lead_time,
                covered_by_asn=asn_id, next_asn_date=as_date(next_asn), supplier_delay_days=delay_days,
                next_asn_likely_date=as_date(next_likely), asn_arrives_in_time=asn_in_time,
                projected_stockout_p50=as_date(stockout_p50), projected_stockout_p90=as_date(stockout_p90),
                risk_state=state, reason=reason,
                _reorder=reorder, created_at=asof.date(), method_version=method_version)


def inventory_risk(items, inventory, asn, fc, demand, asof, method_version):
    weekday = weekday_profile(demand, items, asof)
    delays = supplier_delays(items, demand, asof)
    rows = [assess_item(item,
                        inventory[inventory.sku == item.sku].iloc[0],
                        fc[fc.sku == item.sku],
                        asn[asn.sku == item.sku],
                        asof, method_version, weekday.loc[item.sku].values, delays[item.sku])
            for item in items.itertuples(index=False)]
    df = pd.DataFrame(rows)
    df["risk_method"] = "rules"
    if RISK_METHOD == "ml":
        df = apply_ml(df, asof)
    df["suggested_reorder_qty"] = np.where(df.risk_state.isin(AT_RISK), df._reorder, 0)
    df = df.drop(columns="_reorder")
    # most urgent first: by risk state, then highest chance of running out (ML) or lowest cover (rules)
    by = ["risk_state", "risk_probability" if "risk_probability" in df else "days_of_cover"]
    return df.sort_values(by, ascending=[True, "risk_probability" not in df],
                          key=lambda col: col.map(RISK_ORDER) if col.name == "risk_state" else col)


def apply_ml(df, asof):
    """Set the risk level from the ML models (ml_risk.py): the stockout probability gives High risk / Watch /
    Monitor / Healthy, and the overstock probability gives Possible overstock (for SKUs not at shortage risk).
    The old rule level is kept in rule_risk_state for comparison."""
    from ml_risk import MODEL_FILE, OVERSTOCK_DAYS, OVERSTOCK_P, level_from_probability, score
    if not os.path.exists(MODEL_FILE):
        print(f"WARNING: {MODEL_FILE} not found - using rule-based risk levels. Run stock_sim.py and ml_risk.py.")
        return df
    ml = score(asof).drop(columns=["ml_risk_state", "cover_days"])
    df = df.merge(ml, on="sku")
    df["rule_risk_state"] = df.risk_state
    ml_level = df.risk_probability.map(level_from_probability)
    if "overstock_probability" in df:
        overstock = (df.overstock_probability >= OVERSTOCK_P) & ml_level.isin(["Monitor", "Healthy"])
    else:                                    # model file from before the overstock model: rule-based overstock
        print("WARNING: no overstock model in the ML model file - overstock stays rule-based. Re-run ml_risk.py.")
        overstock = df.rule_risk_state == "Possible overstock"
    df["risk_state"] = np.where(overstock, "Possible overstock", ml_level)
    shortage = ~overstock
    df.loc[shortage, "reason"] = (df.risk_probability[shortage].map(lambda p: f"{p:.0%} chance of running out within 30 days. ")
                                  + df.reason[shortage])
    if "overstock_probability" in df:
        df.loc[overstock, "reason"] = [f"{p:.0%} chance stock + inbound is not used up within {OVERSTOCK_DAYS} days; "
                                       f"{c:.0f} days of stock" for p, c in
                                       zip(df.overstock_probability[overstock], df.days_of_cover[overstock])]
        df.loc[overstock, "top_drivers"] = df.overstock_drivers[overstock]
        df = df.drop(columns="overstock_drivers")
    df["risk_method"] = "ml"
    return df


# ============ live forecast accuracy (forecast ledger vs actuals) ============
def ledger_accuracy(demand, asof):
    """Score every past live forecast (forecast_ledger.csv, written by forecast.py) on the weeks that have
    fully happened since. This is the model's real-world accuracy, as opposed to the backtest."""
    path = f"{OUT_DIR}/forecast_ledger.csv"
    if not os.path.exists(path):
        return None
    ledger = pd.read_csv(path, parse_dates=["week_start", "created_at"])
    ledger = ledger[ledger.week_start + pd.Timedelta(days=6) <= asof]          # only complete weeks
    if ledger.empty:
        return None
    actual = (demand.assign(week_start=demand.date.dt.to_period("W-SUN").dt.start_time)
                    .groupby(["sku", "week_start"]).units_shipped.sum().rename("actual"))
    scored = ledger.join(actual, on=["sku", "week_start"]).fillna({"actual": 0})
    scored["weeks_ahead"] = (scored.week_start - scored.week_start.groupby(scored.created_at).transform("min")
                             ).dt.days // 7 + 1
    table = (scored.assign(abs_err=(scored.forecast_p50 - scored.actual).abs(),
                           within_p90=scored.actual <= scored.forecast_p90)
                   .groupby(["created_at", "method_version", "weeks_ahead"])
                   .agg(actual=("actual", "sum"), forecast_p50=("forecast_p50", "sum"),
                        abs_err=("abs_err", "sum"), p90_coverage=("within_p90", "mean")).reset_index())
    table["wape"] = (table.abs_err / table.actual.clip(lower=1)).round(3)
    table.drop(columns="abs_err").round({"forecast_p50": 0, "p90_coverage": 3}).to_csv(
        f"{OUT_DIR}/forecast_accuracy_tracking.csv", index=False, date_format="%Y-%m-%d")
    return dict(weeks_scored=int(len(scored)), forecasts_scored=int(scored.created_at.nunique()),
                wape=round(float((scored.forecast_p50 - scored.actual).abs().sum() / max(1, scored.actual.sum())), 3),
                p90_coverage=round(float((scored.actual <= scored.forecast_p90).mean()), 3))


def main():
    if not os.path.exists(FORECAST_FILE):
        raise SystemExit(f"{FORECAST_FILE} not found - run forecast.py first with the same FORECAST_MODEL")
    items, demand, inventory, asn, fc = load_inputs()
    asof = inventory.snapshot_date.max()
    method_version = fc.method_version.iloc[0]

    order_fc = None
    if os.path.exists(ORDER_FORECAST_FILE):
        order_fc = pd.read_csv(ORDER_FORECAST_FILE, parse_dates=["week_start"])
    else:
        print(f"WARNING: {ORDER_FORECAST_FILE} not found - order P90 = sum of SKU P90s. Re-run forecast.py.")
    ov = order_volume(fc, demand, asof, order_fc)
    ov["created_at"], ov["method_version"] = asof.date(), method_version
    ov.to_csv(f"{OUT_DIR}/order_volume_forecast.csv", index=False)

    risk = inventory_risk(items, inventory, asn, fc, demand, asof, method_version)
    risk.to_csv(f"{OUT_DIR}/inventory_risk.csv", index=False)

    summary = dict(as_of=str(asof.date()), method_version=method_version, forecast_file=FORECAST_FILE,
                   risk_method=risk.risk_method.iloc[0],
                   risk_counts=risk.risk_state.value_counts().reindex(list(RISK_ORDER), fill_value=0).to_dict())
    fva = forecast_vs_actual(demand, asof)
    fva.to_csv(f"{OUT_DIR}/forecast_vs_actual.csv", index=False)
    rec, fcst = fva.orders_received.sum(), fva.orders_forecast.sum()
    var = round((rec - fcst) / fcst * 100, 1)
    summary["forecast_vs_actual"] = dict(window_days=FVA_DAYS, orders_received=int(rec),
                                         orders_forecast=round(float(fcst)), variance_pct=var, status=fva_status(var))
    summary["live_forecast_accuracy"] = ledger_accuracy(demand, asof)   # None until a past forecast's weeks complete
    summary["shortfall_risk_skus"] = summary["risk_counts"]["High risk"] + summary["risk_counts"]["Watch"]
    with open(f"{OUT_DIR}/risk_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
