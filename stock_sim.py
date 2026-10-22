"""Staci - rebuild a weekly stock history for the ML risk model (synthetic stand-in for real stock snapshots).

We have 3 years of real daily demand but only ONE stock snapshot. This replays the demand against a
planner who reorders from trailing demand, with suppliers that are sometimes late, and records what
stock looked like each Monday and whether the SKU ran out in the following 30 days.

Output: data/stock_history.csv, one row per SKU per week. With real client stock history, drop in a file
with the same columns and ml_risk.py trains on it unchanged.
"""
import numpy as np
import pandas as pd

from config import DATA_DIR

SEED = 7
PLAN_WINDOW = 28          # the planner estimates demand from the trailing 28 days
REVIEW_EVERY = 7          # the planner reviews stock once a week
PLAN_FACTOR = 1.0         # planner uses trailing demand, so it lags peaks and sometimes runs short
ORDER_COVER = 28          # each order tops stock up to this many extra days of demand
LABEL_DAYS = 30           # stockout label: unmet demand within the next 30 days
SUPPLIER_BIAS = [0, 0, 0, 2, 4, 7]   # typical extra days late; drawn once per supplier (per SKU)


def daily_matrix(demand, items, dates):
    """units per day, one row per SKU (zero-filled)."""
    piv = (demand.groupby(["sku", "date"]).units_shipped.sum().unstack(fill_value=0)
                 .reindex(index=items.sku, columns=dates, fill_value=0))
    return piv.values.astype(float)


def simulate_sku(units, it, rng):
    n = len(units)
    lt, sd, safety = int(it.supplier_lead_time_days), int(it.lead_time_std_days), int(it.safety_stock_days)
    bias = rng.choice(SUPPLIER_BIAS)
    csum = np.concatenate([[0], np.cumsum(units)])
    trailing = lambda d: (csum[d] - csum[max(0, d - PLAN_WINDOW)]) / max(1, min(d, PLAN_WINDOW)) if d > 0 else units[:PLAN_WINDOW].mean()

    stock = units[:PLAN_WINDOW].mean() * (lt + safety + 14)
    orders = []                     # open: [promised_day, actual_day, qty]
    delays = []                     # (actual_day, delay_days) of past receipts
    stock_start, unmet_flag = np.zeros(n), np.zeros(n, dtype=bool)
    open_hist = [None] * n
    for d in range(n):
        # receipts
        for o in [o for o in orders if o[1] <= d]:
            stock += o[2]
            delays.append((o[1], o[1] - o[0]))
            orders.remove(o)
        stock_start[d] = stock
        open_hist[d] = [(o[0], o[2]) for o in orders]
        # weekly review and order
        if d % REVIEW_EVERY == 0:
            rate = trailing(d) * PLAN_FACTOR
            position = stock + sum(o[2] for o in orders)
            if position <= rate * (lt + safety + REVIEW_EVERY):
                qty = max(1.0, rate * (lt + safety + REVIEW_EVERY + ORDER_COVER) - position)
                delay = int(round(rng.normal(bias, sd)))
                orders.append([d + lt, d + max(1, lt + delay), qty])
        # demand
        unmet_flag[d] = units[d] > stock
        stock = max(0.0, stock - units[d])
    return stock_start, unmet_flag, open_hist, delays


def build(items, demand, asof):
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(demand.date.min(), asof, freq="D")
    units = daily_matrix(demand, items, dates)
    rows = []
    for k, it in enumerate(items.itertuples(index=False)):
        stock_start, unmet, open_hist, delays = simulate_sku(units[k], it, rng)
        d_arr = np.array([d for d, _ in delays]) if delays else np.array([])
        delay_arr = np.array([x for _, x in delays]) if delays else np.array([])
        unmet_days = np.where(unmet)[0]
        for d in range(0, len(dates) - LABEL_DAYS, 1):
            if dates[d].dayofweek != 0:
                continue
            past = delay_arr[d_arr <= d][-3:] if len(d_arr) else np.array([])
            inbound = [q for p, q in open_hist[d] if p <= d + LABEL_DAYS]
            etas = [p - d for p, _ in open_hist[d]]
            prev_out = unmet_days[unmet_days < d]
            rows.append(dict(
                sku=it.sku, week_start=dates[d], available_qty=round(stock_start[d]),
                inbound_qty_30d=round(sum(inbound)), days_to_next_arrival=min(etas) if etas else np.nan,
                supplier_delay_mean_last3=past.mean() if len(past) else np.nan,
                supplier_delay_max_last3=past.max() if len(past) else np.nan,
                stockout_days_26w=int(((prev_out >= d - 182)).sum()),
                days_since_stockout=int(d - prev_out[-1]) if len(prev_out) else np.nan,
                stockout_next_30d=int(unmet[d:d + LABEL_DAYS].any())))
    return pd.DataFrame(rows)


def supplier_state(items, demand, asof):
    """Supplier-delay and stockout history per SKU at the as-of date (used to score today's snapshot)."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(demand.date.min(), asof, freq="D")
    units = daily_matrix(demand, items, dates)
    out = {}
    for k, it in enumerate(items.itertuples(index=False)):
        _, unmet, _, delays = simulate_sku(units[k], it, rng)
        d = len(dates) - 1
        past = np.array([x for dd, x in delays if dd <= d][-3:])
        prev_out = np.where(unmet)[0]
        out[it.sku] = dict(supplier_delay_mean_last3=past.mean() if len(past) else np.nan,
                           supplier_delay_max_last3=past.max() if len(past) else np.nan,
                           stockout_days_26w=int((prev_out >= d - 182).sum()),
                           days_since_stockout=int(d - prev_out[-1]) if len(prev_out) else np.nan)
    return out


def main():
    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    demand = pd.read_csv(f"{DATA_DIR}/daily_demand.csv", parse_dates=["date"])
    inv = pd.read_csv(f"{DATA_DIR}/inventory_snapshot.csv", parse_dates=["snapshot_date"])
    hist = build(items, demand, inv.snapshot_date.max())
    hist.to_csv(f"{DATA_DIR}/stock_history.csv", index=False, date_format="%Y-%m-%d")
    print(f"{len(hist)} SKU-weeks | stockout in next {LABEL_DAYS}d: {hist.stockout_next_30d.mean():.1%} "
          f"| SKUs ever out: {hist[hist.stockout_next_30d == 1].sku.nunique()}/{items.sku.nunique()}")


if __name__ == "__main__":
    main()
