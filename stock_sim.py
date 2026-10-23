"""Staci - rebuild a daily stock history for the ML risk model (synthetic stand-in for real stock snapshots).

We have 3 years of real daily demand but only ONE stock snapshot. This replays the demand against
planners who reorder from trailing demand (several planning policies), with suppliers that are sometimes
late, and records what stock looked like each day and whether the SKU ran out in the following 30 days.

Output: data/stock_history.csv, one row per planning policy per SKU per day. With real client stock history,
drop in a file with the same columns (policy = 0) and ml_risk.py trains on it unchanged.
"""
import numpy as np
import pandas as pd

from config import DATA_DIR

SEED = 7
LABEL_DAYS = 30           # stockout label: unmet demand within the next 30 days
SUPPLIER_BIAS = [0, 0, 0, 2, 4, 7]   # typical extra days late; drawn once per supplier (per SKU)

# The real planner's habits are unknown, so the history is replayed under several planning policies.
# A model trained on one fixed policy would just learn that policy. Policy 0 is the reference: it is the
# one used for supplier / stockout history at the as-of date and for the like-for-like evaluation.
#   plan_window   the planner estimates demand from the trailing N days
#   review_every  the planner reviews stock every N days
#   plan_factor   multiplier on trailing demand (1.0 lags peaks and sometimes runs short)
#   order_cover   each order tops stock up to this many extra days of demand
POLICIES = [dict(plan_window=28, review_every=7, plan_factor=1.0, order_cover=28),
            dict(plan_window=14, review_every=7, plan_factor=1.1, order_cover=21),
            dict(plan_window=56, review_every=14, plan_factor=0.9, order_cover=35),
            dict(plan_window=28, review_every=3, plan_factor=1.0, order_cover=14),
            dict(plan_window=28, review_every=14, plan_factor=1.6, order_cover=90)]   # over-buyer: gives the
                                                                                      # overstock model examples


def daily_matrix(demand, items, dates):
    """units per day, one row per SKU (zero-filled)."""
    piv = (demand.groupby(["sku", "date"]).units_shipped.sum().unstack(fill_value=0)
                 .reindex(index=items.sku, columns=dates, fill_value=0))
    return piv.values.astype(float)


def simulate_sku(units, it, rng, policy=POLICIES[0]):
    n = len(units)
    PLAN_WINDOW, REVIEW_EVERY = policy["plan_window"], policy["review_every"]
    PLAN_FACTOR, ORDER_COVER = policy["plan_factor"], policy["order_cover"]
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
    """One row per policy x SKU x day (every day, not just Mondays: 7x the rows, from the same history)."""
    dates = pd.date_range(demand.date.min(), asof, freq="D")
    units = daily_matrix(demand, items, dates)
    return pd.concat([build_policy(items, units, dates, p, np.random.default_rng(SEED + p))
                      for p in range(len(POLICIES))])


def build_policy(items, units, dates, p, rng):
    rows = []
    for k, it in enumerate(items.itertuples(index=False)):
        stock_start, unmet, open_hist, delays = simulate_sku(units[k], it, rng, POLICIES[p])
        d_arr = np.array([d for d, _ in delays]) if delays else np.array([])
        delay_arr = np.array([x for _, x in delays]) if delays else np.array([])
        unmet_days = np.where(unmet)[0]
        for d in range(0, len(dates) - LABEL_DAYS):
            past = delay_arr[d_arr <= d][-3:] if len(d_arr) else np.array([])
            inbound = [q for p, q in open_hist[d] if p <= d + LABEL_DAYS]
            etas = [p - d for p, _ in open_hist[d]]
            prev_out = unmet_days[unmet_days < d]
            rows.append(dict(
                policy=p, sku=it.sku, date=dates[d], available_qty=round(stock_start[d]),
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
    print(f"{len(hist)} SKU-days over {len(POLICIES)} policies | stockout in next {LABEL_DAYS}d: {hist.stockout_next_30d.mean():.1%} "
          f"| SKUs ever out: {hist[hist.stockout_next_30d == 1].sku.nunique()}/{items.sku.nunique()}")


if __name__ == "__main__":
    main()
