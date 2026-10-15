"""Staci - weekly SKU demand forecast (P50 + P90) with the model chosen in .env.

FORECAST_MODEL picks the model (see config.py). Reads data/ and writes to outputs/<model>/:
  sku_forecast.csv            weekly P50 / P90 per SKU for the next 13 weeks (input to risk.py)
  backtest_accuracy.csv       per-SKU forecast vs actual on two historical windows
  feature_importance.csv      which features drive the P50 model (tree models only)
  summary.json                overall backtest scores for both deliverables

The backtest scores three things on two past windows:
  sku_demand      units per SKU per week - the engine both deliverables are built on
  order_volume    orders per client / warehouse / channel / week (deliverable 1: Order Volume)
  shortfall_risk  demand during each SKU's lead time (deliverable 2: Shortfall Risk, proxy -
                  a true alert backtest needs historical stock snapshots, which we don't have)
"""
import importlib
import json
import os

import numpy as np
import pandas as pd

from config import DATA_DIR, H, MODEL, MODEL_MODULE, OUT_DIR
from risk import order_volume

model = importlib.import_module(MODEL_MODULE)

ITEM_COLS = ["sku", "client_id", "warehouse", "channel", "category"]
CATS = ["client_id", "warehouse", "channel", "category"]
LAGS = [1, 2, 4, 13, 52]
ROLL_WINDOWS = [4, 8, 13]
FEATURES = ([f"lag_{L}" for L in LAGS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
            + ["roll_std_8", "zero_share_13", "lag_52_ratio", "week_of_year", "month", "event_uplift"]
            + CATS)

# Backtest windows: name -> how many weeks back the test window starts
BACKTEST_WINDOWS = {"last_13_weeks": H, "peak_season_last_year": 52}


# ============ STEP 1: load data ============
def load_data():
    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    demand = pd.read_csv(f"{DATA_DIR}/daily_demand.csv", parse_dates=["date"])
    inventory = pd.read_csv(f"{DATA_DIR}/inventory_snapshot.csv", parse_dates=["snapshot_date"])
    events = pd.read_csv(f"{DATA_DIR}/events.csv", parse_dates=["start_date", "end_date"])
    return items, demand, events, inventory.snapshot_date.max()


# ============ STEP 2: daily -> weekly, one row per SKU x week (zero-filled) ============
def to_weekly(demand, items, asof):
    demand = demand.assign(week=demand.date.dt.to_period("W-SUN").dt.start_time)
    weeks = pd.period_range(demand.date.min(), asof, freq="W-SUN").start_time
    if (asof - weeks[-1]).days < 6:      # drop the incomplete current week
        weeks = weeks[:-1]
    grid = pd.MultiIndex.from_product([items.sku, weeks], names=["sku", "week"])
    weekly = (demand.groupby(["sku", "week"]).units_shipped.sum()
                    .reindex(grid, fill_value=0).rename("units").reset_index())
    return weekly.merge(items[ITEM_COLS], on="sku"), weeks


# ============ STEP 3: event uplift for every client x week (past AND future), computed once ============
def build_event_table(events, clients, weeks):
    all_weeks = pd.date_range(weeks[0], weeks[-1] + pd.Timedelta(weeks=H), freq="7D")
    rows = []
    for client in clients:
        ev = events[events.client_id.isin([client, "ALL"])]
        for wk in all_weeks:
            hit = ev[(ev.start_date <= wk + pd.Timedelta(days=6)) & (ev.end_date >= wk)]
            rows.append((client, wk, float(hit.uplift.max()) if len(hit) else 1.0))   # 1.0 = no event
    return pd.DataFrame(rows, columns=["client_id", "week", "event_uplift"])


# ============ STEP 4: features (only use data BEFORE each week -> no leakage) ============
def add_features(df, event_table):
    df = df.sort_values(["sku", "week"]).copy()
    units = df.groupby("sku").units
    for L in LAGS:
        df[f"lag_{L}"] = units.shift(L)

    past = units.shift(1)                # shift(1) = exclude the current week
    past_by_sku = past.groupby(df.sku)
    zero_by_sku = (past == 0).astype(float).groupby(df.sku)
    def rolling(grouped, window, stat, min_periods=1):
        return getattr(grouped.rolling(window, min_periods=min_periods), stat)().reset_index(level=0, drop=True)
    for w in ROLL_WINDOWS:
        df[f"roll_mean_{w}"] = rolling(past_by_sku, w, "mean")
    df["roll_std_8"] = rolling(past_by_sku, 8, "std", min_periods=2)
    df["zero_share_13"] = rolling(zero_by_sku, 13, "mean")

    df["lag_52_ratio"] = df.lag_52 / (df.roll_mean_8 + 1)   # how last year compares with now
    df["week_of_year"] = df.week.dt.isocalendar().week.astype(int)
    df["month"] = df.week.dt.month
    df = df.merge(event_table, on=["client_id", "week"], how="left")
    df[CATS] = df[CATS].astype("category")
    return df


# ============ STEP 5: train two models - P50 and P90 (see models/) ============
def train(history, event_table):
    tr = add_features(history, event_table).dropna(subset=["lag_52"])   # need one full year of history
    return model.train(tr[FEATURES], tr.units)


# ============ STEP 6: recursive 13-week forecast ============
def forecast(history, items, event_table, m50, m90):
    """Predict week 1, append it as if it were actual, predict week 2, ... (P50 path feeds the lags)."""
    hist = history[["sku", "week", "units"] + CATS].copy()
    last = hist.week.max()
    out = []
    for step in range(1, H + 1):
        wk = last + pd.Timedelta(weeks=step)
        new = items[ITEM_COLS].assign(week=wk, units=np.nan)
        feats = add_features(pd.concat([hist, new]), event_table).query("week == @wk")
        p50 = np.clip(m50.predict(feats[FEATURES]), 0, None)
        p90 = np.maximum(np.clip(m90.predict(feats[FEATURES]), 0, None), p50)   # P90 never below P50
        out.append(pd.DataFrame({"sku": feats.sku.values, "week_start": wk,
                                 "forecast_p50": p50.round(1), "forecast_p90": p90.round(1)}))
        hist = pd.concat([hist, new.assign(units=p50)])   # feed prediction back as "history"
    return pd.concat(out)


def predict(history, items, event_table):
    """13-week forecast from `history`. Returns (forecast, P50 model or None).

    Tree models (lgbm, xgboost) are trained on our features first. Pre-trained models
    (chronos) have no `train` - they forecast straight from the weekly history.
    """
    if hasattr(model, "train"):
        m50, m90 = train(history, event_table)
        return forecast(history, items, event_table, m50, m90), m50
    return model.forecast(history, event_table, H), None


# ============ STEP 7: backtest - train on the past, compare with what actually happened ============
def wape(forecast_col, actual_col):
    """Weighted absolute percentage error: total |error| / total actual."""
    return np.abs(forecast_col - actual_col).sum() / actual_col.sum()


def score(p50, p90, actual, naive):
    """WAPE of P50 and of the naive benchmark, plus how often actual stayed within P90."""
    return dict(model=round(wape(p50, actual), 3), naive=round(wape(naive, actual), 3),
                p90_coverage=round((actual <= p90).mean(), 3))


def order_volume_backtest(fc, demand, items, cut):
    """Deliverable 1 - forecast orders per client / warehouse / channel / week vs actual orders.
    Units are converted to orders the same way as the live run (risk.order_volume)."""
    keys = ["client_id", "warehouse", "channel", "week_start"]
    units = fc[["sku", "week_start", "forecast_p50", "forecast_p90", "naive"]].merge(
        items[["sku", "client_id", "warehouse", "channel"]], on="sku")
    asof = cut - pd.Timedelta(days=1)    # only orders seen before the cut set the orders-per-unit ratio
    model_ov = order_volume(units, demand, asof)
    naive_ov = order_volume(units.assign(forecast_p50=units.naive, forecast_p90=units.naive), demand, asof)

    test = demand[demand.date >= cut].assign(week_start=lambda d: d.date.dt.to_period("W-SUN").dt.start_time)
    actual = (test[test.week_start.isin(fc.week_start.unique())]
              .groupby(keys).orders.sum().rename("actual").reset_index())
    ov = (model_ov.merge(naive_ov[keys + ["orders_p50"]].rename(columns={"orders_p50": "naive"}), on=keys)
                  .merge(actual, on=keys, how="left").fillna({"actual": 0}))
    return score(ov.orders_p50, ov.orders_p90, ov.actual, ov.naive)


def shortfall_backtest(fc, demand, items, cut):
    """Deliverable 2 (proxy) - total demand over each SKU's lead time (+ variability), forecast vs actual.
    This is the demand the stock on hand must cover before a new order can land, so it is the part
    of the shortfall alert the model controls. P90 coverage = share of SKUs that would NOT have been
    caught short if stock had been planned to the P90 figure."""
    rows = []
    for item in items.itertuples(index=False):
        days = int(item.supplier_lead_time_days + item.lead_time_std_days)
        f = fc[fc.sku == item.sku].sort_values("week_start")
        daily = lambda col: np.repeat(f[col].values / 7, 7)[:days].sum()   # weekly forecast spread over 7 days
        sold = demand[(demand.sku == item.sku) & (demand.date >= cut)
                      & (demand.date < cut + pd.Timedelta(days=days))].units_shipped.sum()
        rows.append((daily("forecast_p50"), daily("forecast_p90"), sold, daily("naive")))
    lt = pd.DataFrame(rows, columns=["p50", "p90", "actual", "naive"])
    return score(lt.p50, lt.p90, lt.actual, lt.naive)


def backtest(weekly, weeks, items, event_table, demand, weeks_back):
    cut = weeks[-weeks_back]             # first week of the test window
    history = weekly[weekly.week < cut]
    actual = weekly[(weekly.week >= cut) & (weekly.week < cut + pd.Timedelta(weeks=H))]
    fc, _ = predict(history, items, event_table)
    fc = fc.merge(actual[["sku", "week", "units"]], left_on=["sku", "week_start"], right_on=["sku", "week"])

    # naive benchmark: average of the last 4 weeks before the cut, repeated
    naive = history[history.week >= cut - pd.Timedelta(weeks=4)].groupby("sku").units.mean().rename("naive")
    fc = fc.merge(naive, on="sku")

    overall = {"sku_demand": score(fc.forecast_p50, fc.forecast_p90, fc.units, fc.naive),
               "order_volume": order_volume_backtest(fc, demand, items, cut),
               "shortfall_risk": shortfall_backtest(fc, demand, items, cut)}

    fc["err_model"] = (fc.forecast_p50 - fc.units).abs()
    fc["err_naive"] = (fc.naive - fc.units).abs()
    per_sku = fc.groupby("sku").agg(actual_13w=("units", "sum"), model_13w=("forecast_p50", "sum"),
                                    naive_13w=("naive", "sum"), err_model=("err_model", "sum"),
                                    err_naive=("err_naive", "sum")).reset_index()
    actual_total = per_sku.actual_13w.clip(lower=1)
    per_sku["model_wape"] = (per_sku.err_model / actual_total).round(3)
    per_sku["naive_wape"] = (per_sku.err_naive / actual_total).round(3)
    per_sku = per_sku.drop(columns=["err_model", "err_naive"]).round({"model_13w": 0, "naive_13w": 0})
    return overall, per_sku


# ============ STEP 9: explainability - which features drive the forecast ============
def feature_importance(m50):
    imp = (pd.DataFrame({"feature": FEATURES,
                         "importance_gain": model.feature_importance(m50, FEATURES)})
             .sort_values("importance_gain", ascending=False))
    imp["share"] = (imp.importance_gain / imp.importance_gain.sum()).round(3)
    return imp


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Model: {MODEL} ({model.METHOD_VERSION}) -> {OUT_DIR}/")
    items, demand, events, asof = load_data()
    weekly, weeks = to_weekly(demand, items, asof)
    event_table = build_event_table(events, items.client_id.unique(), weeks)

    # STEP 7: backtests
    results, accuracy = {}, []
    for name, weeks_back in BACKTEST_WINDOWS.items():
        results[name], per_sku = backtest(weekly, weeks, items, event_table, demand, weeks_back)
        accuracy.append(per_sku.assign(window=name))
    pd.concat(accuracy).to_csv(f"{OUT_DIR}/backtest_accuracy.csv", index=False)

    # STEP 8: final models on ALL history -> live forecast
    live, m50 = predict(weekly, items, event_table)
    live = live.merge(items[["sku", "client_id", "warehouse", "channel"]], on="sku")
    live["created_at"], live["method_version"] = asof, model.METHOD_VERSION
    live.to_csv(f"{OUT_DIR}/sku_forecast.csv", index=False)

    # STEP 9: only tree models have feature importance
    imp_file = f"{OUT_DIR}/feature_importance.csv"
    imp = feature_importance(m50) if m50 is not None else None
    if imp is not None:
        imp.to_csv(imp_file, index=False)
    elif os.path.exists(imp_file):
        os.remove(imp_file)

    summary = dict(as_of=str(asof.date()), model=MODEL, method_version=model.METHOD_VERSION,
                   horizon_weeks=H, backtest=results)
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(results, indent=2))
    if imp is not None:
        print(imp.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
