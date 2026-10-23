"""Staci - weekly SKU demand forecast (P50 + P90) with the model chosen in .env.

FORECAST_MODEL picks the model (see config.py). Reads data/ and writes to outputs/<model>/:
  sku_forecast.csv            weekly P50 / P90 per SKU for the next 13 weeks (input to risk.py)
  order_forecast.csv          weekly orders P50 / P90 per client / warehouse / channel (input to risk.py)
  fva_forecast.csv            daily orders the model forecast for the last 14 days, from data before them
                              (the "forecast vs actual" tile, input to risk.py)
  forecast_ledger.csv         every live forecast ever made, to score against actuals once they arrive
  backtest_accuracy.csv       per-SKU forecast vs actual on each backtest window
  feature_importance.csv      which features drive the P50 model (tree models only)
  summary.json                overall backtest scores for both deliverables

Order Volume (deliverable 1), per client / warehouse / channel / week:
  P50  the SKU unit forecast, converted to orders with each SKU's recent orders-per-unit ratio and added up
  P90  a second run of the same model, trained directly on weekly ORDERS of the 12 client / warehouse /
       channel series, with the same features and P90 sizing. Adding up SKU P90s instead would assume every
       SKU runs high at once (actual stayed under it ~98% of the time, not 90%).
  Backtested: the direct model's P50 was less accurate than the SKU-based P50 (12 series are little to learn
  from), so it is reported for comparison (direct_ml_p50) but not used.

The backtest replays the forecast from 8 past start dates and scores three things:
  sku_demand      units per SKU per week - the engine both deliverables are built on
  order_volume    orders per client / warehouse / channel / week (deliverable 1: Order Volume)
  shortfall_risk  demand during each SKU's lead time (deliverable 2: Shortfall Risk, proxy -
                  a true alert backtest needs historical stock snapshots, which we don't have)
The same backtest sizes the P90: its spread above P50 is widened or narrowed so that actual demand stays
under P90 about 90% of the time (separately by weeks ahead, and for peak season).
"""
import importlib
import json
import os

import numpy as np
import pandas as pd

from config import DATA_DIR, H, MODEL, MODEL_MODULE, MODELS, OUT_DIR
from risk import FVA_DAYS, order_volume

model = importlib.import_module(MODEL_MODULE)

ITEM_COLS = ["sku", "client_id", "warehouse", "channel", "category"]
CATS = ["client_id", "warehouse", "channel", "category"]
LAGS = [1, 2, 4, 13, 52]              # weeks before the forecast start
ROLL_WINDOWS = [4, 8, 13]
MIN_HISTORY = 56                      # weeks of history needed before a start date can be used for training
# Tree models predict each of the 13 weeks directly (no feeding predictions back in), as a ratio of the
# SKU's recent level (`scale`), so one model can learn from small and large SKUs alike.
SCALED = ([f"lag_{L}" for L in LAGS] + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
          + ["roll_std_8", "last_year", "last_year_smooth"])
FEATURES = (SCALED + ["zero_share_13", "log_scale", "horizon", "week_of_year", "month",
                      "event_uplift", "event_prev_week", "event_next_week"] + CATS)

# Backtest start dates, in weeks before the last full week. 13 = the last 13 weeks; 52 = last year's peak season.
BACKTEST_WEEKS_BACK = [13, 20, 26, 33, 39, 46, 52, 59]
NAMED_WINDOWS = {"last_13_weeks": 13, "peak_season_last_year": 52}

P90_TARGET = 0.9                      # actual demand should stay under P90 this often
MIN_SPREAD = 0.5                      # units/week: smallest P90 - P50 gap used when resizing P90
PEAK_WEEKS = set(range(45, 54)) | {1} # ISO weeks treated as peak season (November to New Year)


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


def order_series(demand, items, weeks):
    """Weekly orders per client / warehouse / channel, shaped like the SKU table (an `sku` id per series,
    orders in `units`) so every model can forecast it unchanged."""
    keys = ["client_id", "warehouse", "channel"]
    series = items[keys].drop_duplicates().assign(category="all")
    series["sku"] = series.client_id + "|" + series.warehouse + "|" + series.channel
    d = demand.assign(week=demand.date.dt.to_period("W-SUN").dt.start_time,
                      sku=demand.client_id + "|" + demand.warehouse + "|" + demand.channel)
    grid = pd.MultiIndex.from_product([series.sku, weeks], names=["sku", "week"])
    weekly = d.groupby(["sku", "week"]).orders.sum().reindex(grid, fill_value=0).rename("units").reset_index()
    return weekly.merge(series[ITEM_COLS], on="sku"), series[ITEM_COLS].reset_index(drop=True)


# ============ STEP 3: event uplift for every client x week (past AND future), computed once ============
def build_event_table(events, clients, weeks):
    """Average daily demand multiplier over each week (overlapping events multiply, as in ml_risk.py), so a
    4-day event in a 7-day week counts for 4/7 of its uplift. Also the uplift of the week before and after."""
    all_weeks = pd.date_range(weeks[0] - pd.Timedelta(weeks=1), weeks[-1] + pd.Timedelta(weeks=H + 1), freq="7D")
    days = pd.date_range(all_weeks[0], all_weeks[-1] + pd.Timedelta(days=6))
    tables = []
    for client in clients:
        mult = np.ones(len(days))
        for e in events[events.client_id.isin([client, "ALL"])].itertuples():
            mult[(days >= e.start_date) & (days <= e.end_date)] *= e.uplift
        weekly = pd.Series(mult.reshape(-1, 7).mean(axis=1), index=all_weeks)
        tables.append(pd.DataFrame({"client_id": client, "week": all_weeks, "event_uplift": weekly.values,
                                    "event_prev_week": weekly.shift(1, fill_value=1.0).values,
                                    "event_next_week": weekly.shift(-1, fill_value=1.0).values}))
    return pd.concat(tables, ignore_index=True)


# ============ STEP 4: features (only data up to the start date -> no leakage) ============
def make_rows(units, weeks, items, event_table, origins, horizons):
    """One row per SKU x start date x weeks ahead. `units` is SKU x week; a start date is the index of the last
    week whose actuals are known. Features describe the history up to then, plus the target week's calendar
    and events (known in advance) and what the SKU sold in that week last year."""
    all_weeks = pd.DatetimeIndex(weeks).append(pd.date_range(weeks[-1] + pd.Timedelta(weeks=1), periods=H, freq="7D"))
    ev = (items[["sku", "client_id"]].merge(event_table, on="client_id")
          .pivot(index="sku", columns="week").reindex(index=items.sku))
    ev = {c: ev[c].reindex(columns=all_weeks).values for c in ["event_uplift", "event_prev_week", "event_next_week"]}
    n_weeks = units.shape[1]
    out = []
    for t in origins:
        hist = units[:, :t + 1]
        f = {f"lag_{L}": hist[:, -L] for L in LAGS}
        f.update({f"roll_mean_{w}": hist[:, -w:].mean(axis=1) for w in ROLL_WINDOWS})
        f["roll_std_8"] = hist[:, -8:].std(axis=1, ddof=1)
        f["zero_share_13"] = (hist[:, -13:] == 0).mean(axis=1)
        scale = np.maximum(f["roll_mean_13"], 1.0)
        for h in horizons:
            tgt = t + h
            row = {k: v / scale if k in SCALED else v for k, v in f.items()}
            row.update(sku=items.sku.values, week_start=all_weeks[tgt], horizon=h, scale=scale,
                       log_scale=np.log(scale), last_year=units[:, tgt - 52] / scale,
                       last_year_smooth=units[:, tgt - 53:tgt - 50].mean(axis=1) / scale,
                       units=units[:, tgt] if tgt < n_weeks else np.nan,
                       **{c: ev[c][:, tgt] for c in ev})
            out.append(pd.DataFrame(row))
    df = pd.concat(out, ignore_index=True).merge(items[ITEM_COLS], on="sku")
    df["week_of_year"] = df.week_start.dt.isocalendar().week.astype(int)
    df["month"] = df.week_start.dt.month
    df[CATS] = df[CATS].astype("category")
    return df


def unit_matrix(history, items):
    wide = history.pivot(index="sku", columns="week", values="units").reindex(index=items.sku)
    return wide.values.astype(float), wide.columns


# ============ STEP 5 + 6: train two models (P50 and P90, see models/) and forecast 13 weeks ============
def tree_forecast(module, history, items, event_table):
    units, weeks = unit_matrix(history, items)
    last = len(weeks) - 1
    train = make_rows(units, weeks, items, event_table, range(MIN_HISTORY - 1, last), range(1, H + 1))
    train = train.dropna(subset=["units"])
    m50, m90 = module.train(train[FEATURES], train.units / train.scale, train.scale)
    new = make_rows(units, weeks, items, event_table, [last], range(1, H + 1))
    p50 = np.clip(m50.predict(new[FEATURES]), 0, None) * new.scale
    p90 = np.maximum(np.clip(m90.predict(new[FEATURES]), 0, None) * new.scale, p50)   # P90 never below P50
    fc = pd.DataFrame({"sku": new.sku.values, "week_start": new.week_start.values,
                       "forecast_p50": p50.values, "forecast_p90": p90.values})
    return fc, m50


def predict(history, items, event_table, module=model):
    """13-week forecast from `history`. Returns (forecast, P50 model or None).

    Tree models (lgbm, xgboost) are trained on our features first. Pre-trained models (chronos) and
    statistical models (stats) forecast straight from the weekly history. The ensemble averages its members.
    """
    if hasattr(module, "MEMBERS"):
        members = [predict(history, items, event_table, importlib.import_module(MODELS[m]))[0] for m in module.MEMBERS]
        fc = (pd.concat(members).groupby(["sku", "week_start"])[["forecast_p50", "forecast_p90"]]
                .mean().reset_index())
        return fc, None
    if hasattr(module, "train"):
        return tree_forecast(module, history, items, event_table)
    return module.forecast(history, event_table, H), None


# ============ STEP 7: P90 sizing from backtest errors ============
def p90_cell(fc):
    """Which P90 adjustment applies: weeks-ahead band (1-4, 5-8, 9-13) x peak season or not."""
    band = np.select([fc.horizon <= 4, fc.horizon <= 8], [0, 1], 2)
    peak = pd.to_datetime(fc.week_start).dt.isocalendar().week.isin(PEAK_WEEKS).values
    return pd.Series(list(zip(band, peak)), index=fc.index)


def p90_factors(bt):
    """How many 'P90 - P50 gaps' above P50 actual demand reached, at the 90th percentile, per cell."""
    spread = (bt.forecast_p90 - bt.forecast_p50).clip(lower=MIN_SPREAD)
    z = (bt.units - bt.forecast_p50) / spread
    cell = p90_cell(bt)
    factors = z.groupby(cell).quantile(P90_TARGET).to_dict()
    factors["all"] = z.quantile(P90_TARGET)
    return factors


def resize_p90(fc, factors):
    spread = (fc.forecast_p90 - fc.forecast_p50).clip(lower=MIN_SPREAD)
    k = p90_cell(fc).map(lambda c: factors.get(c, factors["all"])).clip(lower=0)
    return fc.assign(forecast_p90=np.maximum(fc.forecast_p50 + k * spread, fc.forecast_p50))


# ============ STEP 8: backtest - train on the past, compare with what actually happened ============
def wape(forecast_col, actual_col):
    """Weighted absolute percentage error: total |error| / total actual."""
    return np.abs(forecast_col - actual_col).sum() / actual_col.sum()


def score(p50, p90, actual, naive, seasonal_naive):
    """WAPE of P50 and of the two benchmarks, plus how often actual stayed within P90."""
    return dict(model=round(wape(p50, actual), 3), naive=round(wape(naive, actual), 3),
                seasonal_naive=round(wape(seasonal_naive, actual), 3),
                p90_coverage=round((actual <= p90).mean(), 3))


def order_volume_backtest(fc, demand, items, cut, orders):
    """Deliverable 1 - forecast orders per client / warehouse / channel / week vs actual orders, built the
    same way as the live run (risk.order_volume): P50 from SKU units, P90 from the direct orders model."""
    keys = ["client_id", "warehouse", "channel", "week_start"]
    units = fc[["sku", "week_start", "forecast_p50", "forecast_p90", "naive", "seasonal_naive"]].merge(
        items[["sku", "client_id", "warehouse", "channel"]], on="sku")
    asof = cut - pd.Timedelta(days=1)    # only orders seen before the cut set the orders-per-unit ratio
    model_ov = order_volume(units, demand, asof)
    bench = {b: order_volume(units.assign(forecast_p50=units[b], forecast_p90=units[b]), demand, asof)
                .set_index(keys).orders_p50.rename(b) for b in ["naive", "seasonal_naive"]}

    test = demand[demand.date >= cut].assign(week_start=lambda d: d.date.dt.to_period("W-SUN").dt.start_time)
    actual = (test[test.week_start.isin(fc.week_start.unique())]
              .groupby(keys).orders.sum().rename("actual").reset_index())
    ov = (model_ov.join(pd.concat(bench.values(), axis=1), on=keys)
                  .merge(actual, on=keys, how="left").fillna({"actual": 0})
                  .merge(orders[keys + ["forecast_p50", "forecast_p90"]], on=keys))
    p90 = np.maximum(ov.forecast_p90, ov.orders_p50)
    return dict(score(ov.orders_p50, p90, ov.actual, ov.naive, ov.seasonal_naive),
                direct_ml_p50=round(wape(ov.forecast_p50, ov.actual), 3),
                sum_of_sku_p90_coverage=round((ov.actual <= ov.orders_p90).mean(), 3))


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
        rows.append((daily("forecast_p50"), daily("forecast_p90"), sold, daily("naive"), daily("seasonal_naive")))
    lt = pd.DataFrame(rows, columns=["p50", "p90", "actual", "naive", "seasonal_naive"])
    return score(lt.p50, lt.p90, lt.actual, lt.naive, lt.seasonal_naive)


def backtest_forecast(weekly, weeks, items, event_table, weeks_back):
    """Forecast from a past start date and line it up with what actually sold."""
    cut = weeks[-weeks_back]             # first week of the test window
    history = weekly[weekly.week < cut]
    fc, _ = predict(history, items, event_table)
    fc = fc.merge(weekly[["sku", "week", "units"]], left_on=["sku", "week_start"], right_on=["sku", "week"])
    # benchmarks: average of the last 4 weeks before the cut, repeated; and the same week last year
    naive = history[history.week >= cut - pd.Timedelta(weeks=4)].groupby("sku").units.mean().rename("naive")
    last_year = weekly.assign(week=weekly.week + pd.Timedelta(weeks=52))[["sku", "week", "units"]]
    fc = (fc.merge(naive, on="sku")
            .merge(last_year.rename(columns={"units": "seasonal_naive"}), on=["sku", "week"], how="left")
            .fillna({"seasonal_naive": 0}))
    fc["horizon"] = (fc.week_start - cut).dt.days // 7 + 1
    return fc.drop(columns="week").assign(weeks_back=weeks_back, cut=cut)


def score_window(fc, demand, items, orders):
    cut = fc.cut.iloc[0]
    overall = {"sku_demand": score(fc.forecast_p50, fc.forecast_p90, fc.units, fc.naive, fc.seasonal_naive),
               "order_volume": order_volume_backtest(fc, demand, items, cut, orders),
               "shortfall_risk": shortfall_backtest(fc, demand, items, cut)}

    fc = fc.assign(err_model=(fc.forecast_p50 - fc.units).abs(), err_naive=(fc.naive - fc.units).abs())
    per_sku = fc.groupby("sku").agg(actual_13w=("units", "sum"), model_13w=("forecast_p50", "sum"),
                                    naive_13w=("naive", "sum"), err_model=("err_model", "sum"),
                                    err_naive=("err_naive", "sum")).reset_index()
    actual_total = per_sku.actual_13w.clip(lower=1)
    per_sku["model_wape"] = (per_sku.err_model / actual_total).round(3)
    per_sku["naive_wape"] = (per_sku.err_naive / actual_total).round(3)
    per_sku = per_sku.drop(columns=["err_model", "err_naive"]).round({"model_13w": 0, "naive_13w": 0})
    return overall, per_sku


def backtest(weekly, weeks, items, event_table, demand, order_weekly, series):
    """Forecast SKU units and orders per series from every start date, resize each window's P90 with factors
    learned on the OTHER windows (so the coverage reported is honest), then score. Returns per-window scores,
    per-SKU accuracy, and the P90 factors from all windows (SKU units, orders) for the live forecast."""
    raw = {wb: backtest_forecast(weekly, weeks, items, event_table, wb) for wb in BACKTEST_WEEKS_BACK}
    raw_orders = {wb: backtest_forecast(order_weekly, weeks, series, event_table, wb) for wb in BACKTEST_WEEKS_BACK}
    others = lambda runs, wb: pd.concat(f for w, f in runs.items() if w != wb)
    results, accuracy = {}, []
    for wb, fc in raw.items():
        resized = resize_p90(fc, p90_factors(others(raw, wb)))
        orders = (resize_p90(raw_orders[wb], p90_factors(others(raw_orders, wb)))
                  .drop(columns=["client_id", "warehouse", "channel"], errors="ignore")
                  .merge(series[["sku", "client_id", "warehouse", "channel"]], on="sku"))
        overall, per_sku = score_window(resized, demand, items, orders)
        overall["sku_demand"]["p90_coverage_before_resizing"] = round((fc.units <= fc.forecast_p90).mean(), 3)
        results[wb] = overall
        accuracy.append(per_sku.assign(window=f"{wb}_weeks_back", window_start=fc.cut.iloc[0].date()))
    return (results, pd.concat(accuracy), p90_factors(pd.concat(raw.values())),
            p90_factors(pd.concat(raw_orders.values())))


def average_scores(results):
    """Mean of each score over all backtest windows."""
    sections = next(iter(results.values()))
    return {s: {k: round(float(np.mean([r[s][k] for r in results.values()])), 3) for k in sections[s]}
            for s in sections}


# ============ STEP 9: explainability - which features drive the forecast ============
def feature_importance(m50):
    imp = (pd.DataFrame({"feature": FEATURES,
                         "importance_gain": model.feature_importance(m50, FEATURES)})
             .sort_values("importance_gain", ascending=False))
    imp["share"] = (imp.importance_gain / imp.importance_gain.sum()).round(3)
    return imp


def factor_labels(factors):
    return {f"{'weeks 1-4' if b == 0 else 'weeks 5-8' if b == 1 else 'weeks 9-13'}{', peak' if p else ''}":
            round(float(v), 2) for (b, p), v in sorted((c, v) for c, v in factors.items() if c != "all")}


def live_forecast(history, items, event_table, factors):
    """Final model on ALL history -> 13-week forecast with the backtest-sized P90."""
    live, m50 = predict(history, items, event_table)
    live["horizon"] = (live.week_start - live.week_start.min()).dt.days // 7 + 1
    live = resize_p90(live, factors).drop(columns="horizon").round({"forecast_p50": 1, "forecast_p90": 1})
    return live, m50


def fva_forecast(weekly, items, event_table, demand, asof):
    """Orders per client / warehouse / day that the model would have forecast for the forecast-vs-actual
    window (the last FVA_DAYS days), trained only on full weeks that ended before the window started.
    Built like the live Order Volume P50 (SKU units -> orders), then split over the days of each week with
    the client / warehouse's recent weekday pattern of orders."""
    start = asof - pd.Timedelta(days=FVA_DAYS - 1)
    history = weekly[weekly.week + pd.Timedelta(days=6) < start]
    cut = history.week.max() + pd.Timedelta(weeks=1)          # first forecast week
    fc, _ = predict(history, items, event_table)
    fc = fc.merge(items[["sku", "client_id", "warehouse", "channel"]], on="sku")
    known = demand[demand.date < cut]
    weekly_orders = (order_volume(fc, known, cut - pd.Timedelta(days=1))
                     .groupby(["client_id", "warehouse", "week_start"]).orders_p50.sum().reset_index())
    recent = known[known.date >= cut - pd.Timedelta(weeks=13)]
    share = recent.groupby(["client_id", "warehouse", recent.date.dt.dayofweek.rename("dow")]).orders.sum()
    share = (share / share.groupby(level=[0, 1]).transform("sum")).rename("share")
    days = pd.DataFrame({"date": pd.date_range(start, asof)})
    days["week_start"] = days.date.dt.to_period("W-SUN").dt.start_time
    out = weekly_orders.merge(days, on="week_start")
    out = out.assign(dow=out.date.dt.dayofweek).join(share, on=["client_id", "warehouse", "dow"]).fillna({"share": 1 / 7})
    out["orders_forecast"] = (out.orders_p50 * out.share).round(1)
    out["forecast_data_to"] = (cut - pd.Timedelta(days=1)).date()
    return out[["client_id", "warehouse", "date", "orders_forecast", "forecast_data_to"]]


def save_to_ledger(live):
    """Keep every live forecast (one per as-of date), so it can be scored against actuals later."""
    path = f"{OUT_DIR}/forecast_ledger.csv"
    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["week_start", "created_at"])
        live = pd.concat([old[old.created_at != live.created_at.iloc[0]], live])
    live.to_csv(path, index=False, date_format="%Y-%m-%d")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Model: {MODEL} ({model.METHOD_VERSION}) -> {OUT_DIR}/")
    items, demand, events, asof = load_data()
    weekly, weeks = to_weekly(demand, items, asof)
    order_weekly, series = order_series(demand, items, weeks)
    event_table = build_event_table(events, items.client_id.unique(), weeks)

    # STEP 8: backtests (they also size the P90)
    results, accuracy, factors, order_factors = backtest(weekly, weeks, items, event_table, demand,
                                                         order_weekly, series)
    accuracy.to_csv(f"{OUT_DIR}/backtest_accuracy.csv", index=False)

    # final models on ALL history -> live forecasts: orders per series (its P90 is used for deliverable 1),
    # units per SKU
    orders, _ = live_forecast(order_weekly, series, event_table, order_factors)
    orders = (orders.merge(series[["sku", "client_id", "warehouse", "channel"]], on="sku").drop(columns="sku")
                    .rename(columns={"forecast_p50": "orders_p50", "forecast_p90": "orders_p90"}))
    orders["created_at"], orders["method_version"] = asof, model.METHOD_VERSION
    orders.to_csv(f"{OUT_DIR}/order_forecast.csv", index=False)

    fva_forecast(weekly, items, event_table, demand, asof).to_csv(f"{OUT_DIR}/fva_forecast.csv", index=False)

    live, m50 = live_forecast(weekly, items, event_table, factors)
    live = live.merge(items[["sku", "client_id", "warehouse", "channel"]], on="sku")
    live["created_at"], live["method_version"] = asof, model.METHOD_VERSION
    live.to_csv(f"{OUT_DIR}/sku_forecast.csv", index=False)
    save_to_ledger(live)

    # STEP 9: only tree models have feature importance
    imp_file = f"{OUT_DIR}/feature_importance.csv"
    imp = feature_importance(m50) if m50 is not None else None
    if imp is not None:
        imp.to_csv(imp_file, index=False)
    elif os.path.exists(imp_file):
        os.remove(imp_file)

    backtest_summary = {name: results[wb] for name, wb in NAMED_WINDOWS.items()}
    backtest_summary["all_windows"] = average_scores(results)
    summary = dict(as_of=str(asof.date()), model=MODEL, method_version=model.METHOD_VERSION,
                   horizon_weeks=H, backtest=backtest_summary,
                   backtest_windows={f"{wb}_weeks_back": r for wb, r in results.items()},
                   p90_factors=factor_labels(factors), order_p90_factors=factor_labels(order_factors))
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(backtest_summary, indent=2))
    if imp is not None:
        print(imp.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
