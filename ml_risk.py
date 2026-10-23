"""Staci - ML stock-risk models: stockout and overstock.

Learns from history when a SKU really runs out (or ends up with too much stock), instead of fixed
"days of stock" rules. Two LightGBM classifiers on the same features, probabilities calibrated (Platt scaling):
  stockout  : will this SKU run out of stock in the next 30 days?
              baselines: the rule (days of stock = available / recent daily sales, flag under 30 days) and a
              simple projection (stock + inbound covers less than 30 days of recent sales)
  overstock : will stock on hand + inbound still not be used up by actual demand over the next 120 days?
              baseline: the rule (120 or more days of stock at recent sales)
  features  : stock, recent and expected sales, seasonality, upcoming events, inbound stock, supplier lateness

  python stock_sim.py   # once: builds data/stock_history.csv (synthetic stand-in for real stock history)
  python ml_risk.py     # evaluates against the baselines, refits on all history, writes outputs/ml_risk/
risk.py then calls score() to add risk_probability, overstock_probability and top_drivers to inventory_risk.csv.
"""
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import DATA_DIR, OUTPUT_ROOT
from stock_sim import LABEL_DAYS

ML_DIR = f"{OUTPUT_ROOT}/ml_risk"
MODEL_FILE = f"{ML_DIR}/model.joblib"

# probability -> the same risk labels the portal uses
HIGH_P, WATCH_P, MONITOR_P = 0.50, 0.25, 0.10
RULE_COVER_DAYS = 30      # baseline rule: flag SKUs with under 30 days of stock
OVERSTOCK_DAYS = 120      # overstock = stock + inbound not used up by actual demand within 120 days (rule: 120+ days)
OVERSTOCK_P = 0.50        # overstock probability at or above this -> "Possible overstock"
NO_ARRIVAL_DAYS = 999     # days to next arrival when nothing is on the way

# time-based split (never shuffle), relative to the as-of date: train on the past, calibrate on the
# next 6 months, test on the most recent months
TRAIN_MONTHS_BACK, VAL_MONTHS_BACK = 15, 9
# label column -> how far ahead it looks. Training stops that many days before each split, so no training
# label can see into the period it is tested on.
TARGETS = {"stockout_next_30d": LABEL_DAYS, "overstock_next_120d": OVERSTOCK_DAYS}
CALIB_FOLD_MONTHS = [24, 18, 12, 6]   # final model: out-of-time predictions from these cut-offs fit the calibration

CATS = ["warehouse", "category", "channel", "client_id"]
NUMERIC = ["available_qty", "avg_daily_7", "avg_daily_28", "avg_daily_91", "cover_days", "trend_28_91", "cv_28",
           "seasonal_ahead", "event_mult_30d_mean", "event_mult_30d_max", "inbound_qty_30d", "inbound_cover_days",
           "coverage_ratio_30d", "days_to_next_arrival", "lead_time_days", "lead_time_std_days", "safety_stock_days",
           "supplier_delay_mean_last3", "supplier_delay_max_last3", "stockout_days_26w", "days_since_stockout",
           "expected_daily_30d", "forecast_cover_days", "arrival_days_adjusted", "stock_vs_arrival",
           "shortfall_share_30d", "week_of_year"]
FEATURES = NUMERIC + CATS
# Tested and left out: monotonic constraints (slightly worse on the test period) and bigger trees (worse).
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=300, subsample=0.7,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, random_state=42, verbose=-1)

# English text for the top drivers (value-aware)
DRIVER_TEXT = {
    "cover_days": lambda v: f"only {v:.0f} days of stock at recent sales",
    "forecast_cover_days": lambda v: f"only {v:.0f} days of stock at expected sales",
    "stock_vs_arrival": lambda v: ("stock runs out before the next shipment lands" if v < 1
                                   else "stock only just lasts until the next shipment"),
    "shortfall_share_30d": lambda v: (f"stock plus inbound is {v:.0%} short of the next 30 days' expected demand"
                                      if v > 0 else None),
    "arrival_days_adjusted": lambda v: ("no shipment on the way" if v >= NO_ARRIVAL_DAYS
                                        else f"next shipment likely in {v:.0f} days, allowing for supplier delays"),
    "coverage_ratio_30d": lambda v: f"stock plus inbound covers only {v:.0%} of the next 30 days' demand",
    "available_qty": lambda v: f"low available stock ({v:,.0f} units)",
    "avg_daily_7": lambda v: f"sales running at {v:.1f} units a day this week",
    "avg_daily_28": lambda v: f"sales running at {v:.1f} units a day",
    "avg_daily_91": lambda v: f"steady demand of {v:.1f} units a day",
    "expected_daily_30d": lambda v: f"expected sales of {v:.1f} units a day over the next 30 days",
    "trend_28_91": lambda v: f"sales {v - 1:+.0%} versus the 3-month average",
    "cv_28": lambda v: "volatile daily demand",
    "seasonal_ahead": lambda v: f"last year this period's demand moved {v - 1:+.0%}",
    "event_mult_30d_mean": lambda v: f"a promotion or peak ahead (about {v - 1:+.0%} demand)",
    "event_mult_30d_max": lambda v: f"a promotion or peak ahead (up to {v - 1:+.0%} demand)",
    "inbound_qty_30d": lambda v: f"only {v:,.0f} units inbound within 30 days" if v else "nothing inbound within 30 days",
    "inbound_cover_days": lambda v: f"inbound stock adds only {v:.0f} days of cover",
    "days_to_next_arrival": lambda v: ("no shipment on the way" if np.isnan(v) else f"next shipment due in {v:.0f} days"),
    "lead_time_days": lambda v: f"long supplier lead time ({v:.0f} days)",
    "lead_time_std_days": lambda v: f"supplier lead time varies by about {v:.0f} days",
    "safety_stock_days": lambda v: f"thin safety stock ({v:.0f} days)",
    "supplier_delay_mean_last3": lambda v: ("no delivery history" if np.isnan(v)
                                            else f"supplier's last deliveries ran {v:.1f} days late on average"),
    "supplier_delay_max_last3": lambda v: ("no delivery history" if np.isnan(v)
                                           else f"supplier's worst recent delivery was {v:.0f} days late"),
    "stockout_days_26w": lambda v: f"ran out on {v:.0f} days in the last 6 months",
    "days_since_stockout": lambda v: f"ran out only {v:.0f} days ago" if v < 120 else None,   # old stockouts say nothing
    "week_of_year": lambda v: "this time of year is usually busy",
}
# English text for what pushes the OVERSTOCK probability up
OVERSTOCK_TEXT = {
    "cover_days": lambda v: f"{v:.0f} days of stock at recent sales",
    "forecast_cover_days": lambda v: f"{v:.0f} days of stock at expected sales",
    "coverage_ratio_30d": lambda v: f"stock plus inbound is {v:.1f}x the next 30 days' demand",
    "available_qty": lambda v: f"{v:,.0f} units on hand",
    "inbound_qty_30d": lambda v: f"{v:,.0f} more units inbound within 30 days" if v else None,
    "inbound_cover_days": lambda v: f"inbound stock adds {v:.0f} days of cover" if v else None,
    "trend_28_91": lambda v: f"sales {v - 1:+.0%} versus the 3-month average" if v < 1 else None,
    "seasonal_ahead": lambda v: f"last year demand dropped {1 - v:.0%} in the coming period" if v < 1 else None,
    "avg_daily_28": lambda v: f"slow sales ({v:.1f} units a day)",
    "expected_daily_30d": lambda v: f"expected sales of only {v:.1f} units a day",
}


# ============ features ============
def event_matrix(items, events, dates):
    """Daily event demand multiplier per SKU (events are known in advance, so future days are fine)."""
    mult = {}
    for c in items.client_id.unique():
        m = np.ones(len(dates))
        for e in events[events.client_id.isin([c, "ALL"])].itertuples():
            m[(dates >= e.start_date) & (dates <= e.end_date)] *= e.uplift
        mult[c] = m
    return np.vstack([mult[c] for c in items.client_id])


def demand_features(u_cs, ev, d):
    """Features from demand BEFORE each day index in d, plus events over the next 30 days.
    u_cs = cumulative daily units (one SKU), ev = daily event multipliers, d = array of day indexes."""
    d = np.asarray(d)
    def avg(a, b):                                     # mean daily units over [a, b), NaN if out of range
        ok = (a >= 0) & (b > a)
        a_, b_ = np.clip(a, 0, None), np.clip(b, 0, None)
        return np.where(ok, (u_cs[b_] - u_cs[a_]) / np.maximum(1, b - a), np.nan)
    a7, a28, a91 = avg(d - 7, d), avg(d - 28, d), avg(d - 91, d)
    # std / mean of the (up to) 28 days before d, from running sums
    units = np.diff(u_cs)
    sq_cs = np.concatenate([[0], np.cumsum(units ** 2)])
    lo = np.clip(d - 28, 0, None)
    n = d - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = (u_cs[d] - u_cs[lo]) / n
        std = np.sqrt(np.clip((sq_cs[d] - sq_cs[lo]) / n - mean ** 2, 0, None))
        cv = np.where((n > 0) & (mean > 0), std / mean, np.nan)
        last_year_next, last_year_prev = avg(d - 364, d - 334), avg(d - 392, d - 364)
        seasonal = np.where(last_year_prev > 0, last_year_next / last_year_prev, np.nan)
        trend = np.where(a91 > 0, a28 / a91, np.nan)
    ahead = sliding_window_view(ev, LABEL_DAYS)[d]
    return pd.DataFrame(dict(avg_daily_7=a7, avg_daily_28=a28, avg_daily_91=a91, trend_28_91=trend, cv_28=cv,
                             seasonal_ahead=seasonal, event_mult_30d_mean=ahead.mean(axis=1),
                             event_mult_30d_max=ahead.max(axis=1)))


def finish_features(df):
    """Features that combine stock and demand (used identically in training and scoring)."""
    a28 = df.avg_daily_28.where(df.avg_daily_28 > 0)
    df["cover_days"] = (df.available_qty / a28).fillna(999).clip(upper=999)
    df["inbound_cover_days"] = (df.inbound_qty_30d / a28).fillna(0).clip(upper=999)
    df["coverage_ratio_30d"] = ((df.available_qty + df.inbound_qty_30d) / (a28 * LABEL_DAYS)).fillna(99).clip(upper=99)
    # expected sales over the next 30 days: recent sales x how demand moved over the same period last year
    expected = df.avg_daily_28 * df.seasonal_ahead.clip(0.5, 3).fillna(1)
    df["expected_daily_30d"] = expected
    expected = expected.where(expected > 0)
    df["forecast_cover_days"] = (df.available_qty / expected).fillna(999).clip(upper=999)
    # when the next shipment really lands: due date + the supplier's recent lateness
    df["arrival_days_adjusted"] = (df.days_to_next_arrival + df.supplier_delay_mean_last3.fillna(0).clip(lower=0)
                                   ).fillna(NO_ARRIVAL_DAYS).clip(upper=NO_ARRIVAL_DAYS)
    # under 1 = stock runs out before the next shipment (looking at most 30 days ahead)
    df["stock_vs_arrival"] = (df.forecast_cover_days / df.arrival_days_adjusted.clip(1, LABEL_DAYS)).clip(upper=99)
    need = expected * LABEL_DAYS
    df["shortfall_share_30d"] = ((need - df.available_qty - df.inbound_qty_30d) / need).fillna(-5).clip(-5, 1)
    for c in CATS:
        df[c] = pd.Categorical(df[c])
    return df


def load_context():
    items = pd.read_csv(f"{DATA_DIR}/items.csv")
    demand = pd.read_csv(f"{DATA_DIR}/daily_demand.csv", parse_dates=["date"])
    events = pd.read_csv(f"{DATA_DIR}/events.csv", parse_dates=["start_date", "end_date"])
    inv = pd.read_csv(f"{DATA_DIR}/inventory_snapshot.csv", parse_dates=["snapshot_date"])
    asn = pd.read_csv(f"{DATA_DIR}/inbound_asn.csv", parse_dates=["expected_arrival_date"])
    return items, demand, events, inv, asn


def daily_arrays(items, demand, events, asof):
    dates = pd.date_range(demand.date.min(), asof + pd.Timedelta(days=OVERSTOCK_DAYS + 5), freq="D")
    units = (demand.groupby(["sku", "date"]).units_shipped.sum().unstack(fill_value=0)
                   .reindex(index=items.sku, columns=dates, fill_value=0).values.astype(float))
    cs = np.hstack([np.zeros((len(items), 1)), np.cumsum(units, axis=1)])
    return dates, cs, event_matrix(items, events, dates)


def training_table(items, demand, events, history, asof):
    dates, cs, ev = daily_arrays(items, demand, events, asof)
    days = np.arange(91, len(dates) - LABEL_DAYS)     # need 91 days of sales history
    feats = pd.concat([demand_features(cs[k], ev[k], days).assign(sku=sku, date=dates[days])
                       for k, sku in enumerate(items.sku)])
    if "policy" not in history:                        # real stock history has a single policy
        history = history.assign(policy=0)
    df = history.merge(feats, on=["sku", "date"]).merge(items, on="sku")
    df = df.rename(columns={"supplier_lead_time_days": "lead_time_days"})
    df["week_of_year"] = df.date.dt.isocalendar().week.astype(int)
    # overstock label: units actually sold over the next 120 days, against stock + inbound. Unknown (NaN) for
    # the last 120 days of history, which have not fully happened yet.
    k = df.sku.map({s: i for i, s in enumerate(items.sku)}).values
    d = (df.date - dates[0]).dt.days.values
    last = (demand.date.max() - dates[0]).days + 1
    end = np.minimum(d + OVERSTOCK_DAYS, last)
    sold = cs[k, end] - cs[k, d]
    df["overstock_next_120d"] = np.where(d + OVERSTOCK_DAYS <= last,
                                         (df.available_qty + df.inbound_qty_30d > sold).astype(float), np.nan)
    return finish_features(df)


# ============ training and evaluation ============
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_model(df, target="stockout_next_30d"):
    return lgb.LGBMClassifier(**PARAMS).fit(df[FEATURES], df[target])


def fit_calibrator(raw, y):
    """Platt scaling: a smooth curve from raw score to probability (isotonic gave a staircase of tied values)."""
    return LogisticRegression(C=1e4).fit(logit(raw), y)


def calibrate(calibrator, raw):
    return calibrator.predict_proba(logit(raw))[:, 1]


def flag_stats(y, flagged):
    tp = int((y[flagged] == 1).sum())
    return dict(flagged=int(flagged.sum()), precision=round(tp / max(1, flagged.sum()), 3),
                recall=round(tp / max(1, int((y == 1).sum())), 3))


def level_from_probability(p):
    return "High risk" if p >= HIGH_P else "Watch" if p >= WATCH_P else "Monitor" if p >= MONITOR_P else "Healthy"


def evaluate(df, p):
    y, cover = df.stockout_next_30d.values, df.cover_days.values
    rule_flags = cover < RULE_COVER_DAYS
    n_flags = int(rule_flags.sum())
    top = np.zeros(len(y), dtype=bool)
    top[np.argsort(-p)[:n_flags]] = True          # ML flags the same number of SKU-days as the rule
    # calibration check: within each risk level, does the predicted chance match how often it really ran out?
    levels = pd.Series(p).map(level_from_probability)
    by_level = {lvl: dict(rows=int((levels == lvl).sum()), predicted=round(float(p[levels == lvl].mean()), 3),
                          actual=round(float(y[levels == lvl].mean()), 3))
                for lvl in ["High risk", "Watch", "Monitor", "Healthy"] if (levels == lvl).any()}
    projection = df.coverage_ratio_30d.values < 1
    return dict(rows=len(y), stockout_rate=round(float(y.mean()), 3),
                ml=dict(pr_auc=round(average_precision_score(y, p), 3), roc_auc=round(roc_auc_score(y, p), 3),
                        brier=round(brier_score_loss(y, p), 4), at_same_flag_count=flag_stats(y, top),
                        at_50pct=flag_stats(y, p >= HIGH_P), by_level=by_level),
                rule=dict(pr_auc=round(average_precision_score(y, -cover), 3),
                          roc_auc=round(roc_auc_score(y, -cover), 3),
                          at_30_days=flag_stats(y, rule_flags)),
                projection=dict(pr_auc=round(average_precision_score(y, -df.coverage_ratio_30d.values), 3),
                                under_30_days=flag_stats(y, projection)))


def evaluate_overstock(df, p):
    """Overstock model vs the rule (120+ days of stock at recent sales)."""
    y, cover = df.overstock_next_120d.values, df.cover_days.values
    rule_flags = cover >= OVERSTOCK_DAYS
    top = np.zeros(len(y), dtype=bool)
    top[np.argsort(-p)[:int(rule_flags.sum())]] = True
    return dict(rows=len(y), overstock_rate=round(float(y.mean()), 3),
                ml=dict(pr_auc=round(average_precision_score(y, p), 3), roc_auc=round(roc_auc_score(y, p), 3),
                        brier=round(brier_score_loss(y, p), 4), at_same_flag_count=flag_stats(y, top),
                        at_50pct=flag_stats(y, p >= OVERSTOCK_P),
                        predicted_rate=round(float(p.mean()), 3)),
                rule=dict(pr_auc=round(average_precision_score(y, cover), 3),
                          roc_auc=round(roc_auc_score(y, cover), 3),
                          at_120_days=flag_stats(y, rule_flags)))


def fit_target(df, target, months_back):
    """1. honest evaluation: train on the past, calibrate on the next 6 months, test on the most recent months.
       Every planning policy is used to learn; scores are reported on the reference policy (0).
    2. model used for scoring: refit on ALL history. Its calibration comes from out-of-time predictions:
       for each cut-off, train on everything before it and predict the 6 months after it."""
    df = df[df[target].notna()]
    gap = pd.Timedelta(days=TARGETS[target])
    train_end, val_end = months_back(TRAIN_MONTHS_BACK), months_back(VAL_MONTHS_BACK)
    tr = df[df.date < train_end - gap]
    va = df[(df.date >= train_end) & (df.date < val_end - gap)]
    te = df[df.date >= val_end]
    clf = fit_model(tr, target)
    cal = fit_calibrator(clf.predict_proba(va[FEATURES])[:, 1], va[target])
    predict = lambda x: calibrate(cal, clf.predict_proba(x[FEATURES])[:, 1])
    va0, te0 = va[va.policy == 0], te[te.policy == 0]

    oof = []
    for m in CALIB_FOLD_MONTHS:
        cut, end = months_back(m), months_back(m - 6)
        block = df[(df.date >= cut) & (df.date < end - gap)]
        fold = fit_model(df[df.date < cut - gap], target)
        oof.append((fold.predict_proba(block[FEATURES])[:, 1], block[target].values))
    final = fit_model(df, target)
    final_cal = fit_calibrator(np.concatenate([r for r, _ in oof]), np.concatenate([y for _, y in oof]))

    evaluator = evaluate if target == "stockout_next_30d" else evaluate_overstock
    info = dict(rows=dict(train=len(tr), validation=len(va), test=len(te), final_model=len(df)),
                split=dict(train_before=str((train_end - gap).date()), validation_from=str(train_end.date()),
                           validation_before=str((val_end - gap).date()), test_from=str(val_end.date())),
                validation=evaluator(va0, predict(va0)), test_recent=evaluator(te0, predict(te0)))
    return final, final_cal, info


def train():
    items, demand, events, inv, _ = load_context()
    path = f"{DATA_DIR}/stock_history.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run stock_sim.py first (or supply real stock history in that format)")
    asof = inv.snapshot_date.max()
    months_back = lambda m: (asof - pd.DateOffset(months=m)).normalize()
    df = training_table(items, demand, events, pd.read_csv(path, parse_dates=["date"]), asof)

    final, final_cal, stockout = fit_target(df, "stockout_next_30d", months_back)
    over, over_cal, overstock = fit_target(df, "overstock_next_120d", months_back)

    metrics = dict(trained_at=str(pd.Timestamp.now().date()), horizon_days=LABEL_DAYS,
                   thresholds=dict(high=HIGH_P, watch=WATCH_P, monitor=MONITOR_P), **stockout,
                   overstock=dict(horizon_days=OVERSTOCK_DAYS, threshold=OVERSTOCK_P, **overstock))
    os.makedirs(ML_DIR, exist_ok=True)
    joblib.dump(dict(model=final, calibrator=final_cal, overstock_model=over, overstock_calibrator=over_cal),
                MODEL_FILE)
    with open(f"{ML_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    (pd.DataFrame(dict(feature=FEATURES, importance=final.booster_.feature_importance("gain")))
       .assign(share=lambda x: (x.importance / x.importance.sum()).round(4))
       .sort_values("importance", ascending=False).to_csv(f"{ML_DIR}/feature_importance.csv", index=False))
    return metrics


# ============ scoring today's snapshot ============
def explain_row(values, contrib, k=3, texts=DRIVER_TEXT):
    """Top-k features pushing risk up, as text. contrib are LightGBM per-feature contributions (log-odds)."""
    order = np.argsort(-contrib)
    out = []
    for i in order:
        f = FEATURES[i]
        if contrib[i] <= 0.05 or f not in texts:
            continue
        text = texts[f](values[f])
        if text is None or text in out:
            continue
        out.append(text)
        if len(out) == k:
            break
    return "; ".join(out)


def score(asof):
    """risk_probability, top_drivers and a rule baseline flag for every SKU at the as-of date."""
    from stock_sim import supplier_state
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"{MODEL_FILE} not found - run ml_risk.py first")
    bundle = joblib.load(MODEL_FILE)
    items, demand, events, inv, asn = load_context()
    dates, cs, ev = daily_arrays(items, demand, events, asof)
    d = list(dates).index(asof)
    sup = supplier_state(items, demand, asof)
    rows = []
    for k, it in enumerate(items.itertuples(index=False)):
        s = inv[inv.sku == it.sku].iloc[0]
        open_asn = asn[(asn.sku == it.sku) & (asn.status != "Received")]
        due = open_asn[open_asn.expected_arrival_date <= asof + pd.Timedelta(days=LABEL_DAYS)]
        rows.append(dict(sku=it.sku, date=asof, available_qty=s.on_hand_qty - s.allocated_qty - s.damaged_hold_qty,
                         inbound_qty_30d=due.expected_qty.sum(),
                         days_to_next_arrival=(open_asn.expected_arrival_date.min() - asof).days if len(open_asn) else np.nan,
                         **sup[it.sku], **demand_features(cs[k], ev[k], [d]).iloc[0]))
    df = pd.DataFrame(rows).merge(items, on="sku").rename(columns={"supplier_lead_time_days": "lead_time_days"})
    df["week_of_year"] = asof.isocalendar().week
    df = finish_features(df)
    raw = bundle["model"].predict_proba(df[FEATURES])[:, 1]
    df["risk_probability"] = np.round(calibrate(bundle["calibrator"], raw), 3)
    contrib = bundle["model"].predict(df[FEATURES], pred_contrib=True)[:, :-1]
    df["top_drivers"] = [explain_row(df.iloc[i], contrib[i]) for i in range(len(df))]
    df["ml_risk_state"] = df.risk_probability.map(level_from_probability)
    if "overstock_model" in bundle:
        raw = bundle["overstock_model"].predict_proba(df[FEATURES])[:, 1]
        df["overstock_probability"] = np.round(calibrate(bundle["overstock_calibrator"], raw), 3)
        contrib = bundle["overstock_model"].predict(df[FEATURES], pred_contrib=True)[:, :-1]
        df["overstock_drivers"] = [explain_row(df.iloc[i], contrib[i], texts=OVERSTOCK_TEXT) for i in range(len(df))]
    return df[["sku", "risk_probability", "ml_risk_state", "top_drivers", "cover_days"]
              + [c for c in ["overstock_probability", "overstock_drivers"] if c in df]]


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
