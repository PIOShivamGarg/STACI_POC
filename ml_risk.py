"""Staci - ML stockout-risk model.

Learns from history when a SKU really runs out, instead of the fixed "days of stock" rule.
  question  : will this SKU run out of stock in the next 30 days?
  model     : LightGBM classifier, probabilities calibrated on a held-out period
  features  : stock, recent sales, trend, seasonality, upcoming events, inbound stock, supplier lateness
  baseline  : the rule (days of stock = available / recent daily sales, flag under 30 days)

  python stock_sim.py   # once: builds data/stock_history.csv (synthetic stand-in for real stock history)
  python ml_risk.py     # trains, evaluates against the rule, writes outputs/ml_risk/
risk.py then calls score() to add risk_probability and top_drivers to inventory_risk.csv.
"""
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from config import DATA_DIR, OUTPUT_ROOT
from stock_sim import LABEL_DAYS

ML_DIR = f"{OUTPUT_ROOT}/ml_risk"
MODEL_FILE = f"{ML_DIR}/model.joblib"

# probability -> the same risk labels the portal uses
HIGH_P, WATCH_P, MONITOR_P = 0.50, 0.25, 0.10
RULE_COVER_DAYS = 30      # baseline rule: flag SKUs with under 30 days of stock

# time-based split (never shuffle), relative to the as-of date: train on the past, calibrate on the
# next 6 months, test on the most recent months
TRAIN_MONTHS_BACK, VAL_MONTHS_BACK = 15, 9

CATS = ["warehouse", "category", "channel", "client_id"]
NUMERIC = ["available_qty", "avg_daily_7", "avg_daily_28", "avg_daily_91", "cover_days", "trend_28_91", "cv_28",
           "seasonal_ahead", "event_mult_30d_mean", "event_mult_30d_max", "inbound_qty_30d", "inbound_cover_days",
           "coverage_ratio_30d", "days_to_next_arrival", "lead_time_days", "lead_time_std_days", "safety_stock_days",
           "supplier_delay_mean_last3", "supplier_delay_max_last3", "stockout_days_26w", "days_since_stockout",
           "week_of_year"]
FEATURES = NUMERIC + CATS
PARAMS = dict(n_estimators=300, learning_rate=0.04, num_leaves=15, min_child_samples=40, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, random_state=42, verbose=-1)

# English text for the top drivers (value-aware)
DRIVER_TEXT = {
    "cover_days": lambda v: f"only {v:.0f} days of stock at recent sales",
    "coverage_ratio_30d": lambda v: f"stock plus inbound covers only {v:.0%} of the next 30 days' demand",
    "available_qty": lambda v: f"low available stock ({v:,.0f} units)",
    "avg_daily_7": lambda v: f"sales running at {v:.1f} units a day this week",
    "avg_daily_28": lambda v: f"sales running at {v:.1f} units a day",
    "avg_daily_91": lambda v: f"steady demand of {v:.1f} units a day",
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
    """Features from demand BEFORE day index d, plus events over the next 30 days. u_cs = cumulative daily units."""
    avg = lambda a, b: (u_cs[b] - u_cs[a]) / max(1, b - a) if b > a >= 0 else np.nan   # mean over [a, b)
    a7, a28, a91 = avg(d - 7, d), avg(d - 28, d), avg(d - 91, d)
    daily = np.diff(u_cs[max(0, d - 28):d + 1])
    last_year_next = avg(d - 364, d - 334)
    last_year_prev = avg(d - 392, d - 364)
    ahead = ev[d:d + LABEL_DAYS]
    return dict(avg_daily_7=a7, avg_daily_28=a28, avg_daily_91=a91,
                trend_28_91=a28 / a91 if a91 and a91 > 0 else np.nan,
                cv_28=daily.std() / daily.mean() if len(daily) and daily.mean() > 0 else np.nan,
                seasonal_ahead=last_year_next / last_year_prev if last_year_prev and last_year_prev > 0 else np.nan,
                event_mult_30d_mean=ahead.mean(), event_mult_30d_max=ahead.max())


def finish_features(df):
    """Features that combine stock and demand (used identically in training and scoring)."""
    a28 = df.avg_daily_28.where(df.avg_daily_28 > 0)
    df["cover_days"] = (df.available_qty / a28).fillna(999).clip(upper=999)
    df["inbound_cover_days"] = (df.inbound_qty_30d / a28).fillna(0).clip(upper=999)
    df["coverage_ratio_30d"] = ((df.available_qty + df.inbound_qty_30d) / (a28 * LABEL_DAYS)).fillna(99).clip(upper=99)
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
    dates = pd.date_range(demand.date.min(), asof + pd.Timedelta(days=LABEL_DAYS + 5), freq="D")
    units = (demand.groupby(["sku", "date"]).units_shipped.sum().unstack(fill_value=0)
                   .reindex(index=items.sku, columns=dates, fill_value=0).values.astype(float))
    cs = np.hstack([np.zeros((len(items), 1)), np.cumsum(units, axis=1)])
    return dates, cs, event_matrix(items, events, dates)


def training_table(items, demand, events, history, asof):
    dates, cs, ev = daily_arrays(items, demand, events, asof)
    idx = {s: i for i, s in enumerate(items.sku)}
    day = {d: i for i, d in enumerate(dates)}
    rows = []
    for r in history.itertuples(index=False):
        k, d = idx[r.sku], day[r.week_start]
        if d < 91:
            continue
        rows.append({**demand_features(cs[k], ev[k], d), "sku": r.sku, "week_start": r.week_start})
    feats = pd.DataFrame(rows)
    df = (history.merge(feats, on=["sku", "week_start"]).merge(items, on="sku"))
    df["week_of_year"] = df.week_start.dt.isocalendar().week.astype(int)
    df = df.rename(columns={"supplier_lead_time_days": "lead_time_days"})
    return finish_features(df)


# ============ training and evaluation ============
def flag_stats(y, flagged):
    tp = int((y[flagged] == 1).sum())
    return dict(flagged=int(flagged.sum()), precision=round(tp / max(1, flagged.sum()), 3),
                recall=round(tp / max(1, int((y == 1).sum())), 3))


def evaluate(y, p, cover):
    y = np.asarray(y)
    rule_flags = np.asarray(cover < RULE_COVER_DAYS)
    n_flags = int(rule_flags.sum())
    top = np.zeros(len(y), dtype=bool)
    top[np.argsort(-p)[:n_flags]] = True          # ML flags the same number of SKU-weeks as the rule
    return dict(rows=len(y), stockout_rate=round(float(y.mean()), 3),
                ml=dict(pr_auc=round(average_precision_score(y, p), 3), roc_auc=round(roc_auc_score(y, p), 3),
                        brier=round(brier_score_loss(y, p), 4), at_same_flag_count=flag_stats(y, top),
                        at_50pct=flag_stats(y, p >= HIGH_P)),
                rule=dict(pr_auc=round(average_precision_score(y, -np.asarray(cover)), 3),
                          roc_auc=round(roc_auc_score(y, -np.asarray(cover)), 3),
                          at_30_days=flag_stats(y, rule_flags)))


def train():
    items, demand, events, inv, _ = load_context()
    path = f"{DATA_DIR}/stock_history.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run stock_sim.py first (or supply real stock history in that format)")
    asof = inv.snapshot_date.max()
    TRAIN_END = (asof - pd.DateOffset(months=TRAIN_MONTHS_BACK)).normalize()
    VAL_END = (asof - pd.DateOffset(months=VAL_MONTHS_BACK)).normalize()
    df = training_table(items, demand, events, pd.read_csv(path, parse_dates=["week_start"]), asof)
    tr, va, te = (df[df.week_start < TRAIN_END], df[(df.week_start >= TRAIN_END) & (df.week_start < VAL_END)],
                  df[df.week_start >= VAL_END])
    clf = lgb.LGBMClassifier(**PARAMS).fit(tr[FEATURES], tr.stockout_next_30d)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
        clf.predict_proba(va[FEATURES])[:, 1], va.stockout_next_30d)
    predict = lambda x: iso.predict(clf.predict_proba(x[FEATURES])[:, 1])

    metrics = dict(trained_at=str(pd.Timestamp.now().date()), horizon_days=LABEL_DAYS,
                   rows=dict(train=len(tr), validation=len(va), test=len(te)),
                   split=dict(train_before=str(TRAIN_END.date()), validation_before=str(VAL_END.date())),
                   thresholds=dict(high=HIGH_P, watch=WATCH_P, monitor=MONITOR_P),
                   validation=evaluate(va.stockout_next_30d, predict(va), va.cover_days),
                   test_recent=evaluate(te.stockout_next_30d, predict(te), te.cover_days))
    os.makedirs(ML_DIR, exist_ok=True)
    joblib.dump(dict(model=clf, calibrator=iso), MODEL_FILE)
    with open(f"{ML_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    (pd.DataFrame(dict(feature=FEATURES, importance=clf.booster_.feature_importance("gain")))
       .assign(share=lambda x: (x.importance / x.importance.sum()).round(4))
       .sort_values("importance", ascending=False).to_csv(f"{ML_DIR}/feature_importance.csv", index=False))
    return metrics


# ============ scoring today's snapshot ============
def level_from_probability(p):
    return "High risk" if p >= HIGH_P else "Watch" if p >= WATCH_P else "Monitor" if p >= MONITOR_P else "Healthy"


def explain_row(values, contrib, k=3):
    """Top-k features pushing risk up, as text. contrib are LightGBM per-feature contributions (log-odds)."""
    order = np.argsort(-contrib)
    out = []
    for i in order:
        f = FEATURES[i]
        if contrib[i] <= 0.05 or f not in DRIVER_TEXT:
            continue
        text = DRIVER_TEXT[f](values[f])
        if text is None:
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
        rows.append(dict(sku=it.sku, week_start=asof, available_qty=s.on_hand_qty - s.allocated_qty - s.damaged_hold_qty,
                         inbound_qty_30d=due.expected_qty.sum(),
                         days_to_next_arrival=(open_asn.expected_arrival_date.min() - asof).days if len(open_asn) else np.nan,
                         **sup[it.sku], **demand_features(cs[k], ev[k], d)))
    df = pd.DataFrame(rows).merge(items, on="sku").rename(columns={"supplier_lead_time_days": "lead_time_days"})
    df["week_of_year"] = asof.isocalendar().week
    df = finish_features(df)
    # categories must match training
    raw = bundle["model"].predict_proba(df[FEATURES])[:, 1]
    df["risk_probability"] = np.round(bundle["calibrator"].predict(raw), 3)
    contrib = bundle["model"].predict(df[FEATURES], pred_contrib=True)[:, :-1]
    df["top_drivers"] = [explain_row(df.iloc[i], contrib[i]) for i in range(len(df))]
    df["ml_risk_state"] = df.risk_probability.map(level_from_probability)
    return df[["sku", "risk_probability", "ml_risk_state", "top_drivers", "cover_days"]]


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
