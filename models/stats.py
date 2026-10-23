"""Statistical models (statsforecast, Apache 2.0): one small model per SKU, no shared training.

Each SKU is routed by its own recent history:
  regular sellers -> MSTL + AutoETS: removes the yearly (52-week) pattern, forecasts the rest with
                     exponential smoothing (level + trend), then adds the yearly pattern back.
                     Plain AutoETS can't do this - it silently drops seasons longer than 24 periods.
  slow movers     -> TSB: forecasts how OFTEN the item sells and how MUCH when it does.

P50 = the point forecast. P90 = upper end of the 80% prediction interval (10th-90th percentile).
TSB has no built-in interval, so its interval comes from its own past errors (conformal).
Events are not used - these models only see each SKU's own history.
"""
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, TSB, AutoETS
from statsforecast.utils import ConformalIntervals

METHOD_VERSION = "stats-mstl-ets-tsb-v0.2"
SEASON = 52                    # weeks in a year
SLOW_ZERO_SHARE = 0.3          # more than 30% zero weeks in the last year -> slow mover (TSB)
TSB_ALPHA = 0.2                # TSB smoothing for "how often" and "how much"
INTERVAL = 80                  # 80% interval: its upper end is the 90th percentile (P90)
CONFORMAL_WINDOWS = 2          # past 13-week windows used to size TSB's interval


def _run(model, name, series, h):
    """Forecast every SKU in `series` with one model. Returns sku, week_start, p50, p90."""
    if series.empty:
        return pd.DataFrame(columns=["sku", "week_start", "p50", "p90"])
    out = StatsForecast(models=[model], freq="W-MON").forecast(df=series, h=h, level=[INTERVAL])
    return pd.DataFrame({"sku": out.unique_id.values, "week_start": pd.to_datetime(out.ds.values),
                         "p50": out[name].values, "p90": out[f"{name}-hi-{INTERVAL}"].values})


def forecast(history, event_table, h):
    """13-week P50 / P90 per SKU, in the same format as the other models' forecast."""
    series = history[["sku", "week", "units"]].rename(columns={"sku": "unique_id", "week": "ds", "units": "y"})
    last_year = series[series.ds > series.ds.max() - pd.Timedelta(weeks=SEASON)]
    zero_share = (last_year.y == 0).groupby(last_year.unique_id).mean()
    slow = series.unique_id.map(zero_share) > SLOW_ZERO_SHARE

    regular = _run(MSTL(season_length=SEASON, trend_forecaster=AutoETS(model="ZZN")), "MSTL",
                   series[~slow], h)
    intermittent = _run(TSB(alpha_d=TSB_ALPHA, alpha_p=TSB_ALPHA,
                            prediction_intervals=ConformalIntervals(h=h, n_windows=CONFORMAL_WINDOWS)),
                        "TSB", series[slow], h)
    fc = pd.concat([regular, intermittent])

    p50 = np.clip(fc.p50.values.astype(float), 0, None)
    p90 = np.maximum(np.clip(fc.p90.values.astype(float), 0, None), p50)   # P90 never below P50
    return pd.DataFrame({"sku": fc.sku.values, "week_start": fc.week_start.values,
                         "forecast_p50": p50.round(1), "forecast_p90": p90.round(1)})
