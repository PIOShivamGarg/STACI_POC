"""Chronos-2 (Amazon, Apache 2.0): a pre-trained time-series foundation model.

No training on our data and no hand-made features: it reads each SKU's weekly history and
returns the 0.5 (P50) and 0.9 (P90) quantiles directly. Event uplift is passed as a covariate,
known for both the past and the 13 future weeks.

The model weights are downloaded from Hugging Face on first use (cached afterwards),
then everything runs locally on CPU.
"""
import numpy as np
import pandas as pd
from chronos import Chronos2Pipeline

METHOD_VERSION = "chronos2-v0.2"
MODEL_ID = "amazon/chronos-2"
DEVICE = "cpu"
QUANTILES = [0.5, 0.9]
CROSS_LEARNING = True   # share information across SKUs in one batch (backtest: slightly better on every measure)

_pipeline = None


def _load():
    global _pipeline
    if _pipeline is None:          # load once, reuse for the backtests and the live forecast
        _pipeline = Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=DEVICE)
    return _pipeline


def forecast(history, event_table, h):
    """13-week P50 / P90 per SKU, in the same format as the tree models' forecast."""
    context = (history[["sku", "week", "units", "client_id"]]
               .merge(event_table, on=["client_id", "week"], how="left"))
    last = history.week.max()
    future_weeks = pd.date_range(last + pd.Timedelta(weeks=1), periods=h, freq="7D")
    future = (history[["sku", "client_id"]].drop_duplicates()
              .merge(pd.DataFrame({"week": future_weeks}), how="cross")
              .merge(event_table, on=["client_id", "week"], how="left"))

    pred = _load().predict_df(
        context[["sku", "week", "units", "event_uplift"]],
        future_df=future[["sku", "week", "event_uplift"]],
        id_column="sku", timestamp_column="week", target="units",
        prediction_length=h, quantile_levels=QUANTILES, cross_learning=CROSS_LEARNING)

    p50 = np.clip(pred["0.5"].values, 0, None)
    p90 = np.maximum(np.clip(pred["0.9"].values, 0, None), p50)   # P90 never below P50
    return pd.DataFrame({"sku": pred["sku"].values, "week_start": pd.to_datetime(pred["week"].values),
                         "forecast_p50": p50.round(1), "forecast_p90": p90.round(1)})
