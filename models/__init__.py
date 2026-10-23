"""Forecasting models. Every module has METHOD_VERSION (label stored with every forecast row), plus one of:

Tree models (lgbm, xgb) - trained on forecast.py's features:
  train(X, y, weight) -> (m50, m90) fitted P50 and P90 models (both have .predict(X)); y is demand as a
                                    ratio of each SKU's recent level and weight is that level
  feature_importance(model, names)  total gain per feature, in the order of `names`

Forecast-only models (chronos, stats) - no separate training step, forecast straight from the weekly history:
  forecast(history, event_table, h) -> DataFrame of sku, week_start, forecast_p50, forecast_p90

Ensembles (ensemble) - MEMBERS, a list of FORECAST_MODEL names whose forecasts are averaged.
"""
