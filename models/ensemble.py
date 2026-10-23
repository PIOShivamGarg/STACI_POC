"""Ensemble: the plain average of several models' P50 and P90 (P90 is then resized by forecast.py like any model).

Different models win in different periods (Chronos in recent weeks, LightGBM / XGBoost in peak season); an
average is usually steadier than any single one. Equal weights: with only a few backtest windows, fitted
weights would mostly chase noise. Members are ML models only. Runs every member, so it takes as long as all
of them together.
"""
METHOD_VERSION = "ensemble-lgbm-chronos-xgboost-v0.2"
MEMBERS = ["lgbm", "chronos", "xgboost"]
