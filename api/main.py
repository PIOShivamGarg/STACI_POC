"""Read-only API that serves the POC outputs (outputs/<model>/ and data/) to the React UI.

Run from the project root:  uvicorn api.main:app --reload --port 8000
"""
import json
import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from config import DATA_DIR, MODELS, OUTPUT_ROOT

HISTORY_WEEKS = 16
RISK_LEVELS = ["High risk", "Watch", "Monitor", "Healthy", "Possible overstock"]

app = FastAPI(title="Staci forecasting API")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["GET"], allow_headers=["*"])


def model_dir(model: str) -> str:
    if model not in MODELS:
        raise HTTPException(404, f"Unknown model {model!r}. Choose one of: {', '.join(MODELS)}")
    path = f"{OUTPUT_ROOT}/{model}"
    if not os.path.exists(f"{path}/inventory_risk.csv"):
        raise HTTPException(404, f"No outputs for {model!r} yet - run forecast.py and risk.py first")
    return path


def csv_mtime(path: str) -> float:
    return os.path.getmtime(path)


@lru_cache(maxsize=64)
def _read(path: str, mtime: float, dates: tuple) -> pd.DataFrame:   # mtime in the key refreshes the cache after a re-run
    return pd.read_csv(path, parse_dates=list(dates))


def read(path: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        raise HTTPException(404, f"{path} not found")
    return _read(path, csv_mtime(path), tuple(parse_dates or ()))


def records(df: pd.DataFrame) -> list[dict]:
    return json.loads(df.to_json(orient="records", date_format="iso"))


@app.get("/api/models")
def models():
    return [m for m in MODELS if os.path.exists(f"{OUTPUT_ROOT}/{m}/inventory_risk.csv")]


@app.get("/api/filters")
def filters(model: str = "lgbm"):
    model_dir(model)
    items = read(f"{DATA_DIR}/items.csv")
    return dict(warehouses=sorted(items.warehouse.unique()),
                clients=[dict(id=i, name=n) for i, n in
                         items[["client_id", "client_name"]].drop_duplicates().sort_values("client_id").values])


@app.get("/api/summary")
def summary(model: str = "lgbm"):
    path = model_dir(model)
    with open(f"{path}/risk_summary.json") as f:
        return json.load(f)


@app.get("/api/volume")
def volume(model: str = "lgbm", warehouse: str | None = None, client: str | None = None):
    """Weekly orders: last 16 full weeks of actuals, then the 13-week forecast (P50 / P90)."""
    path = model_dir(model)
    fc = read(f"{path}/order_volume_forecast.csv", parse_dates=["week_start"])
    demand = read(f"{DATA_DIR}/daily_demand.csv", parse_dates=["date"])
    if warehouse:
        fc, demand = fc[fc.warehouse == warehouse], demand[demand.warehouse == warehouse]
    if client:
        fc, demand = fc[fc.client_id == client], demand[demand.client_id == client]

    first_forecast_week = read(f"{path}/order_volume_forecast.csv", parse_dates=["week_start"]).week_start.min()
    demand = demand[demand.date < first_forecast_week]               # only complete weeks before the forecast
    weekly = (demand.assign(week_start=demand.date.dt.to_period("W-SUN").dt.start_time)
                    .groupby("week_start").orders.sum().tail(HISTORY_WEEKS))
    actual = [dict(week_start=d.date().isoformat(), actual=int(v)) for d, v in weekly.items()]

    f = fc.groupby("week_start")[["orders_p50", "orders_p90"]].sum().round(0)
    forecast = [dict(week_start=d.date().isoformat(), p50=int(r.orders_p50), p90=int(r.orders_p90))
                for d, r in f.iterrows()]
    return dict(actual=actual, forecast=forecast)


@app.get("/api/risk")
def risk(model: str = "lgbm", warehouse: str | None = None, client: str | None = None,
         level: str | None = None):
    """One row per SKU, most urgent first, with item name and client from items.csv."""
    path = model_dir(model)
    df = read(f"{path}/inventory_risk.csv")
    items = read(f"{DATA_DIR}/items.csv")[["sku", "client_name", "category", "description"]]
    df = df.merge(items, on="sku", how="left")
    if warehouse:
        df = df[df.warehouse == warehouse]
    if client:
        df = df[df.client_id == client]
    counts = {lvl: int((df.risk_state == lvl).sum()) for lvl in RISK_LEVELS}   # before the level filter
    if level:
        df = df[df.risk_state == level]
    order = {s: i for i, s in enumerate(RISK_LEVELS)}
    df = df.sort_values(["risk_state", "days_of_cover"],
                        key=lambda c: c.map(order) if c.name == "risk_state" else c)
    df = df.astype(object).where(df.notna(), None)
    return dict(counts=counts, shortfall_risk_skus=counts["High risk"] + counts["Watch"], rows=records(df))


@app.get("/api/forecast-vs-actual")
def forecast_vs_actual(model: str = "lgbm", warehouse: str | None = None, client: str | None = None):
    path = model_dir(model)
    df = read(f"{path}/forecast_vs_actual.csv")
    if warehouse:
        df = df[df.warehouse == warehouse]
    if client:
        df = df[df.client_id == client]
    if df.empty:
        return dict(orders_received=0, orders_forecast=0, variance_pct=0, status="On track", days=[])
    rec, fcst = float(df.orders_received.sum()), float(df.orders_forecast.sum())
    var = round((rec - fcst) / fcst * 100, 1)
    status = "On track" if abs(var) <= 10 else "Watch" if abs(var) <= 20 else "Off track"
    days = (df.groupby("date")[["orders_received", "orders_forecast"]].sum().round(0).reset_index())
    return dict(orders_received=int(rec), orders_forecast=round(fcst), variance_pct=var, status=status,
                days=records(days))
