"""Staci 8.2 POC - compare every model that has been run, side by side, per deliverable.

Reads outputs/<model>/summary.json and risk_summary.json for each model in config.MODELS
(models not run yet are skipped) and writes outputs/model_comparison.csv:
one row per deliverable x measure x backtest window, one column per model, plus the naive benchmark.
"""
import json
import os

import pandas as pd

from config import MODELS, OUTPUT_ROOT

WINDOWS = {"last_13_weeks": "Last 13 weeks", "peak_season_last_year": "Last peak season"}
DELIVERABLES = {   # summary.json section -> (deliverable, what is compared)
    "order_volume": ("1. Order Volume", "orders per client / warehouse / channel / week"),
    "shortfall_risk": ("2. Shortfall Risk", "demand during each SKU's lead time (proxy)"),
    "sku_demand": ("Base: SKU demand", "units per SKU per week"),
}
MEASURES = {"model": ("WAPE (error)", "lower"), "p90_coverage": ("P90 coverage", "close to 90%")}
RISK_STATES = ["High risk", "Watch", "Monitor", "Healthy", "Possible overstock"]


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main():
    runs = {m: load_json(f"{OUTPUT_ROOT}/{m}/summary.json") for m in MODELS}
    runs = {m: s for m, s in runs.items() if s}
    if not runs:
        raise SystemExit(f"No results found under {OUTPUT_ROOT}/ - run forecast.py first")
    any_run = next(iter(runs.values()))

    rows = []
    for section, (deliverable, compared) in DELIVERABLES.items():
        for key, (measure, better) in MEASURES.items():
            for window, window_name in WINDOWS.items():
                row = {"deliverable": deliverable, "compared": compared, "measure": measure,
                       "window": window_name, "better_is": better}
                row.update({m: s["backtest"][window][section][key] for m, s in runs.items()})
                # naive benchmark is the same for every model; it has no P90
                row["naive"] = any_run["backtest"][window][section]["naive"] if key == "model" else None
                rows.append(row)

    for state in RISK_STATES:   # live alerts raised today - a count, not an accuracy score
        row = {"deliverable": "2. Shortfall Risk", "compared": "live alerts at the as-of date",
               "measure": f"{state} items", "window": "Today", "better_is": "-"}
        for m in runs:
            risk = load_json(f"{OUTPUT_ROOT}/{m}/risk_summary.json")
            row[m] = risk["risk_counts"].get(state, 0) if risk else None
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(f"{OUTPUT_ROOT}/model_comparison.csv", index=False)

    shown = table.drop(columns=["compared"]).copy()
    pct = shown.measure.isin([m for m, _ in MEASURES.values()])
    for col in list(runs) + ["naive"]:
        shown[col] = [f"{v:.1%}" if p and pd.notna(v) else ("" if pd.isna(v) else int(v))
                      for v, p in zip(shown[col], pct)]
    print(shown.to_string(index=False))
    skipped = [m for m in MODELS if m not in runs]
    if skipped:
        print(f"\nNot run yet: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
