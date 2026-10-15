"""Synthetic dataset for the Staci 8.2 POC: order-volume forecast + inventory-shortfall early warning.
All data is fictional. Swap these CSVs for real Staci extracts with the same columns."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

START, ASOF = pd.Timestamp("2023-10-01"), pd.Timestamp("2026-10-12")
dates = pd.date_range(START, ASOF, freq="D")

clients = {"CL01": "Northwind Apparel", "CL02": "Contoso Health", "CL03": "Fabrikam Home"}
warehouses = ["Avondale", "Dallas", "Cranbury", "Bolingbrook"]
channels = {"CL01": ["B2C", "Retail"], "CL02": ["B2B"], "CL03": ["B2C", "B2B"]}
categories = {"CL01": ["Tops", "Outerwear", "Accessories"],
              "CL02": ["Supplements", "Devices"],
              "CL03": ["Kitchen", "Decor", "Bedding"]}
peak = {"CL01": 1.9, "CL02": 1.1, "CL03": 1.6}  # Q4 peak strength by client

# ---------- items ----------
items = []
n = 0
for c in clients:
    for i in range(20):
        n += 1
        wh = warehouses[(i + list(clients).index(c)) % 4]
        base = rng.lognormal(mean=2.6, sigma=0.9)          # avg units/day
        intermittent = rng.random() < 0.15
        if intermittent: base = rng.uniform(0.3, 1.5)
        items.append(dict(
            sku=f"{c}-SKU{i+1:03d}", client_id=c, client_name=clients[c], warehouse=wh,
            channel=channels[c][i % len(channels[c])],
            category=categories[c][i % len(categories[c])],
            description=f"{categories[c][i % len(categories[c])]} item {i+1}",
            supplier_lead_time_days=int(rng.choice([7, 10, 14, 21, 28, 35])),
            lead_time_std_days=int(rng.choice([1, 2, 3, 5])),
            safety_stock_days=int(rng.choice([5, 7, 10])),
            unit_cost=round(float(rng.uniform(3, 120)), 2),
            avg_units_per_order=round(float(rng.uniform(1.1, 3.5) if channels[c][i % len(channels[c])]=="B2C" else rng.uniform(4, 25)), 2),
            _base=base, _trend=rng.normal(0.15, 0.25), _intermittent=intermittent))
items = pd.DataFrame(items)

# ---------- events (promos / peaks) ----------
events = []
for y in [2023, 2024, 2025, 2026]:
    bf = pd.Timestamp(f"{y}-11-01") + pd.offsets.Week(weekday=3) * 4 + pd.Timedelta(days=1)  # Black Friday
    events += [dict(event_id=f"BF{y}", client_id="ALL", name="Black Friday / Cyber Week",
                    start_date=bf, end_date=bf + pd.Timedelta(days=4), uplift=1.8),
               dict(event_id=f"PD{y}", client_id="CL01", name="Client summer sale",
                    start_date=pd.Timestamp(f"{y}-07-10"), end_date=pd.Timestamp(f"{y}-07-16"), uplift=1.5),
               dict(event_id=f"NY{y}", client_id="CL02", name="New-year wellness push",
                    start_date=pd.Timestamp(f"{y}-01-02"), end_date=pd.Timestamp(f"{y}-01-15"), uplift=1.4)]
events = pd.DataFrame(events)

def event_mult(client, d):
    m = np.ones(len(d))
    for _, e in events[(events.client_id.isin([client, "ALL"]))].iterrows():
        m[(d >= e.start_date) & (d <= e.end_date)] *= e.uplift
    return m

# ---------- daily demand ----------
doy = dates.dayofyear.values
t = (dates - START).days.values / 365
rows = []
for _, it in items.iterrows():
    seas = 1 + (peak[it.client_id] - 1) * np.exp(-((doy - 340) / 25.0) ** 2)  # Q4 bump
    seas *= 1 + 0.08 * np.sin(2 * np.pi * doy / 365)
    dow = dates.dayofweek.values
    if it.channel == "B2B": wk = np.where(dow >= 5, 0.25, 1.2)
    else: wk = np.where(dow == 0, 1.25, np.where(dow >= 5, 0.85, 1.0))
    mu = it._base * seas * wk * (1 + it._trend * t) * event_mult(it.client_id, dates)
    mu = np.clip(mu, 0.01, None)
    units = rng.negative_binomial(3, 3 / (3 + mu))       # overdispersed noise
    if it._intermittent: units = units * (rng.random(len(dates)) < 0.35)
    orders = np.minimum(units, np.ceil(units / it.avg_units_per_order).astype(int))
    rows.append(pd.DataFrame(dict(date=dates, client_id=it.client_id, warehouse=it.warehouse,
                                  channel=it.channel, sku=it.sku, orders=orders, units_shipped=units)))
demand = pd.concat(rows)
demand = demand[demand.units_shipped > 0]

# ---------- inventory snapshot + inbound ASNs (with deliberate risk cases) ----------
recent = demand[demand.date > ASOF - pd.Timedelta(days=28)].groupby("sku").units_shipped.sum() / 28
inv, asn = [], []
k = 0
scen = rng.permutation(np.array([0]*30 + [1]*12 + [2]*6 + [3]*6 + [4]*6))  # mix of risk cases
for idx, it in items.iterrows():
    r = float(recent.get(it.sku, 0.2))
    lt = it.supplier_lead_time_days
    scenario = int(scen[idx])   # 0 healthy, 1 tight, 2 will stock out before ASN, 3 late ASN, 4 overstock
    cover = {0: lt + 40, 1: lt + 12, 2: lt * 0.4, 3: lt * 0.8, 4: 200}[scenario]
    on_hand = int(max(0, r * cover * rng.uniform(0.9, 1.1)))
    allocated = int(on_hand * rng.uniform(0.02, 0.08))
    inv.append(dict(snapshot_date=ASOF, sku=it.sku, warehouse=it.warehouse,
                    on_hand_qty=on_hand, allocated_qty=allocated, damaged_hold_qty=int(min(on_hand * 0.02, rng.integers(0, 5)))))
    if scenario != 4:
        k += 1
        eta = {0: 12, 1: lt - 2, 2: lt + 4, 3: lt + 10}[scenario]
        asn.append(dict(asn_id=f"ASN{k:05d}", sku=it.sku, warehouse=it.warehouse,
                        created_date=ASOF - pd.Timedelta(days=int(rng.integers(2, 20))),
                        expected_arrival_date=ASOF + pd.Timedelta(days=int(eta)),
                        expected_qty=int(max(10, r * 70)), status="In Transit" if eta < lt else "Open"))
inv, asn = pd.DataFrame(inv), pd.DataFrame(asn)

items.drop(columns=[c for c in items if c.startswith("_")]).to_csv("data/items.csv", index=False)
demand.to_csv("data/daily_demand.csv", index=False, date_format="%Y-%m-%d")
inv.to_csv("data/inventory_snapshot.csv", index=False, date_format="%Y-%m-%d")
asn.to_csv("data/inbound_asn.csv", index=False, date_format="%Y-%m-%d")
events.to_csv("data/events.csv", index=False, date_format="%Y-%m-%d")
print(len(items), "items |", len(demand), "demand rows |", len(asn), "ASNs")
