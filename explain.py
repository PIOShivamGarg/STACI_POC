"""Staci - English explanations of at-risk SKUs (LLM used ONLY to explain, never to predict).

Reads inventory_risk.csv (risk_probability and top_drivers come from the ML model in ml_risk.py) and asks an
LLM to put each at-risk SKU, and the whole portfolio, into words for a warehouse planner.
  - The LLM never sees a request to predict, and never changes a number: it is given the facts and told to use only them.
  - Every number in its answer must appear in the facts, otherwise we fall back to a template sentence.
  - Without AZURE_OPENAI_API_KEY (set it in .env) the template text is used, so the UI always has an explanation.
  - Answers are cached by their facts, so re-running costs nothing unless the facts change.

  python explain.py     ->  outputs/ml_risk/explanations.json
"""
import hashlib
import json
import os
import re
from collections import Counter

import pandas as pd
from dotenv import load_dotenv

from config import OUT_DIR, OUTPUT_ROOT

load_dotenv()
EXPLAIN_FILE = f"{OUTPUT_ROOT}/ml_risk/explanations.json"
EXPLAIN_LEVELS = ("High risk", "Watch")
MODEL = os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini-imsai")  # For Azure, this should be the deployment name

SYSTEM = ("You explain stock-shortage risk to a warehouse planner in English. "
          "Use ONLY the facts you are given. Never invent numbers, dates or causes, and never change the probability. "
          "Be short and concrete. No jargon, no hedging, no recommendations beyond the facts.")


def facts_for(r) -> dict:
    """The model-independent facts about one SKU (same for every forecasting model)."""
    return dict(item=r.description, client=r.client_name, warehouse=r.warehouse, risk_level=r.risk_state,
                chance_of_running_out_in_30_days=f"{r.risk_probability:.0%}", units_available=int(r.available_qty),
                main_drivers=r.top_drivers if isinstance(r.top_drivers, str) else "",
                inbound_shipment=(f"{r.covered_by_asn} due {r.next_asn_date}" if isinstance(r.covered_by_asn, str)
                                  else "none"))


def template_text(f: dict) -> str:
    drivers = f"Main reasons: {f['main_drivers']}." if f["main_drivers"] else ""
    return (f"{f['item']} ({f['warehouse']}) has a {f['chance_of_running_out_in_30_days']} chance of running out "
            f"within 30 days. {drivers}").strip()


def numbers(text: str) -> set:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*\.?\d*", text)}


def grounded(answer: str, facts: str) -> bool:
    """True if every number in the answer is also in the facts."""
    return numbers(answer) <= numbers(facts)


def ask(client, prompt: str) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    try:
        resp = client.chat.completions.create(model=MODEL, temperature=0.2, max_tokens=220, messages=messages)
    except Exception as e:
        # some deployments (e.g. newer reasoning models) reject temperature / max_tokens: retry with defaults
        if type(e).__name__ != "BadRequestError":
            raise
        resp = client.chat.completions.create(model=MODEL, messages=messages)
    return resp.choices[0].message.content.strip()


def get_client():
    """Azure OpenAI if AZURE_OPENAI_ENDPOINT is set (AZURE_OPENAI_DEPLOYMENT_NAME = the DEPLOYMENT name), else OpenAI."""
    key = (os.environ.get("AZURE_OPENAI_API_KEY")).strip()
    if not key:
        return None
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
    if endpoint:
        from openai import AzureOpenAI
        version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21").strip()
        print(f"Using Azure OpenAI: {endpoint} (deployment {MODEL}, api-version {version})")
        return AzureOpenAI(api_key=key, azure_endpoint=endpoint, api_version=version)
    from openai import OpenAI
    print("Using OpenAI (api.openai.com). For an Azure deployment, set AZURE_OPENAI_ENDPOINT in .env")
    return OpenAI(api_key=key)


def digest(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def portfolio_facts(df: pd.DataFrame) -> dict:
    at_risk = df[df.risk_state.isin(EXPLAIN_LEVELS)]
    drivers = Counter(d.strip() for s in at_risk.top_drivers.dropna() for d in s.split(";"))
    kinds = Counter(re.sub(r"[\d.,]+", "#", d) for d in drivers.elements())
    return dict(skus_at_risk=len(at_risk), high_risk=int((df.risk_state == "High risk").sum()),
                watch=int((df.risk_state == "Watch").sum()),
                by_warehouse=at_risk.warehouse.value_counts().to_dict(),
                by_client=at_risk.client_name.value_counts().to_dict(),
                most_common_drivers=[k for k, _ in kinds.most_common(4)])


def main():
    path = f"{OUT_DIR}/inventory_risk.csv"
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run risk.py first")
    risk = pd.read_csv(path)
    if "risk_probability" not in risk:
        raise SystemExit("risk_probability not in inventory_risk.csv - run stock_sim.py, ml_risk.py and risk.py (RISK_METHOD=ml)")
    items = pd.read_csv("data/items.csv")[["sku", "client_name", "description"]]
    df = risk.merge(items, on="sku")

    cache = json.load(open(EXPLAIN_FILE)) if os.path.exists(EXPLAIN_FILE) else {}
    old = cache.get("skus", {})
    client = get_client()
    if client is None:
        print("AZURE_OPENAI_API_KEY not set in .env - using template explanations")

    out = {}
    for r in df[df.risk_state.isin(EXPLAIN_LEVELS)].itertuples():
        f = facts_for(r)
        h = digest(f)
        if r.sku in old and old[r.sku]["facts_hash"] == h and (client is None or old[r.sku]["source"] != "template"):
            out[r.sku] = old[r.sku]
            continue
        text, source = template_text(f), "template"
        if client is not None:
            try:
                facts = json.dumps(f)
                answer = ask(client, f"Facts: {facts}\n\nIn 1-2 short sentences, say why this item is at risk.")
                if grounded(answer, facts):
                    text, source = answer, MODEL
                else:
                    print(f"{r.sku}: answer used numbers that are not in the facts - template used")
            except Exception as e:   # network, quota, bad key: never break the pipeline
                print(f"{r.sku}: LLM call failed ({type(e).__name__}: {str(e)[:150]}) - template used")
        out[r.sku] = dict(text=text, source=source, facts_hash=h)

    pf = portfolio_facts(df)
    ph = digest(pf)
    port = cache.get("portfolio", {})
    if not (port.get("facts_hash") == ph and (client is None or port.get("source") != "template")):
        text = (f"{pf['skus_at_risk']} SKUs are at shortfall risk ({pf['high_risk']} high risk, {pf['watch']} watch).")
        source = "template"
        if client is not None:
            try:
                facts = json.dumps(pf)
                answer = ask(client, f"Facts: {facts}\n\nSummarise the pattern across these SKUs in 2-3 short bullet points "
                                     "(which warehouses or clients stand out, what keeps coming up).")
                if grounded(answer, facts):
                    text, source = answer, MODEL
                else:
                    print("portfolio: answer used numbers that are not in the facts - template used")
            except Exception as e:
                print(f"portfolio: LLM call failed ({type(e).__name__}: {str(e)[:150]}) - template used")
        port = dict(text=text, source=source, facts_hash=ph)

    os.makedirs(os.path.dirname(EXPLAIN_FILE), exist_ok=True)
    with open(EXPLAIN_FILE, "w") as fh:
        json.dump(dict(generated_at=str(pd.Timestamp.now().date()), skus=out, portfolio=port), fh, indent=2)
    print(f"{len(out)} SKU explanations ({Counter(v['source'] for v in out.values())}) -> {EXPLAIN_FILE}")


if __name__ == "__main__":
    main()
