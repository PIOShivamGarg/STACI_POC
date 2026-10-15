"""
The model to run comes from FORECAST_MODEL, set in .env (or the shell, which takes priority).
"""
import os

from dotenv import load_dotenv

load_dotenv()

# FORECAST_MODEL value -> module in models/ that trains it
MODELS = {"lgbm": "models.lgbm", "xgboost": "models.xgb", "chronos": "models.chronos",
          "stats": "models.stats"}

DATA_DIR = "data"
OUTPUT_ROOT = "outputs"
H = 13                                # forecast horizon (weeks)

MODEL = os.environ.get("FORECAST_MODEL", "lgbm").strip().lower()
if MODEL not in MODELS:
    raise SystemExit(f"FORECAST_MODEL={MODEL!r} is not supported. Choose one of: {', '.join(MODELS)}")

MODEL_MODULE = MODELS[MODEL]
OUT_DIR = f"{OUTPUT_ROOT}/{MODEL}"   # every model writes to its own folder
