"""XGBoost: P50 with a tweedie objective, P90 with a 0.9 quantile objective.

Settings mirror models/lgbm.py where XGBoost has an equivalent, so the comparison is fair:
leaf-wise trees capped at 31 leaves, 0.8 row / column sampling, 600 trees at learning rate 0.05.
"""
import xgboost as xgb

METHOD_VERSION = "xgboost-v0.1"

P50_PARAMS = dict(objective="reg:tweedie", tweedie_variance_power=1.2, learning_rate=0.05,
                  grow_policy="lossguide", max_leaves=31, max_depth=0,
                  subsample=0.8, colsample_bytree=0.8, n_estimators=600,
                  tree_method="hist", enable_categorical=True, random_state=42)
P90_PARAMS = {k: v for k, v in P50_PARAMS.items() if k != "tweedie_variance_power"}
P90_PARAMS.update(objective="reg:quantileerror", quantile_alpha=0.9)


def train(X, y):
    m50 = xgb.XGBRegressor(**P50_PARAMS).fit(X, y)
    m90 = xgb.XGBRegressor(**P90_PARAMS).fit(X, y)
    return m50, m90


def feature_importance(model, names):
    gain = model.get_booster().get_score(importance_type="total_gain")   # features never used are left out
    return [gain.get(name, 0.0) for name in names]
