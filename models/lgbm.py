"""LightGBM: P50 with a tweedie objective, P90 with a 0.9 quantile objective."""
import lightgbm as lgb

METHOD_VERSION = "lightgbm-v0.1"

# P50 = likely figure (tweedie suits non-negative, spiky demand); P90 = safer, higher figure
P50_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.2, learning_rate=0.05,
                  num_leaves=31, min_data_in_leaf=30, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, n_estimators=600, verbose=-1, seed=42)
P90_PARAMS = {k: v for k, v in P50_PARAMS.items() if k != "tweedie_variance_power"}
P90_PARAMS.update(objective="quantile", alpha=0.9)


def train(X, y):
    m50 = lgb.LGBMRegressor(**P50_PARAMS).fit(X, y)
    m90 = lgb.LGBMRegressor(**P90_PARAMS).fit(X, y)
    return m50, m90


def feature_importance(model, names):
    return model.booster_.feature_importance("gain")
