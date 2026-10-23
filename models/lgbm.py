"""LightGBM: P50 with a tweedie objective, P90 with a 0.9 quantile objective."""
import lightgbm as lgb

METHOD_VERSION = "lightgbm-v0.2"

# P50 = likely figure (tweedie suits non-negative, spiky demand); P90 = safer, higher figure.
# Tuned on 4 backtest windows: smaller trees (15 leaves, 300 rounds) beat 31 / 63 leaves and 600-1000 rounds;
# tweedie power 1.1 / 1.5 and a median (0.5 quantile) objective for P50 were no better than tweedie 1.2.
P50_PARAMS = dict(objective="tweedie", tweedie_variance_power=1.2, learning_rate=0.05,
                  num_leaves=15, min_data_in_leaf=30, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, n_estimators=300, verbose=-1, seed=42)
P90_PARAMS = {k: v for k, v in P50_PARAMS.items() if k != "tweedie_variance_power"}
P90_PARAMS.update(objective="quantile", alpha=0.9)


def train(X, y, weight=None):
    """y is demand as a ratio of each SKU's recent level; weight = that level, so big SKUs count in units."""
    m50 = lgb.LGBMRegressor(**P50_PARAMS).fit(X, y, sample_weight=weight)
    m90 = lgb.LGBMRegressor(**P90_PARAMS).fit(X, y, sample_weight=weight)
    return m50, m90


def feature_importance(model, names):
    return model.booster_.feature_importance("gain")
