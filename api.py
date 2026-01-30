import numpy as np
import pandas as pd
from .cv import combinatorial_cv
from .metrics import sharpe_ratio

class OverfittingReport:
    def __init__(self, lambdas, asset_name=None):
        self.lambdas = np.array(lambdas)
        self.asset_name = asset_name
        self.pbo = (self.lambdas < 0).mean()

    def risk_level(self):
        if self.pbo < 0.3:
            return "LOW"
        elif self.pbo < 0.5:
            return "MEDIUM"
        else:
            return "HIGH"

    def summary(self):
        return {
            "asset": self.asset_name,
            "PBO": round(self.pbo, 3),
            "lambda_mean": round(self.lambdas.mean(), 3),
            "lambda_median": round(np.median(self.lambdas), 3),
            "risk_level": self.risk_level()
        }


def estimate_overfitting_risk(
    strategy_family,
    selection_rule,
    asset=None,
    price=None,
    metric=sharpe_ratio,
    cost=0.0,
    S=8,
    k=4
):
    if price is None:
        if asset is None:
            raise ValueError("Provide either price or asset")
        
        price = asset.load_price()
        asset_name = getattr(asset, "ticker", None)
        if isinstance(price, pd.DataFrame):
            if price.shape[1]!=1:
                raise ValueError("Price DataFrame must have exactly one column")
            price=price.iloc[:,0]

    else:
        asset_name = None

    lambdas = []

    for train_price, test_price in combinatorial_cv(price, S, k):

        train_scores = []
        test_scores  = []

        for params in strategy_family.param_grid:
            train_ret = strategy_family.run(train_price, params, cost)
            test_ret  = strategy_family.run(test_price,  params, cost)

            train_scores.append(metric(train_ret))
            test_scores.append(metric(test_ret))

        train_scores = np.array(train_scores)
        test_scores  = np.array(test_scores)

        # Skip pathological splits
        if np.all(np.isnan(train_scores)):
            continue

        selected_idx = selection_rule.select(train_scores)

        # Aggregate OOS performance of selected strategies
        agg_test_score = np.nanmedian(test_scores[selected_idx])

        rank = np.sum(test_scores < agg_test_score)
        denom = max(1, len(test_scores) - 1)
        rank_pct = rank / denom

        lambdas.append(
            np.log((rank_pct + 1e-8) / (1 - rank_pct + 1e-8))
        )

    return OverfittingReport(lambdas, asset_name=asset_name)
