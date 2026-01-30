class StrategyFamily:
    def __init__(self, param_grid):
        self.param_grid = param_grid

    def run(self, price, params, cost=0.0):
        raise NotImplementedError

import pandas as pd

class MACrossoverFamily(StrategyFamily):
    def run(self, price, params, cost=0.0):
        short = params["short"]
        long = params["long"]

        ma_s = price.rolling(short).mean()
        ma_l = price.rolling(long).mean()

        signal = (ma_s > ma_l).astype(int)
        returns = price.pct_change().shift(-1)
        trades = signal.diff().abs()

        strat_ret = signal * returns - cost * trades
        return strat_ret.dropna()
