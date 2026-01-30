import numpy as np

def sharpe_ratio(returns, annualization=252):
    returns = returns.dropna()
    if returns.std() == 0:
        return np.nan
    return np.sqrt(annualization) * returns.mean() / returns.std()
