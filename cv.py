import numpy as np
import pandas as pd
from itertools import combinations

def combinatorial_cv(price, S=8, k=4):
    # price MUST be a pandas Series
    if not isinstance(price, pd.Series):
        raise TypeError("price must be a pandas Series")

    n = len(price)
    edges = np.linspace(0, n, S + 1, dtype=int)

    # IMPORTANT: keep blocks as pandas Series
    blocks = [
        price.iloc[edges[i]:edges[i+1]]
        for i in range(S)
    ]

    idx = range(S)

    for train_idx in combinations(idx, k):
        test_idx = [i for i in idx if i not in train_idx]

        train_price = pd.concat([blocks[i] for i in train_idx])
        test_price  = pd.concat([blocks[i] for i in test_idx])

        yield train_price, test_price
