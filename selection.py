import numpy as np

class SelectionRule:
    def select(self, train_scores):
        raise NotImplementedError

class TopOneSelection(SelectionRule):
    def select(self, train_scores):
        return [int(np.nanargmax(train_scores))]

class TopKSelection(SelectionRule):
    def __init__(self, k_pct):
        self.k_pct = k_pct

    def select(self, train_scores):
        n = len(train_scores)
        k = max(1, int(self.k_pct * n))
        return np.argsort(train_scores)[::-1][:k]
