"""
Detector 2 (classical ML family): isolation forest over the same window x template count vectors
used by count_pca.py.
"""

from sklearn.ensemble import IsolationForest

N_ESTIMATORS = 100  # explicit, easily-changeable


class IsolationForestCountsDetector:
    def __init__(self, n_estimators: int = N_ESTIMATORS, random_state: int = 0):
        self.model = IsolationForest(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)

    def fit(self, X_train_normal):
        self.model.fit(X_train_normal)
        return self

    def score(self, X):
        # score_samples: higher = more normal. Flip sign so higher = more anomalous, matching
        # count_pca's reconstruction-error convention.
        return -self.model.score_samples(X)
