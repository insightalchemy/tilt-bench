"""
Detector 1 (statistical family): count-vector + PCA reconstruction error.

Fits PCA on template count-vectors from windows in the (assumed-clean) training period, then
scores any window by its PCA reconstruction error -- a window whose count profile can't be
explained by the normal subspace gets a high score. Higher score = more anomalous, consistent
with the other two detectors' scoring convention.
"""

from sklearn.decomposition import PCA

N_COMPONENTS = 20  # explicit, easily-changeable


class CountPCADetector:
    def __init__(self, n_components: int = N_COMPONENTS, random_state: int = 0):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None

    def fit(self, X_train_normal):
        n_components = min(self.n_components, X_train_normal.shape[0] - 1, X_train_normal.shape[1])
        # svd_solver="full" -- deterministic exact SVD, no randomized-solver seed sensitivity
        self.pca = PCA(n_components=n_components, svd_solver="full", random_state=self.random_state)
        self.pca.fit(X_train_normal)
        return self

    def score(self, X):
        X_recon = self.pca.inverse_transform(self.pca.transform(X))
        return ((X - X_recon) ** 2).sum(axis=1)
