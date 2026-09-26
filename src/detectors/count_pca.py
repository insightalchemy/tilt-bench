"""
Statistical detector: count-vector PCA (Xu et al., SOSP 2009).

Fits PCA on template count vectors from normal training windows and scores a window by its
squared reconstruction error; higher = more anomalous.
"""

from sklearn.decomposition import PCA

N_COMPONENTS = 20


class CountPCADetector:
    def __init__(self, n_components: int = N_COMPONENTS, random_state: int = 0):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None

    def fit(self, X_train_normal):
        n_components = min(self.n_components, X_train_normal.shape[0] - 1, X_train_normal.shape[1])
        # Exact SVD: deterministic, no randomized-solver seed sensitivity.
        self.pca = PCA(n_components=n_components, svd_solver="full", random_state=self.random_state)
        self.pca.fit(X_train_normal)
        return self

    def score(self, X):
        X_recon = self.pca.inverse_transform(self.pca.transform(X))
        return ((X - X_recon) ** 2).sum(axis=1)
