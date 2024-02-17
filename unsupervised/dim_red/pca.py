import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from scipy.linalg import svd

class CustomPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = check_array(X)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        U, s, Vt = svd(X_centered, full_matrices=False)
        self.components_ = Vt.T
        self.explained_variance_ = s ** 2 / (X.shape[0] - 1)
        self.explained_variance_ratio_ = s / np.sum(s)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        X_centered = X - self.mean_
        if self.n_components is not None:
            return X_centered @ self.components_[:, :self.n_components]
        else:
            return X_centered @ self.components_

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)