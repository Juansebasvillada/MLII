import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

class CustomSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=None):
        self.n_components = n_components

    def fit(self, X, y=None):
        X = check_array(X)
        self.U_, self.s_, self.Vt_ = np.linalg.svd(X, full_matrices=False)
        self.explained_variance_ = self.s_ ** 2 / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.s_ / np.sum(self.s_)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        if self.n_components is not None:
            return X @ self.Vt_[:self.n_components, :].T
        else:
            return X @ self.Vt_.T

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

