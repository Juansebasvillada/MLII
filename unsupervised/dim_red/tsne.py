import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.manifold import TSNE

class CustomTSNE(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, perplexity=30.0, learning_rate=200.0, n_iter=1000):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y=None):
        X = check_array(X)
        self.embedding_ = self._tsne(X, self.n_components, self.perplexity, self.learning_rate, self.n_iter)
        return self

    def transform(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return self.embedding_

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def _tsne(self, X, n_components, perplexity, learning_rate, n_iter):
        # Implementación de t-SNE usando sklearn.manifold.TSNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter)
        embedding = tsne.fit_transform(X)
        return embedding