import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin
from sklearn.metrics import pairwise_distances

class KMedoids:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.medoids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            labels = self._assign_labels(X)
            new_medoids = self._update_medoids(X, labels)
            
            if np.allclose(self.medoids, new_medoids):
                break
            
            self.medoids = new_medoids
        
        self.labels_ = self._assign_labels(X)
        return self
    
    def _assign_labels(self, X):
        distances = pairwise_distances(X, self.medoids)
        return pairwise_distances_argmin(X, self.medoids)
    
    def _update_medoids(self, X, labels):
        medoids = np.zeros_like(self.medoids)
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            cluster_distances = pairwise_distances(cluster_points, metric='euclidean')
            total_distance = np.sum(cluster_distances, axis=1)
            medoids[i] = cluster_points[np.argmin(total_distance)]
        return medoids
    
    def predict(self, X):
        return self._assign_labels(X)
