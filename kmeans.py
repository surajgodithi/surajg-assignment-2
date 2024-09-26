import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters=3, init_method='random', max_iter=300):
        self.n_clusters = n_clusters
        self.init_method = init_method
        self.max_iter = max_iter
        self.centroids = None
        self.history = []

    def fit(self, X, manual_centroids=None):
        self.X = X
        self.n_samples, self.n_features = X.shape

        if self.init_method == 'manual':
            if manual_centroids is None:
                raise ValueError("Manual centroids must be provided for manual initialization.")
            self.centroids = np.array(manual_centroids)
        elif self.init_method == 'random':
            self.centroids = self._init_random()
        elif self.init_method == 'kmeans++':
            self.centroids = self._init_kmeans_pp()
        elif self.init_method == 'farthest_first':
            self.centroids = self._init_farthest_first()
        else:
            raise ValueError(f"Unknown initialization method {self.init_method}")

        self.history.append(self.centroids.copy())

        for i in range(self.max_iter):
            labels = self._assign_clusters()
            new_centroids = self._calculate_new_centroids(labels)
            if np.all(new_centroids == self.centroids):
                break

            self.centroids = new_centroids
            self.history.append(self.centroids.copy())

        return labels

    def _init_random(self):
        random_idxs = np.random.choice(self.n_samples, self.n_clusters, replace=False)
        return self.X[random_idxs]

    def _init_farthest_first(self):
        centroids = []
        centroids.append(self.X[np.random.choice(self.n_samples)])
        for _ in range(1, self.n_clusters):
            distances = np.array([min([norm(x - c) for c in centroids]) for x in self.X])
            next_centroid = self.X[np.argmax(distances)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def _init_kmeans_pp(self):
        centroids = []
        centroids.append(self.X[np.random.choice(self.n_samples)])
        for _ in range(1, self.n_clusters):
            distances = np.array([min([norm(x - c)**2 for c in centroids]) for x in self.X])
            prob_distribution = distances / distances.sum()
            next_centroid = self.X[np.random.choice(self.n_samples, p=prob_distribution)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def _assign_clusters(self):
        distances = np.array([[norm(x - centroid) for centroid in self.centroids] for x in self.X])
        return np.argmin(distances, axis=1)

    def _calculate_new_centroids(self, labels):
        new_centroids = np.array([self.X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def predict(self, X):
        distances = np.array([[norm(x - centroid) for centroid in self.centroids] for x in X])
        return np.argmin(distances, axis=1)


def generate_data(n_samples=300, n_features=2):
    X, _ = make_blobs(n_samples=n_samples, centers=3, n_features=n_features, random_state=42)
    return X
