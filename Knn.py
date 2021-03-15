#!/usr/bin/python

from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


class Knn:
    def __init__(self, sim_metric="cosine"):
        self.sim_metric = sim_metric
        
    def fit(self, data_mat, K = 10):
        self.data_mat = data_mat
        self.num_item = data_mat.shape[1]
        self.num_user = data_mat.shape[0]
        self.K = K
        K = min(K, self.num_user)
        self.knn = NearestNeighbors(
            metric=self.sim_metric, algorithm="brute", n_neighbors=K, n_jobs=1)
        self.knn.fit(data_mat)
    
    def predict(self, movie_data):
        neighbors = self.knn.kneighbors(
            movie_data,
            return_distance=True)
        return neighbors