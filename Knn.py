#!/usr/bin/python
"""KNN model component module.

This module includes a Knn class representing KNN algorith wrapper object,
which is providing convenient fit & predict standard method to the clients.
"""
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier


class Knn:
    def __init__(self, sim_metric="cosine"):
        """Distance metric specification used for KNN libarary."""
        self.sim_metric = sim_metric
        
    def fit(self, data_mat, K = 10):
        """Fits a dataset for KNN algorithm with K value.

        Args:
            data_mat: user x movie rating matrix
            K: number of K value for Knn
        """
        self.data_mat = data_mat
        self.num_item = data_mat.shape[1]
        self.num_user = data_mat.shape[0]
        self.K = K
        # if rating matrix doesn't have enough datapoints, we lower down K valuee
        K = min(K, self.num_user)
        self.knn = NearestNeighbors(
            metric=self.sim_metric, algorithm="brute", n_neighbors=K, n_jobs=1)
        self.knn.fit(data_mat)
    
    def predict(self, movie_data):
        """Estimates rating for given movie.

        Args:
            movie_data: data points to be used for KNN to find neighbors

        Returns:
            Tuple list for closest K neighbor's distance and index within rating matrix
        """
        neighbors = self.knn.kneighbors(
            movie_data,
            return_distance=True)
        return neighbors