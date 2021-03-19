"""Recommendation service layer object module.

This module includes a class representing service related with recommendation featuree.
It links to data layer objects to deliver the functionality.
"""
import numpy as np
import pandas as pd

_DEBUG = False
class RecommendService:
    def __init__(self, rating_data, movie_data, user_data, app_user_data, knn, genre):
        """Constructor, it connects data objects in lower level operation layer.

        Args:
            rating_data: rating dataset module
            movie_data: movie dataset module
            user_data: base user dataset module
            app_user_data: app user dataset module
            knn: knn model layer module
            genre: TFxIDF wrapper module
        """
        self.rating_data = rating_data
        self.movie_data = movie_data
        self.user_data = user_data
        self.app_user_data = app_user_data
        self.knn = knn
        self.genre = genre

    def recommend_rating(self, user_id, movie_id, n_neighbor, user_vector=None):
        """Estimates rating for given user and movie.

        Args:
            user_id: user id (not index of the table)
            movie_id: movie id (not index of the table)
            n_neighbor: number of K value for Knn
            user_vector: custom user rating vector to be used

        Returns:
            Weight averaged rating.
        """
        if _DEBUG: print("user_id:", user_id, ", movie_id:", movie_id, ", n_neighbor:", n_neighbor)
        # it retrieves all the users having rating for the given movie
        user_w_movie = self.rating_data.get_user_ratings_by_movie(movie_id)
        # if this user belong to base user (originated from original dataset), 
        # it gets removed from the list to find other neighbor not self
        if user_id in user_w_movie.index:
            # remove same user data
            user_w_movie = user_w_movie.drop([user_id])
        if _DEBUG: print("user_with_movie_rating:\n", user_w_movie)
        # if resulted list is empty, we can't estimate ratings
        if len(user_w_movie) <= 0:
            # no user to compute rating estimation
            if _DEBUG: print("Found no users have rating, returning 0")
            return 0.0
        # allow custom user vector or read from either base or app user rating dataset
        if user_vector is None:
            df = self.rating_data.get_user_vector(user_id)
        else:
            df = user_vector
        if _DEBUG: print("app user ratings:", df)
        # use Knn module to perform finding nearest users from above list of users
        self.knn.fit(np.mat(user_w_movie), n_neighbor)
        nbrs = self.knn.predict(df)
        if _DEBUG: print("found knn neighbors (distance, index):", nbrs)
        total = 0.0
        total_score = 0.0
        indices = nbrs[1][0]
        # go through found neighbors and calculate weighted average rating value
        for i, index in enumerate(indices):
            score = 1.0 / (1.0 + nbrs[0][0][i])  # convert distance to similarity
            # movie id is index + 1
            rating = user_w_movie.iloc[index][movie_id - 1]
            user_id = user_w_movie.iloc[index].name
            if _DEBUG: print("index:", index, ", user_id:", user_id, ", rating:", rating, ", score:", score)
            # sum up enumerator and denominator separately
            total += score * rating
            total_score += score
        avg_rating = total / total_score
        if _DEBUG: print("estimated rating:", avg_rating)
        return avg_rating

    def recommend_movie_by_genre(self, movie_id, n_recommended_movie):
        """Recommends movies by given movie genres.

        Args:
            movie_id: movie id (not index of the table)
            n_recommended_movie: number of recommended movie

        Returns:
            List of recommended movies.
        """
        # bring the custom built dataset having raw genre text 
        df = self.movie_data.get_genre_dataset()
        self.genre.fit(df)
        movies = self.genre.predict(movie_id, n_recommended_movie)
        print("recommended movies by genre:\n", movies)
        return movies