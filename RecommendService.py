import numpy as np

class RecommendService:
    def __init__(self, rating_data, movie_data, user_data, app_user_data, knn):
        self.rating_data = rating_data
        self.movie_data = movie_data
        self.user_data = user_data
        self.app_user_data = app_user_data
        self.knn = knn

    def recommend_rating(self, user_id, movie_id):
        """Estimates rating for given user and movie.

        Args:
            user_id: user id (not index of the table)
            movie_id: movie id (not index of the table)

        Returns:
            none. it prints out estimation as of now.
        """
        user_w_movie = self.rating_data.get_user_ratings_by_movie(movie_id)
        df = self.rating_data.get_user_vector(user_id)
        print(df)
        self.knn.fit(np.mat(user_w_movie))
        nbrs = self.knn.predict(df)
        total = 0.0
        total_score = 0.0
        indices = nbrs[1][0]
        for i, index in enumerate(indices):
            score = 1.0 - nbrs[0][0][i]  # 1 - distance
            rating = user_w_movie.iloc[index][movie_id - 1]
            user_id = user_w_movie.iloc[index].name
            print(i, "index:", index, "user_id:", user_id, "rating:", rating, "score:", score)
            #self.print_ratings_by_user(user_id)
            total += score * rating
            total_score += score
        print("expected rating ===>", total / total_score)