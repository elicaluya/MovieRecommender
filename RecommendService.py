import numpy as np

class RecommendService:
    def __init__(self, rating_data, movie_data, user_data, app_user_data, knn, genre):
        self.rating_data = rating_data
        self.movie_data = movie_data
        self.user_data = user_data
        self.app_user_data = app_user_data
        self.knn = knn
        self.genre = genre

    def recommend_rating(self, user_id, movie_id, n_neighbor):
        """Estimates rating for given user and movie.

        Args:
            user_id: user id (not index of the table)
            movie_id: movie id (not index of the table)
            n_neighbor: number of K value for Knn

        Returns:
            Weight averaged rating.
        """
        user_w_movie = self.rating_data.get_user_ratings_by_movie(movie_id)
        df = self.rating_data.get_user_vector(user_id)
        print(df)
        self.knn.fit(np.mat(user_w_movie), n_neighbor)
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
        avg_rating = total / total_score
        print("expected rating ===>", avg_rating)
        return avg_rating

    def recommend_movie_by_genre(self, movie_title, n_recommended_movie):
        """Recommends movies by given movie genres.

        Args:
            movie_title: movie title (full length)
            n_recommended_movie: number of recommended movie

        Returns:
            List of recommended movies.
        """
        df = self.movie_data.get_genre_dataset()
        self.genre.fit(df)
        movies = self.genre.predict(movie_title, n_recommended_movie)
        print("recommended movies by genre:\n", movies)
        return movies