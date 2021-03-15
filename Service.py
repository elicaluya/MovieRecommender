class MovieService:
    def __init__(self, movie_service):
        self.movie_service = movie_service

    def recommend_movie_by_genre(self, genres):
        return

    def get_most_watched_movie(self, user_id, count):
        """Finds most watched(rated) movies of specified counts and not rated by the given user.

        Args:
            user_id: user id (not index of the table)
            count: number of movies to get

        Returns:
            A dataframe of movies list.
        """
        return self.movie_service.get_most_watched_movies(user_id, count)

    def get_movie_by_title(self, title):
        """Finds specific movie in the movie dataset by given title.

        Args:
            title: movie title (must be exact match as of now)

        Returns:
            A Movie object containing attributes from the movie dataset.
        """
        return self.movie_service.get_movie_by_title(title)


class RatingService:
    def __init__(self, rating_service):
        self.rating_service = rating_service

    def recommend_movie_rating(self, movie_id, user_id):
        return

    def save_app_data(self):
        self.rating_service.save_app_data()

    def get_valid_user_ratings(self, user_id):
        """Finds valid (> 0) ratings for given user.

        Args:
            user_id: unique user id

        Returns:
            A data frame for user_id, item_id, rating
        """
        return self.rating_service.get_valid_user_ratings(user_id)

    def add_user_rating(self, user_id, movie_id, rating):
        """Adds a rating input from user into app dataset.

        Args:
            user_id: user id (not index of the table)
            movie_id: movie id (not index of the table)
            rating: rating (0 - 5)
        """
        self.rating_service.add_user_rating(user_id, movie_id, rating)

    def get_average_ratings_of_movies(self, movie_ids):
        """Calculates a mean ratings for given movies.

        Args:
            movie_ids: list of movie ids
        
        Returns:
            Mean ratings for given movies
        """
        return self.rating_service.get_average_ratings_of_movies(movie_ids)


class RecommendService:
    def __init__(self, recommend_service):
        self.recommend_service = recommend_service

    def recommend_rating(self, user_id, movie_id, n_neighbor):
        """Estimates rating for given user and movie.

        Args:
            user_id: user id (not index of the table)
            movie_id: movie id (not index of the table)
            n_neighbor: number of K value for Knn

        Returns:
            none. it prints out estimation as of now.
        """
        return self.recommend_service.recommend_rating(user_id, movie_id, n_neighbor)
    
    def recommend_movie_by_genre(self, movie_title, n_recommended_movie):
        """Recommends movies by given movie genres.

        Args:
            movie_title: movie title (full length)
            n_recommended_movie: number of recommended movie

        Returns:
            List of recommended movies.
        """
        return self.recommend_service.recommend_movie_by_genre(movie_title, n_recommended_movie)


class UserService:
    def __init__(self, user_service):
        self.user_service = user_service

    def create_new_user(self, name, age, gender):
        """Create an app user by given parameters.

        Args:
            name: name string
            age: age in positive integer
            gender: M or F

        Returns:
            A User object containing attributes generated from given parameters.
        """
        return self.user_service.create_new_user(name, age, gender)

    def find_user(self, user_id):
        """Finds specific user by given user id.

        Args:
            user_id: unique user id.

        Returns:
            A User object containing attributes.
        """
        return self.user_service.search_user(user_id)

    def change_user(self, user_id):
        # Not implemented yet
        raise NotImplementedError("Not Implemented yet")

    def save_app_users(self):
        self.user_service.save_app_users()