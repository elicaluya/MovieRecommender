"""Rating service layer object module.

This module includes a class representing service related with rating data.
It links to data layer object to deliver the functionality.
"""

class RatingService:
    def __init__(self, rating_data, movie_data, user_data, app_user_data):
        """Constructor, it connects data objects in lower level operation layer.

        Args:
            rating_data: rating dataset module
            movie_data: movie dataset module
            user_data: base user dataset module
            app_user_data: app user dataset module
        """
        self.rating_data = rating_data
        self.movie_data = movie_data
        self.user_data = user_data
        self.app_user_data = app_user_data

    def get_movies_with_zero_rating(self):
        """Lists most watched(rated) movies across all users.

        Returns:
            A list of all movies along with their 0 rating counts.
        """
        return self.rating_data.get_movies_with_zero_rating()

    def get_valid_user_ratings(self, user_id):
        """Finds valid (> 0) ratings for given user.

        Args:
            user_id: unique user id

        Returns:
            A data frame for user_id, item_id, rating
        """
        return self.rating_data.get_valid_user_ratings(user_id)

    def save_app_data(self):
        """Store all the app user ratings into app data files."""
        self.rating_data.save()

    def add_user_rating(self, user_id, movie_id, rating):
        """Adds a rating input from user into app dataset.

        Args:
            user_id: user id (not index of the table)
            movie_id: movie id (not index of the table)
            rating: rating (0 - 5)
        """
        print("adding user rating : ", user_id, movie_id, rating)
        self.rating_data.add_user_rating(user_id, movie_id, rating)

    def get_average_ratings_of_movies(self, movie_ids):
        """Calculates a mean ratings for given movies.

        Args:
            movie_ids: list of movie ids
        
        Returns:
            Mean ratings for given movies
        """
        return self.rating_data.get_average_ratings_of_movies(movie_ids)