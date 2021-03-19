"""Movie service layer object module.

This module includes a class representing service related with movie data.
It links to data layer object to deliver the functionality.
"""

class MovieService:
    def __init__(self, movie_data, rating_data):
        """Constructor, it connects data objects in lower level operation layer.

        Args:
            movie_data: movie dataset module
            rating_data: rating dataset module
        """
        self.movie_data = movie_data
        self.rating_data = rating_data

    def get_most_watched_movies(self, user_id, count):
        """Finds most watched(rated) movies of specified counts and not rated by the given user.

        Args:
            user_id: user id (not index of the table)
            count: number of movies to get

        Returns:
            A dataframe of movies list.
        """
        return self.movie_data.get_movies(
            self.rating_data.get_most_watched_movie_index(user_id, count))

    def get_movie_by_title(self, title):
        """Finds specific movie in the movie dataset by given title.

        Args:
            title: movie title (must be exact match as of now)

        Returns:
            A Movie object containing attributes from the movie dataset.
        """
        return self.movie_data.get_movie_by_title(title)