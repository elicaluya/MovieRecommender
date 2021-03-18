"""Movie data layer object module.

This module includes a MovieData class representing data layer object for Movie dataset.
It provides low level access and manipulation on movie dataset.
"""
from Movie import Movie
import pandas as pd


class MovieData:
    def __init__(self, df_movie = None, df_movie_genre = None):
        """Constructor, reads dataset files or use given dataset for movie information,
        and movie genre dataset.

        Args:
            df_movie: custom movie information dataset
            df_movie_genre: custom movie genre information dataset
        """
        if df_movie is None:
            # use base dataset read from file
            self.df_movie = pd.read_csv(
                "movielens/Movielens-02/u.item",
                delimiter="|",
                encoding='latin-1',
                names=['movie_id', 'title', 'release_date', 'video_release_date',
                    'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                    'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
                    'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                    'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        else:
            # use given custom dataset
            self.df_movie = df_movie

        if df_movie_genre is None:
            # Reading data for testing the genre recommender method
            self.df_movie_genre = pd.read_csv("movielens/Movielens-02/movies_w_genre.csv", encoding='latin-1')
        else:
            # use given custom dataset
            self.df_movie_genre = df_movie_genre

    def get_movies(self, movie_ids):
        """Gets a list of movies by given list of ids.

        Args:
            movie_ids: list of movie id

        Returns:
            A list of movies.
        """
        return self.df_movie.loc[self.df_movie["movie_id"].isin(movie_ids)]

    def get_movie_by_title(self, title):
        """Finds specific movie in the movie dataset by given title.

        Args:
            title: movie title (must be exact match as of now)

        Returns:
            A Movie object containing attributes from the movie dataset.
        """
        if self.df_movie.empty:
            return None
        # find movie by given title, this is exact matching basis.
        found_df = self.df_movie.loc[self.df_movie["title"] == title]
        if found_df.empty:
            # Simply return and report movie not found
            return None
        row = found_df.iloc[0]
        # return Movie object for future use
        return Movie(row["movie_id"], row["title"], "unknown")

    def get_genre_dataset(self):
        # passes over entire dataset
        return self.df_movie_genre