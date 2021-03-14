from Movie import Movie
import pandas as pd


class MovieData:
    def __init__(self):
        self.df_movie = pd.read_csv(
            "movielens/Movielens-02/u.item",
            delimiter="|",
            encoding='latin-1',
            names=['movie_id', 'title', 'release_date', 'video_release_date',
                   'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                   'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
                   'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                   'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])

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
        found_df = self.df_movie.loc[self.df_movie["title"] == title]
        if found_df.empty:
            return None
        row = found_df.iloc[0]
        return Movie(row["movie_id"], row["title"], "unknown")