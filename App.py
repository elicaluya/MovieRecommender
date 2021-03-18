#!/usr/bin/python
"""Movie Recommender System application class module.

This module includes a App class representing application instance,
which is instantiated and run for main entry point of the app operation.
"""
import Service as bs
import MovieService as ms
import UserService as us
import RecommendService as cs
import RatingService as rs
import MovieData as md
import UserData as ud
import RatingData as rd
import Knn
import Genre
import UI

class App:
    def configure(
        self, num_of_movie_need_rating=60,
        knn_sim_metric="correlation", knn_n_neighbor=10,
        df_user=None, df_app_user=None,
        df_data=None, df_ratmat=None, df_app_data=None,
        df_movie=None,df_movie_genre=None):
        """Instantiates, wire and configure all the object comprising of the system.

        Args:
            num_of_movie_need_rating: minimum amount of rating required for a user
            knn_sim_metric: KNN metric option, default: "correlation"
            knn_n_neighbor: number of K nearest users for KNN
            df_user: optional custom data set for base user
            df_app_user: optional custom data set for app user
            df_data: optional custom data set for base user rating data
            df_ratmat: optional custom data set for userxrating matrix
            df_app_data: optional custom data set for app user rating data
            df_movie: optional custom data set for movie information
            df_movie_genre: optional custom data set for movie by genre information
        """
        self.num_of_movie_need_rating = num_of_movie_need_rating

        # configure model objects
        self.knn_n_neighbor = knn_n_neighbor
        knn = Knn.Knn(knn_sim_metric)
        genre = Genre.Genre()

        # configure backend data objects
        movie_data = md.MovieData(df_movie=df_movie, df_movie_genre=df_movie_genre)
        user_data = ud.UserData(df_user=df_user)
        app_user_data = ud.AppUserData(user_data, df_app_user=df_app_user)
        rating_data = rd.RatingData(df_data=df_data, df_ratmat=df_ratmat, df_app_data=df_app_data)

        # configure backend service objects
        user_service = us.UserService(user_data, app_user_data)
        movie_service = ms.MovieService(movie_data, user_service, rating_data)
        recommend_service = cs.RecommendService(rating_data, movie_data, user_data, app_user_data, knn, genre)
        rating_service = rs.RatingService(rating_data, movie_data, user_data, app_user_data)

        # app reference to backend service end points
        self.bs_movie = bs.MovieService(movie_service)
        self.bs_user = bs.UserService(user_service)
        self.bs_recommend = bs.RecommendService(recommend_service)
        self.bs_rating = bs.RatingService(rating_service)

        # configure frontend UI object and passes over backend end points
        self.fe = UI.UI(self, num_of_movie_need_rating)
        self.fe.configure(self.bs_movie, self.bs_user, self.bs_recommend, self.bs_rating)
    
    def run(self):
        """Main entry point of the app."""
        # run UI main loop
        self.fe.run()
        # before exiting, saves changed app user and rating data
        self.bs_user.save_app_users()
        self.bs_rating.save_app_data()

def main():
    """Global entry point of the run."""
    app = App()
    # By default, we demand 60 rating input, correlation metric and 10 closest user in KNN.
    # please refer to evaluation report for reasoning of these values
    app.configure(60, "correlation", 10)
    app.run()

if __name__ == "__main__":
    main()
