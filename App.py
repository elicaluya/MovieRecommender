#!/usr/bin/python

import Service as bs
import MovieService as ms
import UserService as us
import RecommendService as cs
import RatingService as rs
import MovieData as md
import UserData as ud
import RatingData as rd
from Knn import Knn
from Genre import Genre
from UI import UI

class App:
    def configure(
        self, 
        num_of_movie_need_rating=60,
        knn_sim_metric="pearson",
        knn_n_neighbor=10):
        self.num_of_movie_need_rating = num_of_movie_need_rating

        # configure backend objects
        knn = Knn(knn_sim_metric, knn_n_neighbor)
        genre = Genre()

        movie_data = md.MovieData()
        user_data = ud.UserData()
        app_user_data = ud.AppUserData(user_data)
        rating_data = rd.RatingData()

        user_service = us.UserService(user_data, app_user_data)
        movie_service = ms.MovieService(movie_data, user_service, rating_data)
        recommend_service = cs.RecommendService(rating_data, movie_data, user_data, app_user_data, knn, genre)
        rating_service = rs.RatingService(rating_data, movie_data, user_data, app_user_data)

        self.bs_movie = bs.MovieService(movie_service)
        self.bs_user = bs.UserService(user_service)
        self.bs_recommend = bs.RecommendService(recommend_service)
        self.bs_rating = bs.RatingService(rating_service)

        # configure frontend UI object
        self.fe = UI(num_of_movie_need_rating)
        self.fe.configure(self.bs_movie, self.bs_user, self.bs_recommend, self.bs_rating)
    
    def run(self):
        self.fe.run()
        self.bs_user.save_app_users()
        self.bs_rating.save_app_data()

def main():
    app = App()
    app.configure(60, "cosine", 10)
    app.run()

    ###For testing genre based
    # df_movie_genre = pd.read_csv("movielens/Movielens-02/movies_w_genre.csv")
    # genre_test = Genre_Based("Toy Story (1995)", df_movie_genre)
    # genre_test.genre_recommender()

if __name__ == "__main__":
    main()
