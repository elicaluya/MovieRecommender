#!/usr/bin/python

import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import ttk
import tkinter.scrolledtext as st
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import NMF


class Movie:
    def __init__(self, id, title, genre):
        self.id = id
        self.title = title
        self.genre = genre
    
    def get_id(self):
        return self.id
    
    def get_title(self):
        return self.title
    
    def get_genre(self):
        return self.genre


class User:
    def __init__(self, id, name, age, gender):
        self.id = int(id)
        self.name = name
        self.age = int(age)
        self.gender = gender
        self.ratings = None
    
    def get_name(self):
        return self.name
    
    def get_age(self):
        return self.age
    
    def get_gender(self):
        return self.gender
    
    def get_id(self):
        return self.id



#
# DataSet class for retrieving and editing the data
#
class DataSet:
    def __init__(self):
        self.df_user = pd.read_csv(
            "movielens/Movielens-02/u.user",
            delimiter="|",
            names=['user_id', 'age', 'gender', 'occupation', 'zipcode'])
        self.df_movie = pd.read_csv(
            "movielens/Movielens-02/u.item",
            delimiter="|",
            encoding='latin-1',
            names=['movie_id', 'title', 'release_date', 'video_release_date',
                   'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation',
                   'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama',
                   'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                   'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'])
        self.df_data = pd.read_csv(
            "movielens/Movielens-02/u.data",
            delimiter="\t",
            names=['user_id', 'item_id', 'rating', 'timestamp'])
        self.df_ratmat = pd.read_csv("movielens/Movielens-02/data_matrix.csv", index_col=0)
        
        self.max_user_id = self.df_user["user_id"].max()

        # Reading data for testing the genre recommender method
        # self.df_movie_genre = pd.read_csv("movielens/Movielens-02/movie_genre.csv")
        
        # read user data
        try:
            self.df_app_user = pd.read_csv("app_user.csv")
        except FileNotFoundError as e:
            # start with empty set
            self.df_app_user = pd.DataFrame(
                columns=['user_id', 'name', 'age', 'gender'])
        
        self.max_app_user_id = self.df_app_user["user_id"].max()
        
        # read user rating data
        try:
            self.df_app_data = pd.read_csv("app_data.csv")
        except FileNotFoundError as e:
            # start with empty set
            self.df_app_data = pd.DataFrame(
                columns=['user_id', 'item_id', 'rating', 'timestamp'])
    
    def _search_app_user(self, id):
        if self.df_app_user.empty:
            return None
        found_df = self.df_app_user.loc[self.df_app_user["user_id"] == int(id)]
        if found_df.empty:
            return None
        row = found_df.iloc[0]
        print("found user:", id, row["name"], row["age"], row["gender"])
        return User(id, row["name"], row["age"], row["gender"])
    
    def _search_user(self, id):
        found_df = self.df_user.loc[self.df_user["user_id"] == int(id)]
        if found_df.empty:
            return None
        row = found_df.iloc[0]
        print("found user:", id, "", row["age"],
              row["gender"], row["occupation"], row["zipcode"])
        return User(id, "", row["age"], row["gender"])
    
    def search_user(self, id):
        # search in app user list
        user = self._search_app_user(id)
        if user:
            return user
        # then search in user list
        return self._search_user(id)
    
    def search_movie_by_title(self, title):
        if self.df_movie.empty:
            return None
        found_df = self.df_movie.loc[self.df_movie["title"] == title]
        if found_df.empty:
            return None
        row = found_df.iloc[0]
        return Movie(row["movie_id"], row["title"], "unknown")
    
    def save(self):
        # save all regardless they were updated or not (user, rating)
        self.df_app_user = self.df_app_user.to_csv("app_user.csv", index=False)
        self.df_app_data = self.df_app_data.to_csv("app_data.csv", index=False)
    
    def add_user(self, name, age, gender):
        new_user_id = max(self.max_user_id, self.max_app_user_id) + 1
        new_row = {'user_id': new_user_id, 'name': name, 'age': age, 'gender': gender}
        self.df_app_user = self.df_app_user.append(new_row, ignore_index=True)
        self.max_app_user_id = new_user_id
        return User(new_user_id, name, age, gender)
    
    def count_user_rating(self, user_id):
        found_ratings = self.df_app_data["user_id"].loc[
            (self.df_app_data["user_id"] == int(user_id)) & (self.df_app_data["rating"].isna() == False)]
        return found_ratings.shape[0]
    
    def find_most_watched_movies(self, id, count):
        zero_ratings = self.df_ratmat.apply(pd.Series.value_counts).iloc[0, 1:]
        ids_rated = self.df_app_data.loc[self.df_app_data["user_id"] == int(id)]["item_id"]
        movies_not_rated = zero_ratings[
            zero_ratings.index.isin(ids_rated.astype(int).astype(str)) != True].sort_values().head(count)
        return self.df_movie.loc[self.df_movie["movie_id"].isin(movies_not_rated.index)]
    
    def print_ratings(self, movie_ids):
        return self.df_data[self.df_data["item_id"].isin(movie_ids)].groupby("item_id").mean()
    
    def is_app_user(self, id):
        return (self.max_user_id < int(id))
    
    def add_user_rating(self, user_id, movie_id, rating):
        print("adding user rating : ", user_id, movie_id, rating)
        new_row = {'user_id': int(user_id), 'item_id': int(movie_id), 'rating': rating}
        self.df_app_data = self.df_app_data.append(new_row, ignore_index=True)
    
    def print_ratings_by_user(self, user_id):
        movies = []
        ratings = self.df_ratmat.iloc[user_id]
        for i, rating in enumerate(ratings):
            if int(rating) > 0:
                movies.append((self.df_ratmat.columns[i], rating))
                # print(movies)
    
    def recommend_rating(self, user_id, movie_id):
        user_w_movie = self.df_ratmat[self.df_ratmat.iloc[:, movie_id - 1] > 0]
        n_users = min(10, user_w_movie.shape[0])
        if n_users < 1:
            print("No ratings found with your movie")
            return
        knn = Knn(user_w_movie, n_users, False, '')
        movies_rated = self.df_app_data.loc[
            (self.df_app_data["user_id"] == int(user_id)) & ((self.df_app_data["rating"].isna() == False))]
        movie_size = self.df_ratmat.shape[1]
        cols = [str(i) for i in range(1, movie_size + 1)]
        df = pd.DataFrame(columns=cols)
        new_row = {}
        for i, r in movies_rated[['item_id', 'rating']].iterrows():
            new_row[str(int(r['item_id']))] = int(r['rating'])
        df = df.append(new_row, ignore_index=True)
        df = df.fillna(0)
        # print(df)
        nbrs = knn.neighbor(df, n_users)
        total = 0.0
        total_score = 0.0
        indices = nbrs[1][0]
        for i, index in enumerate(indices):
            score = 1.0 - nbrs[0][0][i]  # 1 - distance
            rating = user_w_movie.iloc[index][movie_id - 1]
            user_id = user_w_movie.iloc[index].name
            # print(i, "index:", index, "user_id:", user_id, "rating:", rating, "score:", score)
            self.print_ratings_by_user(user_id)
            total += score * rating
            total_score += score
        return "expected rating ===> {}".format(total / total_score)
    
    def rating_by_nmf(self, user_id, movie_id):
        user_w_movie = self.df_ratmat[self.df_ratmat.iloc[:,movie_id - 1] > 0]
        n_users = min(10, user_w_movie.shape[0])
        if n_users < 1:
            #print("No ratings found with your movie")
            return "No ratings found with your movie"
        movies_rated = self.df_app_data.loc[(self.df_app_data["user_id"] == int(user_id)) & ((self.df_app_data["rating"].isna() == False))]
        movie_size = self.df_ratmat.shape[1]
        cols = [str(i) for i in range(1, movie_size + 1)]
        df = pd.DataFrame(columns=cols)
        new_row = {}
        for i, r in movies_rated[['item_id', 'rating']].iterrows():
            new_row[str(int(r['item_id']))] = int(r['rating'])
        df = df.append(new_row, ignore_index = True)
        df = df.fillna(0)
        temp_df = self.df_ratmat.append(df)
        nmf = NMF(100)
        W = nmf.fit_transform(temp_df)
        H = nmf.components_
        h = H[:,movie_id]
        w = W[user_id]
        
        #print("movie_id ===>", movie_id)
        #print("expected rating ===>", np.dot(w, h))
        mov = "movie_id ===>", movie_id, "\n"
        rating = "expected rating ===>", np.dot(w, h)
        return mov + rating

#
# KNN class
#
class Knn:
    def recommend_movie(self, movie_data, knn_model, num_recommendations):
        neighbors = knn_model.kneighbors(
            movie_data,
            n_neighbors=num_recommendations,
            return_distance=True)
        return neighbors
    
    def __init__(self, dataset, K, W, M):  # instance, K, Weighted voting, Metric
        self.cosine_knn = NearestNeighbors(
            metric="cosine", algorithm="brute", n_neighbors=K, n_jobs=1)
        self.cosine_knn.fit(dataset)
    
    def neighbor(self, X, K):
        return self.recommend_movie(X, self.cosine_knn, K)
        # return (0, [0])
    
    def predict(self, X, K):
        distances, indices = self.neighbor(X, K)
        return 0.0

#Recommendation based on similar genre 
class Genre_Based:
    def __init__(self, movie_title, dataset):
        self.movie_title = movie_title
        self.dataset = dataset

        ###transforming data to be used with cos sim in genre recommender method
        #vectorizer to be used when creating the tfidf matrix
    def tfidf_prep(self):
        tfidf_movies_genres = TfidfVectorizer(token_pattern = '[a-zA-Z0-9\-]+')
        #creating the tfidf matrix
        tfidf_movies_genres_matrix = tfidf_movies_genres.fit_transform(self.dataset['genres'])
        #computing the cosine similarity
        cosine_sim_movies = linear_kernel(tfidf_movies_genres_matrix, tfidf_movies_genres_matrix)
        return cosine_sim_movies

    ### recommend movies based on similar genres
    def genre_recommender(self):
        #computing the cosine similarity and storing locally
        cosine_sim_movies = self.tfidf_prep()
        #variable to store index of movie that the user has specified
        movie_index = self.dataset.loc[self.dataset['title'].isin([self.movie_title])]
        movie_index = movie_index.index
        #computing the similarity score and then sorting based on the score
        movies_sim_scores = list(enumerate(cosine_sim_movies[movie_index][0]))
        movies_sim_scores = sorted(movies_sim_scores, key=lambda x: x[1], reverse=True)
        #fetch score of the most similar movies and get their movie index to be used when printing out result to user
        movies_sim_scores = movies_sim_scores[1:21]
        movie_indices = [i[0] for i in movies_sim_scores]

        #outputting results sorted by ratings
        return print(self.dataset[['title','genres','rating']].iloc[movie_indices].sort_values('rating', ascending=False))
    

#
# Main GUI class where App is run
#
class App:
    def __init__(self, num_of_movie_need_rating):
        self.num_of_movie_need_rating = num_of_movie_need_rating
        self.dataset = DataSet()
        
        # Create GUI
        self.window = tk.Tk()
        self.window.geometry('700x700')
    
    def user_input(self, prompt):
        val = input(prompt)
        return val
    
    # Starting window asking for new or returning user
    def log_in(self):
        title = tk.Label(self.window,text="Movie Recommender System").pack()
        self.return_button = tk.Button(self.window, text="Returning User", command=lambda : self.return_user_window())
        self.return_button.pack()
        self.new_button = tk.Button(self.window, text="New User", command=lambda : self.new_user())
        # Main Menu elements
        self.new_button.pack()
        self.by_rating = tk.Button(self.window, text="Recommend Movie by Rating", command=lambda : self.rec_rating_window())
        self.by_genre = tk.Button(self.window, text="Recommend Movie by Genre")
        self.chg_user = tk.Button(self.window, text="Change User", command=lambda : self.return_user_window())
        self.quit_button = tk.Button(self.window, text="Quit", command=lambda : self.window.destroy())
        self.quit_button.pack()
        # Analysis elements
        self.status_label = tk.Label(self.window)
        self.user_info = tk.Label(self.window)
        self.user_count = tk.Label(self.window)
        self.user_status = tk.Label(self.window)
        self.ratings = tk.Label(self.window)
        self.movies_label = tk.Label(self.window, text="Movies and Average Ratings:")
        self.movies = st.ScrolledText(self.window, width=50, height=10,font=("Times New Roman", 10))

        

        #
        # Brad's code
        #
        # while True:
        #     user_id = self.user_input(
        #         "Type in user id or [n for new user] or [q for quit]:")
        #     if user_id == 'n':
        #         name = self.user_input("Type in user name:")
        #         age = self.user_input("Type in user age:")
        #         gender = self.user_input("Type in user gender (M: male, F: femail):")
        #         return self.dataset.add_user(name, age, gender)
        #     elif user_id == 'q':
        #         return None
        #     elif user_id.isnumeric():
        #         user = self.dataset.search_user(user_id)
        #         if user:
        #             return user
        #         print("User id not found")
        # return None
        
        
    def show_menu(self):
        self.return_button.pack_forget()
        self.new_button.pack_forget()
        self.quit_button.pack_forget()
        self.by_rating.pack()
        self.by_genre.pack()
        self.chg_user.pack()
        self.chg_user.pack()
        self.status_label.pack()
        self.user_info.pack()
        self.quit_button.pack()

    def clear_output(self):
        self.status_label.pack_forget()
        self.user_info.pack_forget()
        self.user_count.pack_forget()
        self.user_status.pack_forget()
        self.ratings.pack_forget()
        self.movies_label.pack_forget()
        self.movies.pack_forget()

    
    # Function to update the status label
    def update_status(self, status, info):
        self.status_label.config(text=status)
        self.user_info.config(text=info)
    
    # Method for input of user id of returning user
    def return_user_window(self):
        newWindow = Toplevel(self.window)
        label = tk.Label(newWindow, text="Please enter User ID:").pack()
        self.uid = tk.Entry(newWindow)
        self.uid.pack()
        submit_button = tk.Button(newWindow, text="Submit", command=lambda : self.valid_userid(newWindow)).pack()
        back_button = tk.Button(newWindow, text="Back", command=lambda : newWindow.destroy()).pack()
    
    # Check if input is valid for user id
    def valid_userid(self, newWindow):
        uid = self.uid.get()
        error_label = tk.Label(newWindow)
        if uid.isnumeric() == False:
            error_label.pack_forget()
            error_label.config(text="Error: Invalid User ID \nMust Only Contain Numbers")
            error_label.pack()
        else:
            user = self.dataset.search_user(uid)
            if  user:
                newWindow.withdraw()
                self.user = user
                status = "Current User ID: {}".format(self.user.get_id())
                info = "Found User: {} {} {}".format(self.user.get_id(), self.user.get_age(), self.user.get_gender())
                self.update_status(status, info)
                self.clear_output()
                self.show_menu()
            else:
                error_label.pack_forget()
                error_label.config(text="Error: User Not Found")
                error_label.pack()
                
                
    
    # Method for input of info for new user
    def new_user(self):
        newWindow = Toplevel(self.window)
        label = tk.Label(newWindow, text="Please Enter the Following Info:")
        label.grid(row=0)
        name_label = tk.Label(newWindow, text="User Name:")
        name_label.grid(row=1)
        self.name_entry = tk.Entry(newWindow)
        self.name_entry.grid(row=1, column=1)
        age_label = tk.Label(newWindow,text="User Age:")
        age_label.grid(row=2)
        self.age_entry = entry = tk.Entry(newWindow)
        self.age_entry.grid(row=2, column=1)
        gen_label = tk.Label(newWindow,text="Gender (M or F):")
        gen_label.grid(row=3)
        self.gen_entry = tk.Entry(newWindow)
        self.gen_entry.grid(row=3, column=1)
        submit_button = tk.Button(newWindow, text="Submit", command=lambda : self.valid_new_user(newWindow))
        submit_button.grid(row=4)
        back_button = tk.Button(newWindow, text="Back to Login", command=lambda: newWindow.destroy())
        back_button.grid(row=5)
        
    
    # Checking if the inputs for the new user is valid
    def valid_new_user(self, newWindow):
        name = self.name_entry.get()
        age =  self.age_entry.get()
        gender = self.gen_entry.get()

        err_msg = ""
        
        if name.isalpha() == False:
            err_msg += "- Invalid character in Name \n"
        
        if age.isnumeric() == False:
            err_msg += "- Invalid character in Age \n"
            
        if gender != "M" and gender != "F":
            err_msg += "- Gender must be M or F"
            
        error_label = tk.Label(newWindow)
        if len(err_msg) > 0:
            error_label.config(text="Error: %s" %err_msg)
            error_label.grid(row=6)
        else:
            newWindow.withdraw()
            self.user = self.dataset.add_user(name, age, gender)
            status = "Current User ID: {}".format(self.user.get_id())
            info = "Created New User: {} {} {}".format(self.user.get_id(), self.user.get_age(), self.user.get_gender()) 
            self.update_status(status, info)
            self.clear_output()
            self.show_menu()
            
    
    
    # Window for the main menu with buttons corresponding to each option
    #def main_menu(self):
        #
        # Brad's code
        #
        # print("Main menu:")
        # print("- [r]recommend movie by user rating")
        # print("- [g]recommend movie by genre (* not supported yet)")
        # print("- [c]hange user")
        # print("- [q]quit")
        # while True:
        #     func = self.user_input("Select function or [q for quit]:")
        #     if func == 'q':
        #         break
        #     if func == 'r':
        #         return func
        #     if func == 'g':
        #         return None
        #     if func == 'c':
        #         return func
        # return None
    
    
    # Method to display recommend by rating window
    def rec_rating_window(self):
        newWindow = Toplevel(self.window)
        label = tk.Label(newWindow, text="Recommend By Rating:")
        label.grid(row=0)
        movie_label = tk.Label(newWindow, text="Movie Title:")
        movie_label.grid(row=1)
        self.movie_entry = tk.Entry(newWindow)
        self.movie_entry.grid(row=1, column=1)
        submit_btn = tk.Button(newWindow, text="Submit", command=lambda : self.input_movie(newWindow))
        submit_btn.grid(row=3, column=1)
        back_btn = tk.Button(newWindow, text="Back", command=lambda : newWindow.destroy())
        back_btn.grid(row=4, column=1)
    
    
    def input_movie(self, newWindow):
        movie = self.movie_entry.get()
        found_movie = self.dataset.search_movie_by_title(movie)
        error_label = tk.Label(newWindow)
        if not found_movie:
            error_label.config(text="Error: Movie Not Found")
            error_label.grid(row=4)
        else:
            newWindow.withdraw()
            self.recommend_rating(found_movie)
        
        # while True:
        #     title = self.user_input("Type in a movie title or [q for quit function]:")
        #     if title == 'q':
        #         break
        #     found_movie = self.dataset.search_movie_by_title(title)
        #     if found_movie:
        #         return found_movie
        #     print("Movie is not found")
        # return None
    
    def input_genre(self):
        # not supported yet
        return None
    
    def recommend_movie(self, genre):
        # not supported yet
        return
    
    def input_ratings(self, movies):
        newWindow = Toplevel(self.window)
        newWindow.geometry('400x400')
        tk.Label(newWindow, text="Please Input ratings for the following movies:").pack()
        ratings = []
        for i, movie in movies.iterrows():
            ttk.Label(newWindow, text=movie["title"]).pack()
            n = tk.StringVar()
            rating_input = ttk.Combobox(newWindow,textvariable=n)
            rating_input['values'] = ('0','1','2','3','4','5')
            rating_input.pack()
            ratings.append(rating_input)

        tk.Button(newWindow, text="Submit", command=lambda : self.submitRatings(movies, ratings, newWindow)).pack()
        tk.Button(newWindow, text="Close", command=lambda : newWindow.destroy()).pack()
        

        # # take rating information from user
        # ratings = []
        # for i, movie in movies.iterrows():
        #     rating = self.user_input("[%s] --- %s : " % (i, movie["title"]))
        #     if rating == 'q':
        #         return None
        #     ratings.append(rating)
        # # now save valid ratings
        # for i, rate in enumerate(ratings):
        #     if rate.isnumeric():
        #         r = int(rate)
        #         # zero is valid rating too
        #     else:
        #         # empty means didn't watch, record as NaN
        #         r = np.nan
        #     self.dataset.add_user_rating(self.user.get_id(), int(movies["movie_id"].iloc[i]), r)
        # return True
    
    def submitRatings(self, movies, ratings, newWindow):
        for i in range(len(ratings)):
            rate = ratings[i].get()
            if rate.isnumeric():
                r = int(rate)
                # zero is valid rating too
            else:
                # empty means didn't watch, record as NaN
                r = np.nan
            self.dataset.add_user_rating(self.user.get_id(), int(movies["movie_id"].iloc[i]), r)
        newWindow.destroy()

    
    def recommend_rating(self, movie):
        count = self.dataset.count_user_rating(self.user.get_id())
        self.user_count.config(text="user count = %d" %count)
        self.user_count.pack()

        if count >= self.num_of_movie_need_rating:
            self.user_status.config(text="user rating count is bigger than needed, here's the expected rating for your movie: {}".format(movie.get_title()))
            self.user_status.pack()
            self.ratings.config(text=self.dataset.recommend_rating(self.user.get_id(), movie.get_id()))
            self.ratings.pack()
        else:
            self.user_status.config(text="user rating count is less than needed, please input your rating for the following movie: {}".format(movie.get_title()))
            self.user_status.pack()
            movie_ratings = []
            self.movies.config(state=tk.NORMAL)
            self.movies.delete('1.0', END)
            # finds movies most watched by others
            movies = self.dataset.find_most_watched_movies(
                     self.user.get_id(),
                     math.ceil((self.num_of_movie_need_rating - count) * 1.5))
            for index, row in movies.iterrows():
                movie_info = "{} | {} | {} => ".format(row["movie_id"], row["title"], row["release_date"])
                movie_ratings.append(movie_info)
                
            # their average ratings?
            ratings = self.dataset.print_ratings(movies["movie_id"])
            
            mv_index = 0
            for index, row in ratings.iterrows():
                rating_info = "{}\n\n".format(row["rating"])
                movie_ratings[mv_index] += rating_info
        
                self.movies.insert(tk.INSERT, movie_ratings[mv_index])
                mv_index += 1

            self.movies.config(state=tk.DISABLED)
            self.movies_label.pack()
            self.movies.pack()

            if not self.dataset.is_app_user(self.user.get_id()):
                print("not allowed to update rating for dataset users")
                return

            self.input_ratings(movies)            

        
        
        # while True:
        #     count = self.dataset.count_user_rating(self.user.get_id())
        #     print("user count =", count)
        #     if count >= self.num_of_movie_need_rating:
        #         print("user rating count is bigger than needed, here's the expected rating for your movie")
        #         #self.dataset.recommend_rating(self.user.get_id(), movie.get_id())
        #         self.dataset.rating_by_nmf(self.user.get_id(), movie.get_id())
        #         return
        #     else:
        #         print("user rating count is less than needed, please input your rating for the following movie")
        #         # finds movies most watched by others
        #         movies = self.dataset.find_most_watched_movies(
        #             self.user.get_id(),
        #             math.ceil((self.num_of_movie_need_rating - count) * 1.5))
        #         print(movies)
        #         # their average ratings?
        #         self.dataset.print_ratings(movies["movie_id"])
        #         # only allow ratings from app user
        #         if not self.dataset.is_app_user(self.user.get_id()):
        #             print("not allowed to update rating for dataset users")
        #             break
        #         if not self.input_ratings(movies):
        #             break
        # return
    
    def run(self):
        self.user = self.log_in()
        if not self.user:
            return
        while True:
            action = self.main_menu()
            if not action or action == 'q':
                break
            elif action == 'r':
                # recommend movie rating
                movie_found = self.input_movie()
                self.recommend_rating(movie_found)
            elif action == 'g':
                # recommend in genre
                genre = self.input_genre()
                self.recommend_movie(genre)
            elif action == 'c':
                # change user : Not yet supported
                self.change_user()
        self.dataset.save()

def main():
    app = App(5)
    app.run()
    tk.mainloop()

    ###For testing genre based
    # df_movie_genre = pd.read_csv("movielens/Movielens-02/movie_genre.csv")
    # genre_test = Genre_Based("Toy Story (1995)", df_movie_genre)
    # genre_test.genre_recommender()

if __name__ == "__main__":
    main()
