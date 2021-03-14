#!/usr/bin/python

import tkinter as tk
from tkinter import *
from tkinter import messagebox
import tkinter.scrolledtext as st
import numpy as np
import pandas as pd
import math
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


#
# Main GUI class where App is run
#
class UI:
    def __init__(self, num_of_movie_need_rating):
        self.num_of_movie_need_rating = num_of_movie_need_rating
        
        # Create GUI
        self.window = tk.Tk()
        self.window.geometry('700x700')
    
    def configure(self, bs_movie, bs_user, bs_recommend, bs_rating):
        self.bs_movie = bs_movie
        self.bs_user = bs_user
        self.bs_recommend = bs_recommend
        self.bs_rating = bs_rating

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
        self.movies_label = tk.Label(self.window, text="Movies:")
        self.movies = st.ScrolledText(self.window, width=50, height=10,font=("Times New Roman", 10))
        self.ratings_label = tk.Label(self.window, text="Ratings:")
        self.ratings_scroll = st.ScrolledText(self.window, width=50, height=10,font=("Times New Roman", 10))        
        
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
        self.user_count = tk.Label(self.window).pack_forget()
        self.user_status = tk.Label(self.window).pack_forget()
    
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
            user = self.bs_user.find_user(uid)
            if  user:
                newWindow.withdraw()
                self.user = user
                status = "Current User ID: {}".format(self.user.get_id())
                info = "Found User: {} {} {}".format(self.user.get_id(), self.user.get_age(), self.user.get_gender())
                self.update_status(status, info)
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
            self.user = self.bs_user.create_new_user(name, age, gender)
            status = "Current User ID: ", self.user.get_id()
            info = "Created User: {} {} {} {}".format(self.user.get_id(), self.user.get_name(), self.user.get_age(), self.user.get_gender())
            self.update_status(status, info)
            self.show_menu()
    
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
        found_movie = self.bs_movie.get_movie_by_title(movie)
        error_label = tk.Label(newWindow)
        if not found_movie:
            error_label.config(text="Error: Movie Not Found")
            error_label.grid(row=4)
        else:
            newWindow.withdraw()
            self.recommend_rating(found_movie)
    
    def input_genre(self):
        # not supported yet
        return None
    
    def recommend_movie(self, genre):
        # not supported yet
        return
    
    def input_ratings(self, movies):
        # take rating information from user
        ratings = []
        for i, movie in movies.iterrows():
            rating = self.user_input("[%s] --- %s : " % (i, movie["title"]))
            if rating == 'q':
                return None
            ratings.append(rating)
        # now save valid ratings
        for i, rate in enumerate(ratings):
            if rate.isnumeric():
                r = int(rate)
                # zero is valid rating too
            else:
                # empty means didn't watch, record as NaN
                r = np.nan
            self.bs_user.add_user_rating(self.user.get_id(), int(movies["movie_id"].iloc[i]), r)
        return True
    
    def change_user(self):
        # not supported yet
        return
    
    def recommend_rating(self, movie):
        found_ratings = self.bs_rating.get_valid_user_ratings(self.user.get_id())
        count = found_ratings.shape[0]
        self.user_count.config(text="user count = %d" %count)
        self.user_count.pack()
        if count >= self.num_of_movie_need_rating:
            self.user_status.config(text="user rating count is bigger than needed, here's the expected rating for your movie")
            self.user_status.pack()
            self.ratings.config(text=self.bs_recommend.recommend_rating(self.user.get_id(), movie.get_id()))
        else:
            self.user_status.config(text="user rating count is less than needed, please input your rating for the following movie")
            self.user_status.pack()
            # finds movies most watched by others
            movies = self.bs_movie.get_most_watched_movie(
                     self.user.get_id(),
                     math.ceil((self.num_of_movie_need_rating - count) * 1.5))
            for index, row in movies.iterrows():
                movie_info = "{} | {} | {}\n".format(row["movie_id"], row["title"], row["release_date"])
                self.movies.insert(tk.INSERT,  movie_info)
            # their average ratings?
            ratings = self.bs_rating.get_average_ratings_of_movies(movies["movie_id"])
            print(ratings)
            for index, row in ratings.iterrows():
                rating_info = "{}\n".format(row["rating"])
                self.ratings_scroll.insert(tk.INSERT, rating_info)
    
            self.movies_label.pack()
            self.movies.pack()
            self.ratings_label.pack()
            self.ratings_scroll.pack()
    
    def run(self):
        self.log_in()
        tk.mainloop()