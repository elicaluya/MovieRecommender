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
from tkinter.scrolledtext import ScrolledText


#
# Main GUI class where App is run
#
class UI:
    def __init__(self, app, num_of_movie_need_rating):
        self.num_of_movie_need_rating = num_of_movie_need_rating
        self.app_ref = app

        # Create GUI
        self.window = tk.Tk()
        self.window.geometry('700x700')
    
    def configure(self, bs_movie, bs_user, bs_recommend, bs_rating):
        self.bs_movie = bs_movie
        self.bs_user = bs_user
        self.bs_recommend = bs_recommend
        self.bs_rating = bs_rating

    
    # Starting window asking for new or returning user
    def log_in(self):
        title = tk.Label(self.window,text="Movie Recommender System").pack()
        self.return_button = tk.Button(self.window, text="Returning User", command=lambda : self.return_user_window())
        self.return_button.pack()
        self.new_button = tk.Button(self.window, text="New User", command=lambda : self.new_user())
        # Main Menu elements
        self.new_button.pack()
        self.by_rating = tk.Button(self.window, text="Recommend Movie by Rating", command=lambda : self.rec_rating_window())
        self.by_genre = tk.Button(self.window, text="Recommend Movie by Genre", command=lambda : self.rec_genre_window())
        self.chg_user = tk.Button(self.window, text="Change User", command=lambda : self.return_user_window())
        self.create_user = tk.Button(self.window, text="Create New User", command=lambda : self.new_user())
        self.quit_button = tk.Button(self.window, text="Quit", command=lambda : self.window.destroy())
        self.quit_button.pack()
        # Analysis elements
        self.status_label = tk.Label(self.window)
        self.user_info = tk.Label(self.window)
        self.user_count = tk.Label(self.window)
        self.user_status = tk.Label(self.window)
        self.ratings = tk.Label(self.window)
        self.movies_label = tk.Label(self.window, text="Movies:")
        self.movies = st.ScrolledText(self.window, width=75, height=20,font=("Times New Roman", 10))
      
    
    # Displays main menu buttons after login or user creation on the main window
    def show_menu(self):
        self.return_button.pack_forget()
        self.new_button.pack_forget()
        self.quit_button.pack_forget()
        self.by_rating.pack()
        self.by_genre.pack()
        self.chg_user.pack()
        self.create_user.pack()
        self.status_label.pack()
        self.user_info.pack()
        self.quit_button.pack()

    # Clears the output section in the main menu
    def clear_output(self):
        self.status_label.pack_forget()
        self.user_info.pack_forget()
        self.user_count.pack_forget()
        self.user_status.pack_forget()
        self.ratings.pack_forget()
        self.movies_label.pack_forget()
        self.movies.pack_forget()

    # Clears all of the widgets in a window
    def clear_input_ratings(self, newWindow):
        widget_list = newWindow.winfo_children()

        for item in widget_list:
            if item.winfo_children():
                widget_list.extend(item.winfo_children())

        for item in widget_list:
            item.pack_forget()
    
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
            self.user = self.bs_user.create_new_user(name, age, gender)
            status = "Current User ID: {}".format(self.user.get_id())
            info = "Created User: {} {} {} {}".format(self.user.get_id(), self.user.get_name(), self.user.get_age(), self.user.get_gender())
            self.update_status(status, info)
            self.clear_output()
            self.show_menu()
    
    # Method to display recommend by rating window
    def rec_rating_window(self):
        #new window on top with the labels defined in below code
        newWindow = Toplevel(self.window)
        label = tk.Label(newWindow, text="Recommend By Rating:")
        label.grid(row=0)
        movie_label = tk.Label(newWindow, text="Movie Title:")
        movie_label.grid(row=1)
        self.movie_entry = tk.Entry(newWindow)
        self.movie_entry.grid(row=1, column=1)
        #button to execute function
        submit_btn = tk.Button(newWindow, text="Submit", command=lambda : self.input_movie(newWindow))
        submit_btn.grid(row=3, column=1)
        back_btn = tk.Button(newWindow, text="Back", command=lambda : newWindow.destroy())
        back_btn.grid(row=4, column=1)

    # Method to display recommend by genre window
    def rec_genre_window(self):
        #new window on top with the labels defined in below code
        newWindow = Toplevel(self.window)
        label = tk.Label(newWindow, text="Recommend By Genre:")
        label.grid(row=0)
        movie_label = tk.Label(newWindow, text="Movie Title:")
        movie_label.grid(row=1)
        self.movie_entry = tk.Entry(newWindow)
        self.movie_entry.grid(row=1, column=1)
        #button to execute function
        submit_btn = tk.Button(newWindow, text="Submit", command=lambda : self.genre_input_movie(newWindow))
        submit_btn.grid(row=3, column=1)
        back_btn = tk.Button(newWindow, text="Back", command=lambda : newWindow.destroy())
        back_btn.grid(row=4, column=1)
    
    def input_movie(self, newWindow):
        #get movie title input from user
        movie = self.movie_entry.get()
        #check if movie exists in dataset
        found_movie = self.bs_movie.get_movie_by_title(movie)
        #label to be presented if movie title is invalid
        error_label = tk.Label(newWindow)
        #if movie not found, let user know. else, continue 
        if not found_movie:
            error_label.config(text="Error: Movie Not Found")
            error_label.grid(row=4)
        else:
            newWindow.withdraw()
            self.recommend_rating(found_movie)

    def genre_input_movie(self, newWindow):
        #get movie title input from user
        movie = self.movie_entry.get()
        #check if movie exists in dataset
        found_movie = self.bs_movie.get_movie_by_title(movie)
        #label to be presented if movie title is invalid
        error_label = tk.Label(newWindow)
        #if movie not found, let user know. else, continue 
        if not found_movie:
            error_label.config(text="Error: Movie Not Found")
            error_label.grid(row=4)
        else:
            newWindow.withdraw()
            self.input_genre(found_movie)
    
    def input_genre(self, movie):
<<<<<<< Updated upstream
        genre_movies = self.bs_recommend.recommend_movie_by_genre(movie.get_id(), 20)
        # not supported yet
=======
        #get and store list of similar movies based on genre
        genre_movies = self.bs_recommend.recommend_movie_by_genre(movie, 20)
        #set up window and listbox to display results
>>>>>>> Stashed changes
        newWindow = Toplevel(self.window)
        newWindow.geometry('650x500')
        canvas = tk.Canvas(newWindow)
        scrollbar = tk.Scrollbar(newWindow, orient="vertical")
        mov_output = tk.Listbox(newWindow, width=80, height=20, yscrollcommand=scrollbar.set)
        scrollbar = Scrollbar(newWindow, orient=VERTICAL, command=canvas.yview)
        frame = Frame(canvas)
        tk.Label(frame, text="Since you liked the movie {}, we recommend watching the following movies that have similar genres:".format(movie.get_title())).pack()

        tk.Button(frame, text="Close", command=lambda : newWindow.destroy()).pack()
        #loop through list of similar movies and print to listbox
        i=0
        for index, moviess in genre_movies.iterrows():
            mov_output.insert(i, moviess[0])
            i += 1

        canvas.create_window(0,0,anchor='nw',window=frame)
        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox('all'),yscrollcommand=scrollbar.set)
        canvas.pack(fill='both',expand=True, side='left')
        scrollbar.pack(fill='y', side='right')
        mov_output.pack(fill="both", expand=True)
        mov_output.place(relx = 0.5, rely = 0.5, anchor="center")
    


    # Window for user to input ratings of movies they have seen
    def input_ratings(self, movies, current_movie, newWindow):
        # Clear window output
        self.clear_input_ratings(newWindow)
        
        # Create frame and canvas for scrollbar
        canvas = tk.Canvas(newWindow)
        scrollbar = Scrollbar(newWindow, orient=VERTICAL, command=canvas.yview)
        frame = Frame(canvas)

        # take rating information from user
        tk.Label(frame, text="Please Input ratings for the following movies or leave blank if not seen:").pack()
        ratings = []
        for i, movie in movies.iterrows():
            ttk.Label(frame, text=movie["title"]).pack()
            n = tk.StringVar()
            rating_input = ttk.Combobox(frame,textvariable=n)
            rating_input['values'] = ('','0','1','2','3','4','5')
            rating_input.pack()
            ratings.append(rating_input)

        tk.Button(frame, text="Submit", command=lambda : self.submitRatings(movies, ratings, newWindow, current_movie)).pack()
        tk.Button(frame, text="Close", command=lambda : newWindow.destroy()).pack()

        # Make the window have a scrollbar if there are a lot of movies to rate
        canvas.create_window(0,0,anchor='nw',window=frame)
        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox('all'),yscrollcommand=scrollbar.set)
        canvas.pack(fill='both',expand=True, side='left')
        scrollbar.pack(fill='y', side='right')



    # Submit the ratings from user. If doesn't meet number of movies that need a rating, continue to ask user for ratings
    def submitRatings(self, movies, ratings, newWindow, current_movie):
        for i in range(len(ratings)):
            rate = ratings[i].get()
            if rate.isnumeric():
                r = int(rate)
                # zero is valid rating too
            else:
                # empty means didn't watch, record as NaN
                r = np.nan
            self.bs_rating.add_user_rating(self.user.get_id(), int(movies["movie_id"].iloc[i]), r)

        # Update the count of the number of ratings by the user
        found_ratings = self.bs_rating.get_valid_user_ratings(self.user.get_id())
        count = found_ratings.shape[0]
        self.user_count.config(text="User Count = %d" %count)
        self.user_count.pack()

        # When the user has enough ratings, get the prediction
        if count >= self.num_of_movie_need_rating:
            self.recommend_rating(current_movie)
            newWindow.destroy()
        # If the user still has not entered enough ratings, keep asking them to rate movies they have seen
        else:
            newMovies = self.get_movie_ratings(count)
            self.input_ratings(newMovies, current_movie, newWindow)

        

    # Get the top movie ratings and their average ratings
    def get_movie_ratings(self, count):
        movie_ratings = []
        self.movies.config(state=tk.NORMAL)
        self.movies.delete('1.0', END)
        # finds movies most watched by others that user has not seen
        movies = self.bs_movie.get_most_watched_movie(
                    self.user.get_id(),
                    math.ceil((self.num_of_movie_need_rating - count) * 1.5))
        for index, row in movies.iterrows():
            movie_info = "{} | {} | {} => ".format(row["movie_id"], row["title"], row["release_date"])
            movie_ratings.append(movie_info)
        # their average ratings
        ratings = self.bs_rating.get_average_ratings_of_movies(movies["movie_id"])
        
        # Update the textbox with the new movies that need a rating    
        mv_index = 0
        for index, row in ratings.iterrows():
            rating_info = "{}\n\n".format(row["rating"])
            movie_ratings[mv_index] += rating_info
        
            self.movies.insert(tk.INSERT, movie_ratings[mv_index])
            mv_index += 1
    
        self.movies.config(state=tk.DISABLED)
        self.movies_label.pack()
        self.movies.pack()

        return movies
    
    
    # Recommend the rating of movie based on movies the user has rated and other ratings
    def recommend_rating(self, movie):
        self.clear_output()
        found_ratings = self.bs_rating.get_valid_user_ratings(self.user.get_id())
        count = found_ratings.shape[0]
        self.user_count.config(text="User Count = %d" %count)
        self.user_count.pack()
        
        # User has submitted enough ratings to get a predicted rating
        if count >= self.num_of_movie_need_rating:
            self.user_status.config(text="User rating count is bigger than needed, here's the expected rating for your movie: {}".format(movie.get_title()))
            self.user_status.pack()
            self.ratings.config(text=self.bs_recommend.recommend_rating(self.user.get_id(), movie.get_id(), self.app_ref.knn_n_neighbor))
            self.ratings.pack()
        # The user needs to add more ratings in order to get a prediction
        else:
            self.user_status.config(text="User rating count is less than needed, please input your rating for the following movie: {}".format(movie.get_title()))
            self.user_status.pack()
            movies = self.get_movie_ratings(count)

            if self.user.get_id() <= 943:
                print("not allowed to update rating for dataset users")
                return 

            newWindow = Toplevel(self.window)
            newWindow.geometry('500x500')
            self.input_ratings(movies, movie, newWindow) 


    def run(self):
        self.log_in()
        tk.mainloop()