#!/usr/bin/python

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


#Recommendation based on similar genre 
class Genre:
    def __init__(self, sim_metric="cosine"):
        self.tfidf_movies_genres = TfidfVectorizer(token_pattern = '[a-zA-Z0-9\-]+')

    def fit(self, dataset):
        self.dataset = dataset
        #creating the tfidf matrix
        tfidf_movies_genres_matrix = self.tfidf_movies_genres.fit_transform(self.dataset['genre'])
        #computing the cosine similarity
        self.cosine_sim_movies = linear_kernel(tfidf_movies_genres_matrix, tfidf_movies_genres_matrix)

    def predict(self, movie_title, K):
        self.movie_title = movie_title
        #variable to store index of movie that the user has specified
        #movie_index = self.dataset.loc[self.dataset['movie title'].isin([self.movie_title])]
        movie_index = self.dataset.loc[self.dataset['movie title'] == self.movie_title.get_title()]
        movie_index = movie_index.index
        #computing the similarity score and then sorting based on the score
        movies_sim_scores = list(enumerate(self.cosine_sim_movies[movie_index][0]))
        movies_sim_scores = sorted(movies_sim_scores, key=lambda x: x[1], reverse=True)
        #fetch score of the most similar movies and get their movie index to be used when printing out result to user
        movies_sim_scores = movies_sim_scores[1:K+1]
        movie_indices = [i[0] for i in movies_sim_scores]

        #outputting results sorted by ratings
        return self.dataset[['movie title','genre','rating']].iloc[movie_indices].sort_values('rating', ascending=False)