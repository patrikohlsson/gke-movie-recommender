import torch

import pandas as pd

from flask import Flask
app = Flask(__name__)

class Recommender:
    def __init__(self):
        # Load model
        self.model = torch.jit.load('baseline.pt', map_location='cpu').eval()

        # Load data
        ratings_df = pd.read_csv('./ratings.csv')
        movies_df = pd.read_csv('./movies.csv')

        unique_users = ratings_df.userId.unique()
        unique_movies = ratings_df.movieId.unique()

        # Create user-item matrix
        r_ui = torch.zeros(len(unique_users), len(unique_movies)) * float('nan')
        for u, i, r in zip(ratings_df.userId.factorize()[0], ratings_df.movieId.factorize()[0], ratings_df.rating):
            r_ui[u, i] = r
        
        self.r_ui = r_ui
        self.ratings = ratings_df
        self.movies = movies_df

    def get_top_k_unseen(self, user_id, k=10):
        user_ratings = self.r_ui[user_id]
        unseen_movies = user_ratings.isnan().nonzero().squeeze(1)
        k = min(k, len(unseen_movies))
        with torch.no_grad():
            pred_ratings = self.model(torch.tensor([user_id]), unseen_movies)
            # print(f'{torch.tensor([user_id]).shape = } {unseen_movies.shape = } {pred_ratings.shape = } {k = }')
            movie_idx = pred_ratings.topk(k).indices.tolist()
            movie_ids = [int(self.ratings.movieId.iloc[i]) for i in movie_idx]
            
            # Grab the movie name from the self.movies dataframe by looking up the row index in the movieId column for each entry and grabbing the title
            movie_titles = [self.movies[self.movies.movieId == movie_id].title.iloc[0] for movie_id in movie_ids]
            
        return movie_ids, movie_titles

@app.route('/')
def hello():
    return 'Hello from Python on GKE!\n'

@app.route('/recommendations/<user_id>')
def recommendations(user_id):
    recommender = Recommender()

    movie_ids, movie_names = recommender.get_top_k_unseen(int(user_id), k=10)

    return {'user_id': user_id, 'movie_ids': movie_ids, 'movie_names': movie_names}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
