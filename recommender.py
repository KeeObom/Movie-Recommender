# Import Packages
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

# load MovieLens dataset
def load_data():
    movies = pd.read_csv('data/movies.csv') # Movies data
    ratings = pd.read_csv('data/ratings.csv') # Ratings data

    return movies, ratings

# Merge moovie and rating data
def merge_data(movies, ratings):
    movie_data = pd.merge(ratings, movies, on='movieId')
    
    return movie_data

# Create a user-item matrix
def create_user_item_matrix(data):
    user_movie_ratings = data.pivot_table(index='userId', columns='title', values='rating')
    return user_movie_ratings

# Build collaborative filtering using user-based similarity
def collaborative_filtering_with_explanation(user_id, user_movie_ratings, n_recommendations=5):
    user_similarity = 1 - pairwise_distances(user_movie_ratings.fillna(0), metric='cosine')
    
    # Find similar users
    similar_users = np.argsort(user_similarity[user_id])[-n_recommendations:]
    
    recommended_movies = []
    explanations = []
    
    for user in similar_users:
        user_rated_movies = user_movie_ratings.iloc[user].dropna().index
        recommended_movies.extend(user_rated_movies)
        explanations.append(f"Recommended based on similar user {user + 1}'s preferences.")
    
    recommended_movies = list(set(recommended_movies))  # Remove duplicates
    return recommended_movies, explanations


# Build content-based filtering using TF-IDF for genres
def content_based_filtering_with_explanation(movie_title, movies, n_recommendations=5):
    # Vectorize the genres
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'])
    
    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # Get movie index
    movie_index = movies[movies['title'] == movie_title].index[0]
    
    # Get similar movies
    similar_movies = list(enumerate(cosine_sim[movie_index]))
    similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:n_recommendations + 1]
    
    recommended_movie_indices = [i[0] for i in similar_movies]
    recommendations = movies['title'].iloc[recommended_movie_indices].values
    
    explanations = [f"Recommended because it has similar genres to {movie_title}." for _ in recommendations]
    
    return recommendations, explanations


def hybrid_recommendation_with_explanation(user_id, movie_title, user_movie_ratings, movies, n_recommendations=5):
    collab_recs, collab_explanations = collaborative_filtering_with_explanation(user_id, user_movie_ratings, n_recommendations)
    content_recs, content_explanations = content_based_filtering_with_explanation(movie_title, movies, n_recommendations)
    
    # Combine and deduplicate recommendations
    combined_recs = list(set(collab_recs + list(content_recs)))
    combined_explanations = collab_explanations + content_explanations
    
    return combined_recs[:n_recommendations], combined_explanations[:n_recommendations]