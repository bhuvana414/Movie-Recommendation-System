import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.sparse import csr_matrix
import streamlit as st

class MovieRecommender:
    def __init__(self):
        self.movies = None
        self.tfidf_matrix = None
        self.user_movie_matrix = None
        
    @st.cache_data
    def load_data(_self):
        """Load and preprocess movie data"""
        data = pd.read_csv('movie-recommendation-system-main/data/content_based_filtering_dataset.csv')
        
        # Clean and preprocess
        data['title_clean'] = data['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
        data['year'] = data['title'].str.extract(r'\((\d{4})\)')
        data['genres'] = data['genres'].fillna('').str.replace('|', ' ')
        
        # Use combined_features if available
        if 'combined_features' in data.columns:
            data['features'] = data['combined_features'].fillna('')
        else:
            data['features'] = data['genres'].fillna('')
        
        # Remove rows with empty features
        data = data[data['features'].str.strip() != '']
        
        return data
    
    @st.cache_resource
    def build_content_model(_self, data):
        """Build content-based filtering model using TF-IDF"""
        tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
        tfidf_matrix = tfidf.fit_transform(data['features'])
        return tfidf_matrix
    
    def content_based_recommendations(self, title, top_n=5, genre_filter=None):
        """Get content-based recommendations"""
        # Find the movie
        matches = self.movies[self.movies['title'] == title]
        
        if matches.empty:
            matches = self.movies[self.movies['title'].str.contains(title, case=False, na=False)]
        
        if matches.empty:
            return pd.DataFrame()
        
        idx = matches.index[0]
        
        # Compute cosine similarity
        cosine_similarities = linear_kernel(self.tfidf_matrix[idx:idx+1], self.tfidf_matrix).flatten()
        
        # Get top similar movies
        similar_indices = cosine_similarities.argsort()[::-1][1:top_n+100]
        
        recommendations = self.movies.iloc[similar_indices].copy()
        recommendations['similarity_score'] = cosine_similarities[similar_indices]
        
        # Apply genre filter
        if genre_filter and len(genre_filter) > 0:
            genre_pattern = '|'.join([f'\\b{genre}\\b' for genre in genre_filter])
            recommendations = recommendations[
                recommendations['genres'].str.contains(genre_pattern, case=False, na=False, regex=True)
            ]
        
        return recommendations[['title', 'genres', 'year', 'similarity_score', 'imdbId']].head(top_n)
    
    def hybrid_recommendations(self, title, top_n=5, genre_filter=None):
        """Hybrid recommendations combining content-based filtering"""
        return self.content_based_recommendations(title, top_n, genre_filter)
    
    def get_recommendations_by_preferences(self, preferred_genres, top_n=5):
        """Get recommendations based on user genre preferences"""
        if not preferred_genres:
            # Return popular movies
            return self.movies.head(top_n)[['title', 'genres', 'year', 'imdbId']]
        
        # Filter movies by preferred genres
        genre_pattern = '|'.join([f'\\b{genre}\\b' for genre in preferred_genres])
        filtered_movies = self.movies[
            self.movies['genres'].str.contains(genre_pattern, case=False, na=False, regex=True)
        ]
        
        if filtered_movies.empty:
            return pd.DataFrame()
        
        # Return top movies from filtered set
        return filtered_movies.head(top_n)[['title', 'genres', 'year', 'imdbId']]
