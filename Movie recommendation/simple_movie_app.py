import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import requests

@st.cache_data
def load_and_process_data():
    """Load and process movie data"""
    data = pd.read_csv('movie-recommendation-system-main/data/content_based_filtering_dataset.csv')
    
    # Clean title
    if 'title' not in data.columns:
        st.error("Title column not found in dataset!")
        return pd.DataFrame()
    
    data['title_clean'] = data['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True)
    
    # Handle genres - replace pipe with space for better matching
    if 'genres' in data.columns:
        data['genres'] = data['genres'].fillna('').str.replace('|', ' ')
    
    # Use combined_features if available, otherwise use genres
    if 'combined_features' in data.columns:
        data['features'] = data['combined_features'].fillna('')
    else:
        data['features'] = data['genres'].fillna('')
    
    # Remove rows with empty features
    data = data[data['features'].str.strip() != '']
    
    return data

@st.cache_resource
def build_tfidf_matrix(_data):
    """Build TF-IDF matrix"""
    tfidf = TfidfVectorizer(stop_words='english', max_features=3000)
    tfidf_matrix = tfidf.fit_transform(_data['features'])
    return tfidf_matrix

@st.cache_data(ttl=3600)
def get_movie_poster(imdb_id, title):
    """Fetch movie poster from TMDb API using IMDb ID"""
    try:
        if pd.notna(imdb_id) and imdb_id != '':
            # Format imdbId properly
            imdb_str = 'tt' + str(int(float(imdb_id))).zfill(7)
            
            # TMDb API - free and more reliable
            # Get your own key from https://www.themoviedb.org/settings/api
            api_key = "8265bd1679663a7ea12ac168da84d2e8"  # Demo key
            
            # First, find the movie by IMDb ID
            search_url = f"https://api.themoviedb.org/3/find/{imdb_str}?api_key={api_key}&external_source=imdb_id"
            
            response = requests.get(search_url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check if movie was found
                if data.get('movie_results') and len(data['movie_results']) > 0:
                    movie = data['movie_results'][0]
                    poster_path = movie.get('poster_path')
                    
                    if poster_path:
                        # TMDb image base URL
                        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
                        return poster_url
    except Exception as e:
        print(f"Error fetching poster for {title}: {e}")
    
    # Fallback to placeholder
    title_short = title.split('(')[0].strip()[:20]
    title_encoded = title_short.replace(' ', '+')
    return f"https://via.placeholder.com/300x450/667eea/FFFFFF?text={title_encoded}"

def get_recommendations(title, data, tfidf_matrix, top_n=10, genre_filter=None):
    """Get movie recommendations"""
    # Find the movie - use exact match first, then partial
    matches = data[data['title'] == title]
    
    if matches.empty:
        matches = data[data['title'].str.contains(title, case=False, na=False)]
    
    if matches.empty:
        return pd.DataFrame()
    
    idx = matches.index[0]
    
    # Compute cosine similarity only for this movie
    cosine_similarities = linear_kernel(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
    
    # Get top similar movies
    similar_indices = cosine_similarities.argsort()[::-1][1:top_n+100]
    
    recommendations = data.iloc[similar_indices].copy()
    recommendations['similarity'] = cosine_similarities[similar_indices]
    
    # Apply genre filter if specified
    if genre_filter and len(genre_filter) > 0:
        # Create a pattern that matches any of the selected genres
        genre_pattern = '|'.join([f'\\b{genre}\\b' for genre in genre_filter])
        recommendations = recommendations[
            recommendations['genres'].str.contains(genre_pattern, case=False, na=False, regex=True)
        ]
    
    # Return top N after filtering with imdbId
    return recommendations[['title', 'genres', 'similarity', 'imdbId']].head(top_n)

def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide", initial_sidebar_state="expanded")
    
    # Enhanced Custom CSS
    st.markdown("""
        <style>
        /* Main background */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
        }
        
        /* Movie card styling */
        .movie-card {
            background: white;
            border-radius: 15px;
            padding: 15px;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            margin: 10px;
            height: 100%;
        }
        .movie-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.3);
        }
        
        /* Movie title */
        .movie-title {
            font-size: 15px;
            font-weight: 600;
            margin-top: 12px;
            color: #2d3748;
            min-height: 45px;
            line-height: 1.4;
        }
        
        /* Movie genres */
        .movie-genres {
            font-size: 12px;
            color: #718096;
            margin-top: 8px;
            font-style: italic;
        }
        
        /* Header styling */
        .main-header {
            text-align: center;
            color: white;
            font-size: 48px;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .sub-header {
            text-align: center;
            color: rgba(255, 255, 255, 0.9);
            font-size: 18px;
            margin-bottom: 30px;
        }
        
        /* Button styling */
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 12px 40px;
            font-size: 16px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.6);
        }
        
        /* Selectbox styling */
        .stSelectbox {
            background: white;
            border-radius: 10px;
        }
        
        /* Checkbox styling */
        .stCheckbox {
            margin-bottom: 8px;
        }
        
        /* Remove default streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Content container */
        .content-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 30px;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🎬 Movie Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Discover your next favorite movie with AI-powered recommendations</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading movie database..."):
        data = load_and_process_data()
        tfidf_matrix = build_tfidf_matrix(data)
    
    # Sidebar with genre checkboxes
    st.sidebar.header("Filter by Genres")
    
    # Get all unique genres
    all_genres = set()
    for genres in data['genres'].dropna():
        all_genres.update(genres.split())
    
    # Remove empty strings and '(no genres listed)'
    all_genres = {g.strip() for g in all_genres if g.strip() and g.strip() != '(no' and g.strip() != 'genres' and g.strip() != 'listed)'}
    all_genres = sorted(list(all_genres))
    
    # Initialize session state for genres if not exists
    if 'selected_genres' not in st.session_state:
        st.session_state.selected_genres = set()
    
    # Create checkboxes for each genre
    selected_genres = []
    for genre in all_genres:
        if st.sidebar.checkbox(genre, value=genre in st.session_state.selected_genres):
            selected_genres.append(genre)
    
    # Update session state
    st.session_state.selected_genres = set(selected_genres)
    
    # Movie selection in a nice container
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🎥 Select a Movie")
        movie_titles = data['title'].tolist()
        selected_movie = st.selectbox(
            "Choose a movie you like",
            options=movie_titles,
            index=0,
            label_visibility="collapsed"
        )
        
        # Recommend button centered
        st.markdown("<br>", unsafe_allow_html=True)
        recommend_button = st.button("✨ Get Recommendations", type="primary", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Get recommendations
    if recommend_button:
        with st.spinner("Finding similar movies..."):
            recommendations = get_recommendations(
                selected_movie, 
                data, 
                tfidf_matrix, 
                top_n=12,
                genre_filter=selected_genres if selected_genres else None
            )
            
            if recommendations.empty:
                st.error("No recommendations found. Try removing genre filters.")
            else:
                st.markdown("---")
                
                # Display in 3-column grid
                num_cols = 3
                rows = [recommendations.iloc[i:i+num_cols] for i in range(0, len(recommendations), num_cols)]
                
                with st.spinner("🎬 Loading movie posters..."):
                    for row_data in rows:
                        cols = st.columns(num_cols)
                        for col, (idx, movie) in zip(cols, row_data.iterrows()):
                            with col:
                                poster_url = get_movie_poster(movie['imdbId'], movie['title'])
                                
                                # Display using enhanced card design
                                st.markdown(f"""
                                    <div class="movie-card">
                                        <img src="{poster_url}" style="width: 100%; border-radius: 10px; height: 380px; object-fit: cover;" onerror="this.src='https://via.placeholder.com/300x450/667eea/FFFFFF?text=No+Image';" />
                                        <div class="movie-title">{movie['title']}</div>
                                        <div class="movie-genres">🎭 {movie['genres']}</div>
                                    </div>
                                """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
