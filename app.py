import streamlit as st
import pandas as pd
import numpy as np
import requests
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="🎬 AI Movie Recommender", 
    page_icon="🍿", 
    layout="wide"
)

# ==========================================
# API CONFIGURATION
# ==========================================
# API Key from TMDB
TMDB_API_KEY = "4dcf3b6f2a75860bc432a0491b3b3940"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
POSTER_BASE_URL = "https://image.tmdb.org/t/p/w500"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
import time

@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    """Fetch movie poster URL using TMDB API."""
    for _ in range(3):
        try:
            url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            poster_path = data.get('poster_path')
            if poster_path:
                return f"{POSTER_BASE_URL}{poster_path}"
            break # Break if request succeeded but no poster
        except Exception as e:
            time.sleep(0.5)
            pass
    # Return a placeholder image if poster not found
    return "https://dummyimage.com/500x750/cccccc/000000.png&text=No+Poster"

@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):
    """Fetch additional details like rating and runtime."""
    for _ in range(3):
        try:
            url = f"{TMDB_BASE_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            data = response.json()
            return {
                'rating': data.get('vote_average', 'N/A'),
                'runtime': data.get('runtime', 'N/A')
            }
        except Exception as e:
            time.sleep(0.5)
            pass
    return {'rating': 'N/A', 'runtime': 'N/A'}

@st.cache_data(show_spinner=False)
def load_and_preprocess_data():
    """Load dataset and prepare features for ML."""
    try:
        # Try to load local 'movies.csv' (e.g., from TMDB 5000 movies Kaggle dataset)
        movies = pd.read_csv('movies.csv')
    except FileNotFoundError:
        # Fallback to a dummy dataset for presentation if the file is missing
        st.warning("⚠️ 'movies.csv' not found. Loading a sample dataset for demonstration. For full experience, download a TMDB Kaggle dataset, rename it to 'movies.csv' and place it here.")
        dummy_data = {
            'id': [278, 238, 155, 13, 122, 680, 11, 157336, 27205, 550, 24428, 99861, 744, 293660],
            'title': ['The Shawshank Redemption', 'The Godfather', 'The Dark Knight', 'Forrest Gump', 'The Lord of the Rings: The Return of the King', 'Pulp Fiction', 'Star Wars', 'Interstellar', 'Inception', 'Fight Club', 'The Avengers', 'Avengers: Age of Ultron', 'Top Gun', 'Deadpool'],
            'genres': ['[{"name": "Drama"}, {"name": "Crime"}]', '[{"name": "Drama"}, {"name": "Crime"}]', '[{"name": "Drama"}, {"name": "Action"}, {"name": "Crime"}, {"name": "Thriller"}]', '[{"name": "Comedy"}, {"name": "Drama"}, {"name": "Romance"}]', '[{"name": "Adventure"}, {"name": "Fantasy"}, {"name": "Action"}]', '[{"name": "Thriller"}, {"name": "Crime"}]', '[{"name": "Adventure"}, {"name": "Action"}, {"name": "Science Fiction"}]', '[{"name": "Adventure"}, {"name": "Drama"}, {"name": "Science Fiction"}]', '[{"name": "Action"}, {"name": "Science Fiction"}, {"name": "Adventure"}]', '[{"name": "Drama"}]', '[{"name": "Science Fiction"}, {"name": "Action"}, {"name": "Adventure"}]', '[{"name": "Action"}, {"name": "Adventure"}, {"name": "Science Fiction"}]', '[{"name": "Action"}, {"name": "Drama"}]', '[{"name": "Action"}, {"name": "Adventure"}, {"name": "Comedy"}]'],
            'overview': [
                "Framed in the 1940s for the double murder of his wife and her lover, upstanding banker Andy Dufresne begins a new life at the Shawshank prison.",
                "Spanning the years 1945 to 1955, a chronicle of the fictional Italian-American Corleone crime family.",
                "Batman raises the stakes in his war on crime. With the help of Lt. Jim Gordon and District Attorney Harvey Dent.",
                "A man with a low IQ has accomplished great things in his life and been present during significant historic events.",
                "Aragorn is revealed as the heir to the ancient kings as he, Gandalf and the other members of the broken fellowship struggle to save Gondor.",
                "A burger-loving hit man, his philosophical partner, a drug-addled gangster's moll and a washed-up boxer converge.",
                "Princess Leia is captured and held hostage by the evil Imperial forces in their effort to take over the galactic Empire.",
                "The adventures of a group of explorers who make use of a newly discovered wormhole.",
                "Cobb, a skilled thief who commits corporate espionage by infiltrating the subconscious of his targets.",
                "A ticking-time-bomb insomniac and a slippery soap salesman channel primal male aggression into a shocking new form of therapy.",
                "When an unexpected enemy emerges and threatens global safety and security, Nick Fury, director of the international peacekeeping agency known as S.H.I.E.L.D., finds himself in need of a team to pull the world back from the brink of disaster.",
                "When Tony Stark tries to jumpstart a dormant peacekeeping program, things go awry and Earth’s Mightiest Heroes are put to the ultimate test as the fate of the planet hangs in the balance.",
                "For Lieutenant Pete 'Maverick' Mitchell and his friend and co-pilot Nick 'Goose' Bradshaw, being accepted into an elite training school for fighter pilots is a dream come true.",
                "Deadpool tells the origin story of former Special Forces operative turned mercenary Wade Wilson, who after being subjected to a rogue experiment that leaves him with accelerated healing powers, adopts the alter ego Deadpool."
            ],
            'keywords': ['[{"name": "prison"}]', '[{"name": "mafia"}]', '[{"name": "dc comics"}, {"name": "batman"}]', '[{"name": "vietnam veteran"}]', '[{"name": "elves"}, {"name": "orcs"}]', '[{"name": "hitman"}]', '[{"name": "android"}]', '[{"name": "space"}, {"name": "wormhole"}]', '[{"name": "dream"}]', '[{"name": "dual identity"}]', '[{"name": "marvel comic"}, {"name": "superhero"}]', '[{"name": "marvel comic"}, {"name": "superhero"}]', '[{"name": "fighter pilot"}]', '[{"name": "mercenary"}, {"name": "superhero"}]'],
            'cast': ['[{"name": "Tim Robbins"}, {"name": "Morgan Freeman"}]', '[{"name": "Marlon Brando"}, {"name": "Al Pacino"}]', '[{"name": "Christian Bale"}, {"name": "Heath Ledger"}]', '[{"name": "Tom Hanks"}, {"name": "Robin Wright"}]', '[{"name": "Elijah Wood"}, {"name": "Viggo Mortensen"}]', '[{"name": "John Travolta"}, {"name": "Samuel L. Jackson"}]', '[{"name": "Mark Hamill"}, {"name": "Harrison Ford"}]', '[{"name": "Matthew McConaughey"}, {"name": "Anne Hathaway"}]', '[{"name": "Leonardo DiCaprio"}]', '[{"name": "Edward Norton"}, {"name": "Brad Pitt"}]', '[{"name": "Robert Downey Jr."}, {"name": "Chris Evans"}]', '[{"name": "Robert Downey Jr."}, {"name": "Chris Hemsworth"}]', '[{"name": "Tom Cruise"}, {"name": "Kelly McGillis"}]', '[{"name": "Ryan Reynolds"}, {"name": "Morena Baccarin"}]']
        }
        movies = pd.DataFrame(dummy_data)

    # Ensure required columns exist
    expected_cols = ['id', 'title', 'genres', 'overview', 'cast', 'keywords']
    for col in expected_cols:
        if col not in movies.columns:
            if col == 'id' and 'movie_id' in movies.columns:
                movies['id'] = movies['movie_id']
            else:
                st.error(f"Dataset is missing required column: '{col}'")
                st.stop()
                
    # Fill NaN values only for text columns to avoid dtype errors with numeric columns
    string_cols = ['title', 'genres', 'overview', 'cast', 'keywords']
    for col in string_cols:
        if col in movies.columns:
            movies[col] = movies[col].fillna('')

    # Helper function to extract names from JSON-like strings
    def extract_names(text):
        if isinstance(text, str) and text.startswith('['):
            try:
                items = ast.literal_eval(text)
                return ' '.join([item['name'] for item in items])
            except:
                return text
        return text

    # Apply extraction
    movies['genres_clean'] = movies['genres'].apply(extract_names)
    movies['cast_clean'] = movies['cast'].apply(extract_names)
    movies['keywords_clean'] = movies['keywords'].apply(extract_names)

    # Basic text preprocessing function
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    # Combine features for ML model
    movies['combined_features'] = (
        movies['genres_clean'].apply(clean_text) + ' ' + 
        movies['overview'].apply(clean_text) + ' ' + 
        movies['keywords_clean'].apply(clean_text) + ' ' + 
        movies['cast_clean'].apply(clean_text)
    )

    return movies

@st.cache_resource(show_spinner=False)
def compute_similarity(movies_df):
    """Compute and return the cosine similarity matrix using TF-IDF."""
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'])
    similarity = cosine_similarity(tfidf_matrix)
    return similarity

def get_recommendations(movie_title, movies_df, similarity_matrix, top_n=5):
    """Recommend similar movies based on cosine similarity scores."""
    try:
        # Find index of the selected movie
        movie_idx = movies_df[movies_df['title'] == movie_title].index[0]
        
        # Retrieve similarity scores
        distances = similarity_matrix[movie_idx]
        
        # Sort scores and get top N indices (excluding the movie itself)
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]
        
        recommended_movies = []
        for i in movies_list:
            idx = i[0]
            recommended_movies.append({
                'title': movies_df.iloc[idx]['title'],
                'id': movies_df.iloc[idx]['id']
            })
        return recommended_movies
    except IndexError:
        return []

# ==========================================
# MAIN UI LAYOUT
# ==========================================

# Apply some custom CSS for a modern look
st.markdown("""
    <style>
        .css-18e3th9 { padding-top: 2rem; }
        .stImage > img { border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); transition: transform 0.3s ease; }
        .stImage > img:hover { transform: scale(1.05); }
        .movie-title { font-weight: bold; font-size: 1.1rem; margin-top: 0.5rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
        .movie-caption { color: #888; font-size: 0.9rem; }
    </style>
""", unsafe_allow_html=True)

st.title("🎬 AI Movie Recommender")
st.markdown("Discover your next favorite movie using Machine Learning (TF-IDF & Cosine Similarity) and TMDB API!")

# Initialize session state for future recommendations and memory
if 'history' not in st.session_state:
    st.session_state.history = []

# Load data and ML model
with st.spinner("Loading dataset and calculating ML similarities..."):
    movies = load_and_preprocess_data()
    similarity = compute_similarity(movies)

# Extract unique genres for the dropdown
all_genres = []
for genres in movies['genres_clean']:
    all_genres.extend(genres.split())
unique_genres = sorted(list(set([g.capitalize() for g in all_genres if g])))

st.sidebar.header("🎯 Your Preferences")
selected_genre = st.sidebar.selectbox("1. Choose a Genre", ["Select..."] + unique_genres)

if selected_genre != "Select...":
    # Filter movies by genre (case insensitive)
    genre_movies = movies[movies['genres_clean'].str.contains(selected_genre, case=False, na=False)]
    
    if not genre_movies.empty:
        st.subheader(f"🍿 Movies in '{selected_genre}' Genre")
        selected_movie = st.selectbox(
            "2. Select a movie you like to get recommendations:", 
            ["Select a movie..."] + genre_movies['title'].tolist()
        )
        
        if selected_movie != "Select a movie...":
            if st.button("Get Recommendations", type="primary"):
                # Save to history for future recommendations
                if selected_movie not in st.session_state.history:
                    st.session_state.history.append(selected_movie)
                    
                st.markdown(f"### ✨ Top Matches for '{selected_movie}'")
                
                with st.spinner("Finding the best matches using AI..."):
                    recommendations = get_recommendations(selected_movie, movies, similarity, top_n=5)
                
                if recommendations:
                    # Display recommendations in a grid format
                    cols = st.columns(5)
                    for col, rec in zip(cols, recommendations):
                        with col:
                            poster_url = fetch_poster(rec['id'])
                            details = fetch_movie_details(rec['id'])
                            
                            st.image(poster_url, use_container_width=True)
                            st.markdown(f"<div class='movie-title'>{rec['title']}</div>", unsafe_allow_html=True)
                            st.markdown(f"<div class='movie-caption'>⭐ {details['rating']}/10 <br>⏱️ {details['runtime']}m</div>", unsafe_allow_html=True)
                else:
                    st.warning("Could not find recommendations for this movie.")
    else:
        st.info(f"No movies found for the genre '{selected_genre}'. Try another one.")
else:
    st.info("👈 Please select a genre from the sidebar to begin!")

# ==========================================
# FUTURE RECOMMENDATIONS SECTION
# ==========================================
st.markdown("---")
st.header("🕰️ Future Recommendations")

if st.session_state.history:
    st.write(f"Based on your recent interest in: **{', '.join(st.session_state.history[-3:])}**")
    
    # Recommend based on the most recently viewed movie
    last_movie = st.session_state.history[-1]
    future_recs = get_recommendations(last_movie, movies, similarity, top_n=10)
    
    # Exclude the immediate top 5 (since they were likely just shown)
    future_recs = future_recs[5:] 
    
    if future_recs:
        cols = st.columns(len(future_recs))
        for col, rec in zip(cols, future_recs):
            with col:
                poster_url = fetch_poster(rec['id'])
                st.image(poster_url, use_container_width=True)
                st.markdown(f"<div class='movie-title' style='font-size: 0.9rem;'>{rec['title']}</div>", unsafe_allow_html=True)
    else:
        st.info("Keep exploring to get more diverse future recommendations!")
else:
    st.info("Interact with the recommender above to build your history and get personalized future suggestions!")

# ==========================================
# FOOTER
# ==========================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Built with ❤️ using Streamlit, Scikit-Learn, and TMDB API.<br>"
    "<i>Features TF-IDF Vectorization & Cosine Similarity for Content-Based Filtering.</i>"
    "</div>", 
    unsafe_allow_html=True
)
