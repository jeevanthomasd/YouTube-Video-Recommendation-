import streamlit as st
import pandas as pd
import pickle
from sqlalchemy import create_engine

# Load the PostgreSQL database
DATABASE_TYPE = 'postgresql'
DBAPI = 'psycopg2'
ENDPOINT = 'localhost'
USER = 'user name'
PASSWORD = 'Your Password'
PORT = 5432
DATABASE = 'ytdb'

# Create an engine to connect to the PostgreSQL database
engine = create_engine(f'{DATABASE_TYPE}+{DBAPI}://{USER}:{PASSWORD}@{ENDPOINT}:{PORT}/{DATABASE}')

# Load TF-IDF Vectorizer and K-Means Model
with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('kmeans_model.pkl', 'rb') as kmeans_file:
    kmeans = pickle.load(kmeans_file)

# Load video data from PostgreSQL
table_name = 'video_tags_clusters'
video_df = pd.read_sql(table_name, engine)

# Streamlit app layout
st.title('YouTube Video Recommendation System')

# Display the YouTube logo
logo_url = 'E:/finalpro/logo.png'  # Replace with your logo URL or local file path
st.image(logo_url, width=200)  # Adjust width as needed

# Search bar
search_query = st.text_input('Enter tags to search for videos:', '')

if search_query:
    # Vectorize the search query
    query_vector = vectorizer.transform([search_query])
    
    # Predict the cluster for the search query
    cluster = kmeans.predict(query_vector)[0]
    
    # Filter videos by the predicted cluster
    recommended_videos = video_df[video_df['cluster'] == cluster]
    
    # Display recommended videos
    if not recommended_videos.empty:
        st.write(f"Recommended Videos for Tags: `{search_query}`")
        for index, row in recommended_videos.iterrows():
            video_url = f"https://www.youtube.com/watch?v={row['video_id']}"
            st.write(f"{row['title']}")
            st.video(video_url)  # Embed the video
            st.write("---")
    else:
        st.write("No recommendations found.")
