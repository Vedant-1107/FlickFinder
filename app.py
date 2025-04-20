from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# TMDb API setup
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")  # Securely loaded from environment variables
TMDB_API_URL = "https://api.themoviedb.org/3"

# Load and prepare the dataset
df = pd.read_csv("assets/movie_dataset.csv")
features = ['keywords', 'cast', 'genres', 'director']
for feature in features:
    df[feature] = df[feature].fillna('')
df["combined_features"] = df.apply(lambda row: ' '.join([row[f] for f in features]), axis=1)

# CountVectorizer + Cosine Similarity
cv = CountVectorizer()
count_matrix = cv.fit_transform(df["combined_features"])
cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return df.iloc[index]["title"]

def get_index_from_title(title):
    matches = df[df.title.str.lower() == title.lower()]
    return matches.index[0] if not matches.empty else None

def search_movie_by_title(title):
    url = f"{TMDB_API_URL}/search/movie?api_key={TMDB_API_KEY}&query={title}"
    response = requests.get(url).json()
    return response['results'][0]['id'] if response.get('results') else None

def get_movie_details(movie_id):
    url = f"{TMDB_API_URL}/movie/{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url).json()
    return {
        "poster_url": f"https://image.tmdb.org/t/p/w500{response['poster_path']}" if response.get('poster_path') else None,
        "plot": response.get('overview', 'No plot available'),
        "cast": "Not Available",  # Can be added with credits API
        "release_date": response.get('release_date', 'N/A'),
    }

@app.route("/recommend", methods=["GET"])
def recommend():
    movie_title = request.args.get("movie")
    if not movie_title:
        return jsonify({"error": "Missing movie parameter"}), 400

    index = get_index_from_title(movie_title)
    if index is None:
        return jsonify({"error": "Movie not found"}), 404

    similar_movies = list(enumerate(cosine_sim[index]))
    sorted_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:6]

    recommendations, movie_details = [], []
    for i, _ in sorted_movies:
        movie_name = get_title_from_index(i)
        movie_id = search_movie_by_title(movie_name)
        if movie_id:
            details = get_movie_details(movie_id)
            recommendations.append(movie_name)
            movie_details.append(details)

    return jsonify({"recommendations": recommendations, "movie_details": movie_details})

@app.route("/search", methods=["GET"])
def search_movies():
    query = request.args.get("q", "").lower()
    matches = df[df["title"].str.lower().str.contains(query)]
    titles = matches["title"].head(10).tolist()
    return jsonify({"titles": titles})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
