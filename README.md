# FlickFinder - Movie Recommendation System

FlickFinder is a web-based movie recommendation system built with **Flask**, **AngularJS**, and **TMDb API**.<br>
It recommends similar movies based on a given title and displays detailed information including poster, plot, release date, and more.<br>

## Features

- 🎥 Movie recommendations based on content similarity (cast, keywords, director, genre)<br>
- 🔍 Autocomplete movie title input<br>
- 🖼️ Posters and details from TMDb API<br>
- 🌓 Dark-themed UI with hover-based interactive cards<br>
- ⚡ Instant recommendations with Enter key support<br>
- 🔒 CORS-enabled Flask backend for local frontend access<br>

### Install dependencies

Make sure you have Python 3 and pip installed. Then run:

```pip install flask pandas scikit-learn requests flask-cors```

### Run the Flask server

```python app.py```

By default, it runs on http://localhost:5000.

### Open the app

Open index.html in your browser. (Use a local server or enable CORS if needed.)


## How It Works

1. Combines movie metadata (cast, genres, keywords, director)<br>
2. Uses CountVectorizer and cosine_similarity from scikit-learn<br>
3. Fetches recommended movie posters and plots using TMDb API<br>
4. Displays recommendations in a responsive card layout<br>
