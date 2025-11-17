# 🎬 FlickFinder — Movie Discovery & Watchlist Platform

FlickFinder is a full-stack movie discovery website built with **React (Vite)** and **FastAPI**, integrating **TMDB API** for movie data and **MongoDB Atlas** for user authentication and persistent watchlists.  
It supports trending movies, search, recommendations, detailed movie pages, login/signup, and a fully synced cloud watchlist.

---

## 🚀 Features

### **Frontend (React + Vite)**
- Beautiful UI with TailwindCSS + Framer Motion  
- Browse trending, top-rated, upcoming movies  
- Advanced search  
- Movie detail pages with trailers & cast  
- Recommendations  
- Watchlist UI (add/remove/update sorted list)  
- Authentication (login/signup)  
- Axios API wrapper with token interceptors  
- Local + server-sync watchlist logic  

### **Backend (FastAPI)**
- JWT-based authentication  
- TMDB API proxy endpoints:  
  - `/api/trending`  
  - `/api/top`  
  - `/api/upcoming`  
  - `/api/movie/{id}`  
  - `/api/search`  
  - `/api/recommendations/{id}`  
- Watchlist endpoints:  
  - `/api/watchlist` (GET)  
  - `/api/watchlist` (POST)  
  - `/api/watchlist/{movie_id}` (DELETE)  
- MongoDB Atlas storage (users + watchlists)  
- CORS configured for Vite frontend  

---

## 🗂 Tech Stack

### **Frontend**
- React (Vite)  
- TailwindCSS  
- Framer Motion  
- Axios  

### **Backend**
- FastAPI  
- MongoDB Atlas (Motor)  
- JWT Authentication  
- TMDB API  

---

## 🔧 Environment Variables

### **Frontend (`.env`)**
VITE_API_BASE=http://localhost:8000<br>
VITE_TMDB_KEY=your_tmdb_api_key<br>

### **Backend (`.env`)**
MONGO_URI=your_mongodb_atlas_uri<br>
MONGO_DB=FlickFinderDB<br>
JWT_SECRET=your_secret<br>
JWT_ALGORITHM=HS256<br>
TMDB_API_KEY=your_tmdb_api_key<br>
CORS_ORIGINS=http://localhost:5173<br>

---
