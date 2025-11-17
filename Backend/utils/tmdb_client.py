# backend/utils/tmdb_client.py
import os
import urllib.parse
from dotenv import load_dotenv
import httpx
from typing import Optional, Any, Dict
from ..db.redis import redis_client as redis

load_dotenv()
TMDB_KEY = os.getenv("TMDB_API_KEY")
TMDB_BASE = "https://api.themoviedb.org/3"

# Helper to build TMDB urls
def tmdb_url(path: str, params: Optional[Dict[str, Any]] = None) -> str:
    params = params or {}
    params["api_key"] = TMDB_KEY
    query = urllib.parse.urlencode(params)
    return f"{TMDB_BASE}{path}?{query}"

# Async client used for all TMDB calls
async_client = httpx.AsyncClient(timeout=20.0)

# Caching helpers - use redis client from db/redis.py
async def get_cached(key: str):
    val = await redis.get(key)
    if val:
        return val
    return None

async def set_cached(key: str, value: str, ttl: int):
    await redis.set(key, value, ex=ttl)

# Public async helpers to fetch from TMDB with caching rules
import json

async def fetch_tmdb(path: str, params: dict, cache_key: str = None, ttl: int = 3600):
    # check cache
    if cache_key:
        cached = await get_cached(cache_key)
        if cached:
            return json.loads(cached)
    url = tmdb_url(path, params)
    resp = await async_client.get(url)
    resp.raise_for_status()
    data = resp.json()
    if cache_key:
        await set_cached(cache_key, json.dumps(data), ttl)
    return data

# Specific wrappers used by routes
async def get_trending():
    return await fetch_tmdb("/trending/movie/week", {}, cache_key="tmdb:trending", ttl=3600)

async def get_movie_details(movie_id: int):
    return await fetch_tmdb(f"/movie/{movie_id}", {"append_to_response":"videos,credits,similar"}, cache_key=f"tmdb:movie:{movie_id}", ttl=7*24*3600)

async def search_movies(query: str, page:int=1):
    key = f"tmdb:search:{query.lower()}:{page}"
    return await fetch_tmdb("/search/movie", {"query": query, "page": page}, cache_key=key, ttl=24*3600)

async def get_recommendations(movie_id:int):
    key = f"tmdb:similar:{movie_id}"
    # Use /movie/{id}/similar
    return await fetch_tmdb(f"/movie/{movie_id}/similar", {}, cache_key=key, ttl=3*24*3600)

async def get_genres():
    return await fetch_tmdb("/genre/movie/list", {}, cache_key="tmdb:genres", ttl=30*24*3600)

async def get_top_rated():
    return await fetch_tmdb("/movie/top_rated", {}, cache_key="tmdb:top", ttl=12*3600)

async def get_upcoming():
    return await fetch_tmdb("/movie/upcoming", {}, cache_key="tmdb:upcoming", ttl=12*3600)