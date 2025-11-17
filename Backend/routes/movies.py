# backend/routes/movies.py
from fastapi import APIRouter, HTTPException, Query
from ..utils.tmdb_client import (
    get_trending, get_movie_details, search_movies,
    get_recommendations, get_genres, get_top_rated, get_upcoming
)

router = APIRouter(prefix="/api", tags=["movies"])

@router.get("/trending")
async def trending():
    data = await get_trending()
    return data

@router.get("/movie/{movie_id}")
async def movie_detail(movie_id: int):
    data = await get_movie_details(movie_id)
    return data

@router.get("/search")
async def search(q: str = Query(..., min_length=1), page: int = 1):
    data = await search_movies(q, page)
    return data

@router.get("/recommendations/{movie_id}")
async def recommendations(movie_id: int):
    data = await get_recommendations(movie_id)
    return data

@router.get("/genres")
async def genres():
    data = await get_genres()
    return data

@router.get("/top")
async def top():
    data = await get_top_rated()
    return data

@router.get("/upcoming")
async def upcoming():
    data = await get_upcoming()
    return data