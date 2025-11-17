# backend/routes/watchlist.py
from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import jwt  # pip install pyjwt
from bson import ObjectId

from ..db.mongo import users_collection, watchlists_collection  # update import path if needed

router = APIRouter()

JWT_SECRET = os.getenv("JWT_SECRET", "change_me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# Pydantic models
class WatchlistItemIn(BaseModel):
    movie_id: int

class WatchlistItemOut(BaseModel):
    movie_id: int
    added_at: datetime

# helper: decode token and return user doc
async def get_current_user(request: Request):
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing auth token")
    token = auth.split(" ", 1)[1].strip()
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    # Expect the token to include a 'sub' or 'user_id' claim; fallback to 'email'
    user_identifier = payload.get("sub") or payload.get("user_id") or payload.get("email")
    if not user_identifier:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token missing user claim")

    # Try _id first (if token contains it) else try email/username
    user = None
    try:
        # if an ObjectId string present
        if ObjectId.is_valid(str(user_identifier)):
            user = await users_collection.find_one({"_id": ObjectId(str(user_identifier))})
    except Exception:
        user = None

    if not user:
        # try email or username
        user = await users_collection.find_one({"email": user_identifier}) or await users_collection.find_one({"username": user_identifier})

    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    # normalize id as string
    user["_id_str"] = str(user["_id"])
    return user

# GET /api/watchlist
@router.get("/api/watchlist", response_model=List[WatchlistItemOut])
async def get_watchlist(current_user: dict = Depends(get_current_user)):
    user_id = current_user["_id_str"]
    cursor = watchlists_collection.find({"user_id": user_id}).sort("added_at", -1)
    items = []
    async for doc in cursor:
        items.append(WatchlistItemOut(movie_id=int(doc["movie_id"]), added_at=doc.get("added_at")))
    return items

# POST /api/watchlist
@router.post("/api/watchlist", status_code=201)
async def add_watchlist_item(payload: WatchlistItemIn, current_user: dict = Depends(get_current_user)):
    user_id = current_user["_id_str"]
    movie_id = int(payload.movie_id)
    doc = {
        "user_id": user_id,
        "movie_id": movie_id,
        "added_at": datetime.utcnow()
    }
    try:
        await watchlists_collection.insert_one(doc)
    except Exception as e:
        # if duplicate due to unique index, return 200 OK or 409 depending on preference
        if "duplicate" in str(e).lower():
            return {"message": "Already in watchlist"}
        raise HTTPException(status_code=500, detail="Could not add to watchlist")
    return {"message": "added", "movie_id": movie_id}

# DELETE /api/watchlist/{movie_id}
@router.delete("/api/watchlist/{movie_id}")
async def remove_watchlist_item(movie_id: int, current_user: dict = Depends(get_current_user)):
    user_id = current_user["_id_str"]
    res = await watchlists_collection.delete_one({"user_id": user_id, "movie_id": int(movie_id)})
    if res.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Item not found in watchlist")
    return {"message": "removed", "movie_id": int(movie_id)}