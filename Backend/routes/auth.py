# backend/routes/auth.py
from fastapi import APIRouter, HTTPException, status, Depends, Request
from pydantic import BaseModel
from passlib.context import CryptContext
from ..db.mongo import users_collection
from ..models.user_model import UserCreate, UserOut
from ..utils.jwt_handler import create_access_token, decode_access_token
from bson.objectid import ObjectId
import asyncio

router = APIRouter(prefix="/auth", tags=["auth"])
pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

class LoginSchema(BaseModel):
    email: str
    password: str

@router.post("/signup", response_model=UserOut)
async def signup(payload: UserCreate):
    # check if email exists
    existing = await users_collection.find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    password_hash = pwd_context.hash(payload.password)
    user_doc = {
        "username": payload.username,
        "email": payload.email,
        "password_hash": password_hash
    }
    res = await users_collection.insert_one(user_doc)
    user_out = UserOut(id=str(res.inserted_id), username=payload.username, email=payload.email)
    return user_out

@router.post("/login")
async def login(payload: LoginSchema):
    user = await users_collection.find_one({"email": payload.email})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not pwd_context.verify(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token({"sub": str(user["_id"]), "email": user["email"], "username": user["username"]})
    return {"access_token": token, "token_type": "bearer", "user": {"id": str(user["_id"]), "username": user["username"], "email": user["email"]}}

# Protected endpoint example
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid auth token")
    user_id = payload.get("sub")
    user = await users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {"id": str(user["_id"]), "username": user["username"], "email": user["email"]}

@router.get("/me")
async def me(user=Depends(get_current_user)):
    return user