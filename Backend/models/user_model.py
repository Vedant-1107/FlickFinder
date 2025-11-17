# backend/models/user_model.py
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from bson import ObjectId

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)

class UserInDB(BaseModel):
    id: Optional[str]
    username: str
    email: EmailStr
    password_hash: str

class UserOut(BaseModel):
    id: str
    username: str
    email: EmailStr