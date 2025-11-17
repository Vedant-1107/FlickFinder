# backend/db/mongo.py  (update)
import os
from pathlib import Path
from dotenv import load_dotenv

# ensure we load Backend/.env explicitly
BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()  # fallback to any .env in cwd

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "movies_auth_db")

# debug print (you can remove after confirming)
print(f"[DEBUG] MONGO_URI (first 80 chars): {MONGO_URI[:80] if MONGO_URI else 'None'}")

from motor.motor_asyncio import AsyncIOMotorClient
client = AsyncIOMotorClient(MONGO_URI)
db = client[MONGO_DB]

# existing users collection (you already have this)
users_collection = db.get_collection("users")

# new: watchlists collection
watchlists_collection = db.get_collection("watchlists")

# create useful indexes (idempotent)
# - unique compound index to prevent duplicates
# - index for fast lookup by user_id
async def ensure_indexes():
    await watchlists_collection.create_index(
        [("user_id", 1), ("movie_id", 1)],
        unique=True,
        name="user_movie_unique"
    )
    await watchlists_collection.create_index("user_id", name="user_idx")

# If you call ensure_indexes() at app startup it will run once.
