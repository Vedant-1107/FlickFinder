# backend/db/redis.py
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"

if ENV_PATH.exists():
    load_dotenv(ENV_PATH)
else:
    load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")
print(f"[DEBUG] REDIS_URL being used: {REDIS_URL}")

import redis.asyncio as redis
redis_client = redis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)