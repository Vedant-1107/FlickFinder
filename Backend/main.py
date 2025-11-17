# backend/main.py
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# load .env first so env vars are available
load_dotenv()

# import your routers (keep same names as your package)
from .routes import auth, movies
# watchlist router we added earlier
from .routes.watchlist import router as watchlist_router

# mongo helper with ensure_indexes
from .db.mongo import ensure_indexes

app = FastAPI(title="MovieRec API")

# configure CORS from env or fallback to Vite default
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# include routers (order doesn't matter much, but keep consistent)
app.include_router(auth.router)
app.include_router(movies.router)
app.include_router(watchlist_router)

@app.on_event("startup")
async def startup_event():
    # create indexes for watchlist collection (idempotent)
    try:
        await ensure_indexes()
    except Exception as e:
        # don't crash on index creation failure, but log it
        print("ensure_indexes failed:", e)

    # debug: list registered routes so you can verify endpoints
    print("Registered routes:")
    for r in app.routes:
        try:
            methods = ",".join(sorted(m for m in r.methods if m not in ("HEAD", "OPTIONS")))
        except Exception:
            methods = ""
        print(f"{r.path}  [{methods}]")

@app.get("/")
async def root():
    return {"message": "MovieRec API — backend running"}
