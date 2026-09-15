from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import auth, chat
from database.mongo import init_db
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="TechMart Customer Support API")

# Setup CORS for Vercel Frontend communication
frontend_url = os.getenv("FRONTEND_URL", "https://ai-f.vercel.app")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_db_client():
    await init_db()

@app.get("/")
def read_root():
    return {"message": "TechMart Backend API is running."}

app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat"])
