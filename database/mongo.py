from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
client = AsyncIOMotorClient(MONGO_URI)

db = client.techmart_db
users_collection = db.users
sessions_collection = db.sessions
messages_collection = db.messages

async def init_db():
    # Create indexes if needed
    await users_collection.create_index("email", unique=True)
    print("Database initialized")
