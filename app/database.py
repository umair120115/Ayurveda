# app/database.py
import os
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from dotenv import load_dotenv
import certifi # Often needed for TLS/SSL connections with Atlas

load_dotenv()

MONGO_URL = os.getenv("MONGODB_URL")
DATABASE_NAME = os.getenv("DATABASE_NAME")

if not MONGO_URL:
    raise ValueError("MONGODB_URL environment variable not set")
if not DATABASE_NAME:
    raise ValueError("DATABASE_NAME environment variable not set")

class Database:
    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None

db_manager = Database()
# connecting with MongoDB
async def connect_to_mongo():
    print("Connecting to MongoDB Atlas...")
    try:
        # Pass tlsCAFile to use certifi's certificate bundle
        db_manager.client = AsyncIOMotorClient(MONGO_URL, tlsCAFile=certifi.where())
        db_manager.db = db_manager.client[DATABASE_NAME]
        await db_manager.client.admin.command('ping') # Verify connection
        print(f"Successfully connected to MongoDB database: {DATABASE_NAME}")
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")
        db_manager.client = None
        db_manager.db = None
        raise ConnectionError(f"Could not connect to MongoDB: {e}") from e
#Closing MongoDB connnection
async def close_mongo_connection():
    print("Closing MongoDB connection...")
    if db_manager.client:
        db_manager.client.close()
        print("MongoDB connection closed.")
    db_manager.client = None
    db_manager.db = None

#Access the database
def get_database() -> AsyncIOMotorDatabase:
    if db_manager.db is None:
        raise ConnectionError("Database not initialized.")
    return db_manager.db


#for getting collection from MongoDB
def get_collection(collection_name: str):
    db = get_database()
    return db[collection_name]