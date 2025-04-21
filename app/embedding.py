# app/embedding.py
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import Optional, List

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2") # Default model

# Load the model once (can be resource-intensive)
# Consider loading this lazily or managing it within the app's lifespan if memory is a concern
try:
    print(f"Loading embedding model: {MODEL_NAME}...")
    embedding_model = SentenceTransformer(MODEL_NAME)
    print("Embedding model loaded.")
    EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {EMBEDDING_DIM}")
except Exception as e:
    print(f"Failed to load embedding model '{MODEL_NAME}': {e}")
    embedding_model = None
    EMBEDDING_DIM = None # Important for index creation check

def get_embedding(text: str) -> Optional[List[float]]:
    """Generates embedding for a given text."""
    if embedding_model and text:
        vector = embedding_model.encode(text).tolist()
        return vector
    return None

# --- Placeholder for OpenAI (if using) ---
# import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")
# def get_embedding_openai(text: str, model="text-embedding-3-small") -> Optional[List[float]]:
#     if not text: return None
#     try:
#         response = openai.embeddings.create(input=[text], model=model)
#         return response.data[0].embedding
#     except Exception as e:
#         print(f"OpenAI embedding error: {e}")
#         return None