import os
from dotenv import load_dotenv

load_dotenv()

# === API KEYS ===
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# === MODEL CONFIG ===
GROQ_MODEL = "llama-3.1-70b-versatile"

 # === SEARCH CONFIG ===
SERPAPI_KEY = os.getenv("SERPAPI_KEY")  

# === VECTOR STORE CONFIG ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "..", "data", "vector_store.faiss")