"""
Configuration file for Three-Tier LLM System
"""
import os

# ===== PROJECT PATHS =====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DOCUMENTS_DIR = os.path.join(DATA_DIR, "documents")
VECTOR_DB_PATH = os.path.join(DATA_DIR, "vector_db")


# Model configurations
SMALL_MODEL = "gemma3:270m"
LARGE_MODEL = "deepseek-r1:1.5b"
EMBEDDING_MODEL = "nomic-embed-text"

# Embedding generation settings
EMBEDDING_BATCH_SIZE = 10

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 100

DB_PATH = "./data/vector_db"

# ===== SIMILARITY THRESHOLDS =====
HIGH_SIMILARITY_THRESHOLD = 0.90   # Use cache (Tier 1)
MEDIUM_SIMILARITY_THRESHOLD = 0.70 # Use small model (Tier 2)
# Below 0.70 uses large model (Tier 3)

# ===== DATABASE SETTINGS =====
COLLECTION_DOCUMENTS = "documents"
COLLECTION_QUESTIONS = "questions_cache"
COLLECTION_HISTORY = "user_history"

# ===== RASPBERRY PI OPTIMIZATIONS =====
MAX_CONCURRENT_REQUESTS = 2
MEMORY_LIMIT_MB = 6000  # Leave 2GB for system

# ===== LOGGING =====
LOG_LEVEL = "INFO"
VERBOSE = True

print(f"📁 Project root: {PROJECT_ROOT}")
print(f"📄 Documents directory: {DOCUMENTS_DIR}")
print(f"💾 Vector DB path: {VECTOR_DB_PATH}")