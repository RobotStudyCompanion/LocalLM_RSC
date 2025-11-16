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

# Tier Configuration
TIER_1_NAME = "cache"
TIER_2_NAME = "1B"
TIER_3_NAME = "4B"

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 100

DB_PATH = "./data/vector_db"

# ===== SIMILARITY THRESHOLDS =====
HIGH_SIMILARITY_THRESHOLD = 0.90   # Use cache (Tier 1)
MEDIUM_SIMILARITY_THRESHOLD = 0.70 # Use small model (Tier 2)
# Below 0.70 uses large model (Tier 3)

# ===== CROSS-ENCODER SETTINGS =====
USE_CROSS_ENCODER = True  # Enable cross-encoder re-ranking for better semantic matching
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Lightweight model for Pi
CROSS_ENCODER_TOP_K = 5  # Number of candidates to re-rank (keep low for performance)
EMBEDDING_WEIGHT = 0.3   # Weight for embedding similarity (0-1)
CROSS_ENCODER_WEIGHT = 0.7  # Weight for cross-encoder score (0-1)

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

# Context Settings
MAX_CONTEXT_ITEMS = 3  # Number of context chunks for LLM
MIN_CONTEXT_RELEVANCE = 0.5  # Minimum similarity to consider context relevant (below this, ignore context)

# Question Generation Settings
AUTO_GENERATE_QUESTIONS = True       # Generate questions on initialization
MAX_CHUNKS_FOR_QUESTIONS = 5        # Max chunks to process (None = all)
QUESTIONS_PER_CHUNK = 2  

print(f"Project root: {PROJECT_ROOT}")
print(f"Documents directory: {DOCUMENTS_DIR}")
print(f"Vector DB path: {VECTOR_DB_PATH}")