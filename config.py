"""
Configuration file for Three-Tier LLM System
"""

# Model configurations
SMALL_MODEL = "gemma3:270m"
LARGE_MODEL = "deepseek-r1:1.5b"
EMBEDDING_MODEL = "nomic-embed-text"

# Chunking parameters
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 100

DB_PATH = "./data/vector_db"