# 1_config.py
"""
Central configuration file for API keys, model names, and file paths.
"""

# --- API KEYS ---
# !IMPORTANT! - Fill in your API keys here
PINECONE_API_KEY = ""

# --- GenAI Lab Keys ---
GENAI_LLM_API_KEY = "" 
GENAI_EMBED_API_KEY = "" 
MAAS_API_KEY = ""
# NEWS_API_KEY removed

# --- FILE PATHS ---
FILE_PATH = "../data/RAG_and_LangChain.pdf"
# Path to save/load the fitted sparse encoder
SPARSE_MODEL_DUMP_PATH = "../data/bm25_values.json" 

# --- PINECONE ---
PINECONE_INDEX_NAME = "rag-agent"

# --- Model Names & Dimensions ---
LLM_MODEL_NAME = "azure/genailab-maas-gpt-35-turbo"
EMBEDDING_MODEL_NAME = "azure/genailab-maas-text-embedding-3-large"
EMBEDDING_DIMENSION = 3072