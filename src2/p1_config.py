# p1_config.py
"""
Central configuration file for API keys, model names, and file paths.
"""

# --- API KEYS ---

# --- GenAI Lab Keys ---
MAAS_API_KEY = "".strip() 


# --- Model Names & Dimensions ---
LLM_MODEL_NAME = "azure/genailab-maas-gpt-4o-mini"
EMBEDDING_MODEL_NAME = "azure/genailab-maas-text-embedding-3-large"
EMBEDDING_DIMENSION = 3072

# --- Vision and Audio Models (New/Updated) ---
VISION_MODEL_NAME = "azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct" 
AUDIO_MODEL_NAME = "azure/genailab-maas-whisper"