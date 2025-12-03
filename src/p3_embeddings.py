# p3_embeddings.py
"""
Initializes the single embedding model for the application.
"""
import httpx
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings

# Import config
import p1_config as config

def get_openai_embeddings() -> Embeddings:
    """
    Initializes and returns the OpenAIEmbeddings model configured 
    for GenAI Lab with BOTH sync and async SSL bypass.
    """
    print(f"Initializing OpenAIEmbeddings model: {config.EMBEDDING_MODEL_NAME}")
    
    # 1. Sync Client (for standard blocking calls)
    http_client = httpx.Client(verify=False)
    
    # 2. Async Client (CRITICAL for RAGAs evaluation)
    http_async_client = httpx.AsyncClient(verify=False)

    # Create the embedding model instance
    embedding_model = OpenAIEmbeddings(
        base_url="https://genailab.tcs.in",
        model=config.EMBEDDING_MODEL_NAME,
        api_key=config.GENAI_EMBED_API_KEY,
        http_client=http_client,
        http_async_client=http_async_client # <--- NEW ADDITION
    )
    
    print("OpenAIEmbeddings model initialized.")
    return embedding_model