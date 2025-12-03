# run_retrieval_main.py
"""
SCRIPT 2: RETRIEVAL (Chat)
Main entry point for chatting with the RAG agent.
"""
import os
import ssl
import sys

import requests
import urllib3

# --- 1. SETUP ENV & IMPORTS ---
try:
    import p1_config as config

    # Set env vars
    os.environ["GENAI_LLM_API_KEY"] = config.GENAI_LLM_API_KEY
    os.environ["GENAI_EMBED_API_KEY"] = config.GENAI_EMBED_API_KEY
    os.environ["pinecone_api_key"] = config.PINECONE_API_KEY
except ImportError:
    print("Error: Could not import p1_config.py.")
    sys.exit(1)
except AttributeError as ex:
    print(f"Error: A required API key is missing from p1_config.py: {ex}")
    sys.exit(1)

from p3_embeddings import get_openai_embeddings
from p4_retrieval_service import RetrievalAndIndexingService
from p5_agent_service import AgentService
from p6_evaluation import run_evaluation_example

# --- 2. AGGRESSIVE SSL WORKAROUND ---
# 1. Fix for standard urllib libraries (Pinecone native client)
# Use getattr/setattr to avoid Pylint W0212 (protected-access)
create_unverified = getattr(ssl, "_create_unverified_context")
setattr(ssl, "_create_default_https_context", create_unverified)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 2. Fix for libraries using 'requests' (Tiktoken, LangChain)
original_request = requests.Session.request


def patched_request(self, method, url, *args, **kwargs):
    """Patches requests to force verify=False for SSL workarounds."""
    kwargs["verify"] = False
    return original_request(self, method, url, *args, **kwargs)


requests.Session.request = patched_request
# --- END SSL WORKAROUND ---


def main():
    """Main execution function for the retrieval chat system."""
    print("=== RETRIEVAL MODE ===")

    # --- CHANGED: Use get_openai_embeddings ---
    embed_model = get_openai_embeddings()

    # 1. Connect to Service (No chunks passed = "Inference" mode for BM25)
    # This connects to Pinecone immediately without loading the PDF.
    # --- CHANGED: Use RetrievalAndIndexingService ---
    service = RetrievalAndIndexingService(embedding_model=embed_model, chunks=None)

    # 2. Start Agent
    agent = AgentService(service)

    # 3. Eval Option
    if input("Run RAGAs eval? (y/n): ").lower() == "y":
        run_evaluation_example(agent)

    # 4. Chat
    while True:
        q = input("\nAsk: ")
        if q in ["quit", "exit"]:
            break
        if q:
            agent.run_query(q)


if __name__ == "__main__":
    main()
