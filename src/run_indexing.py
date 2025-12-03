"""
Part 1: INDEXING SCRIPT
Handles document loading, chunking, and indexing to the Pinecone vector database.
Run this script ONCE to prepare your RAG system's knowledge base.
"""

import os
import ssl
import sys

import requests
import urllib3

# --- 1. SETUP ENV & IMPORTS ---
try:
    import p1_config as config

    os.environ["GENAI_LLM_API_KEY"] = config.GENAI_LLM_API_KEY
    os.environ["GENAI_EMBED_API_KEY"] = config.GENAI_EMBED_API_KEY
    os.environ["pinecone_api_key"] = config.PINECONE_API_KEY
except ImportError:
    print("Error: Could not import p1_config.py.")
    sys.exit(1)
except AttributeError as ex:
    print(f"Error: A required API key is missing from p1_config.py: {ex}")
    sys.exit(1)

# Project imports
from p2_document_processor import DocumentProcessor
from p3_embeddings import get_openai_embeddings
from p4_retrieval_service import RetrievalAndIndexingService

# --- 2. AGGRESSIVE SSL WORKAROUND (Fixes tiktoken/requests errors) ---
# 1. Fix for standard urllib libraries (Pinecone native client)
# We use getattr/setattr here to access protected members ('_') dynamically,
# which prevents Pylint W0212 errors without requiring a disable comment.
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
    """Main function for indexing documents."""
    print("=" * 50)
    print("🚀 Starting RAG Document Indexing Pipeline")
    print(f"Document Path: {config.FILE_PATH}")
    print("=" * 50)

    # --- 1. Initialize Embedding Model ---
    print("\n--- 1. Initializing Embedding Model ---")
    embed_model = get_openai_embeddings()

    # --- 2. Load and Chunk Documents ---
    print("\n--- 2. Initializing DocumentProcessor (Load and Chunk) ---")
    processor = DocumentProcessor(config.FILE_PATH)
    chunks = processor.load_and_chunk(embed_model)

    if not chunks:
        print("No chunks were created. Indexing aborted.")
        return

    # --- 3. Initialize and Run Indexing Service ---
    # BM25 will be fitted and SAVED here.
    print("\n--- 3. Initializing RetrievalAndIndexingService (FITTING BM25) ---")
    retrieval_service = RetrievalAndIndexingService(
        embedding_model=embed_model, chunks=chunks
    )

    print("\n--- 4. Indexing Documents to Pinecone ---")
    retrieval_service.index_documents()

    print("\n" + "=" * 50)
    print("✅ Indexing Complete! Your RAG knowledge base is ready.")
    print(f"Index Name: {config.PINECONE_INDEX_NAME}")
    print(f"BM25 sparse model saved to {config.SPARSE_MODEL_DUMP_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
