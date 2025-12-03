# p4_retrieval_service.py
"""
Defines the RetrievalAndIndexingService for hybrid search.
Manages Pinecone index setup, hybrid retriever, and indexing.
Handles saving and loading of the fitted BM25 sparse encoder.
"""
import os
import json
import sys
from typing import List, Optional
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone_text.sparse import BM25Encoder
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Import project modules
import p1_config as config

# Get the base directory of the current script to resolve relative paths reliably
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

class RetrievalAndIndexingService:
    """
    Manages Pinecone index setup, hybrid retriever, and indexing.
    """
    # --- FIX 1: Allow chunks to be Optional in type hint ---
    def __init__(self, embedding_model: Embeddings, chunks: Optional[List[Document]] = None):
        self.embedding_model = embedding_model # This is our dense model
        self.chunks = chunks
        
        # --- FIX 2: Handle NoneType for chunks safely ---
        if self.chunks:
            self.corpus = [chunk.page_content for chunk in self.chunks]
            self.metadatas = [chunk.metadata for chunk in self.chunks]
        else:
            self.corpus = []
            self.metadatas = []
        # ------------------------------------------------
        
        # Initialize BM25 Encoder
        self.sparse_model = BM25Encoder()
        
        # Logic to either FIT (Indexing mode) or LOAD (Retrieval mode) BM25
        if self.corpus:
            # Case 1: Chunks exist (Running INDEXING script)
            print("Fitting BM25 sparse model (Indexing Mode)...")
            self.sparse_model.fit(self.corpus)
            self._save_bm25_model()
            print("BM25 sparse model fitted and saved.")
        else:
            # Case 2: Chunks are empty (Running RETRIEVAL script)
            print("No documents provided (Retrieval Mode). Attempting to load saved BM25 model...")
            if not self._load_bm25_model():
                raise RuntimeError(
                    "BM25 sparse model must be fitted and saved by running index_documents.py first. "
                    f"Expected file at: {self._get_bm25_dump_path()}"
                )
            print("BM25 sparse model loaded successfully.")

        # 1. Setup Pinecone connection
        self.pc = Pinecone(api_key=os.environ["pinecone_api_key"],
                           ssl_verify=False)
        self.pinecone_index = self._get_or_create_pinecone_index()

        # 2. Setup PineconeHybridSearchRetriever
        print("Initializing PineconeHybridSearchRetriever...")
        self.pinecone_hybrid_retriever = PineconeHybridSearchRetriever(
            embeddings=self.embedding_model,
            sparse_encoder=self.sparse_model,
            index=self.pinecone_index,
            alpha=0.5, # 0.5 = 50% dense, 50% sparse. Tune as needed.
            top_k=3
        )
        print("PineconeHybridSearchRetriever initialized.")
        
    def _get_bm25_dump_path(self):
        """Constructs the absolute path for the BM25 dump file."""
        return os.path.normpath(os.path.join(BASE_DIR, config.SPARSE_MODEL_DUMP_PATH))

    def _save_bm25_model(self):
        """Saves the fitted BM25 model to disk."""
        path = self._get_bm25_dump_path()
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.sparse_model.dump(path)
        except Exception as e:
            print(f"Warning: Failed to save BM25 model dump at {path}: {e}")

    def _load_bm25_model(self) -> bool:
        """Loads the fitted BM25 model from disk."""
        path = self._get_bm25_dump_path()
        if not os.path.exists(path):
            return False
        
        try:
            self.sparse_model.load(path)
            return True
        except Exception as e:
            print(f"Error: Failed to load BM25 model dump at {path}: {e}")
            return False


    def _get_or_create_pinecone_index(self):
        """Connects to Pinecone and creates the index if it doesn't exist."""
        index_names = [idx_info['name'] for idx_info in self.pc.list_indexes()]
        
        if config.PINECONE_INDEX_NAME not in index_names:
            print(f"Creating new Pinecone index: {config.PINECONE_INDEX_NAME}")
            self.pc.create_index(
                name=config.PINECONE_INDEX_NAME,
                dimension=config.EMBEDDING_DIMENSION, 
                metric="dotproduct", 
                spec=ServerlessSpec(cloud='aws', region='us-east-1')
            )
            print("Index created successfully.")
        else:
            print(f"Index '{config.PINECONE_INDEX_NAME}' already present.")
            
        return self.pc.Index(config.PINECONE_INDEX_NAME)

    def index_documents(self):
        """Adds the chunked documents to the Pinecone hybrid index."""
        if not self.chunks:
            print("No chunks provided for indexing. Skipping index update.")
            return

        print(f"Adding {len(self.chunks)} chunks to Pinecone hybrid index...")
        try:
            self.pinecone_hybrid_retriever.add_texts(
                texts=self.corpus,
                metadatas=self.metadatas
            )
            print("Successfully added documents to Pinecone index.")
        except Exception as e:
            print(f"An error occurred during indexing: {e}")

    def get_hybrid_retriever(self) -> PineconeHybridSearchRetriever:
        """Returns the main hybrid retriever."""
        return self.pinecone_hybrid_retriever