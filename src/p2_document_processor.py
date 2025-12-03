# 2_document_processor.py
"""
Contains the DocumentProcessor class for loading and chunking various file types.
Uses SemanticChunker for intelligent splitting.
"""
from typing import List
import pypdf
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

# --- Import Loaders ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


class DocumentProcessor:
    """
    Handles loading a file (PDF or CSV) and splitting it into semantic chunks.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        print(f"DocumentProcessor initialized for: {self.file_path}")

    def _load_from_pdf(self) -> List[Document]:
        """Loads documents from a PDF file."""
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} pages from PDF.")
        return documents

    def _load_from_csv(self) -> List[Document]:
        """Loads documents from a CSV file. Each row becomes a document."""
        loader = CSVLoader(file_path=self.file_path, encoding="utf-8")
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} rows from CSV.")
        return documents

    def load_and_chunk(self, embed_model: Embeddings) -> List[Document]:
        """
        Loads and chunks the document based on its file extension.
        """
        # 1. Load documents based on file type
        file_path_lower = self.file_path.lower()
        documents: List[Document] = []

        if file_path_lower.endswith('.pdf'):
            documents = self._load_from_pdf()
        elif file_path_lower.endswith('.csv'):
            documents = self._load_from_csv()
        else:
            raise ValueError(f"Unsupported file type: {self.file_path}. Only .pdf and .csv are supported.")

        # 2. Chunk the loaded documents (common step)
        if not documents:
            print("No documents loaded, skipping chunking.")
            return []
            
        print("Starting document chunking...")
        # Uses the embedding model for context-aware splitting
        text_splitter = SemanticChunker(embed_model) 
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} semantic chunks.")
        
        return chunks