# p2_document_processor.py
"""
Contains the DocumentProcessor class for loading and chunking various file types.
Uses SemanticChunker for PDFs and Row-based chunking for CSVs.
"""
from typing import List
# pypdf is used internally by PyPDFLoader, ensuring it's available
import pypdf 
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

# --- Import Loaders ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


class DocumentProcessor:
    """
    Handles loading a file (PDF or CSV) and splitting it into chunks.
    Strategy:
    - PDF: Semantic Chunking (using Embeddings)
    - CSV: 1 Row = 1 Chunk (Native CSVLoader behavior)
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
        file_path_lower = self.file_path.lower()

        # --- CASE 1: CSV FILES (Fast, Row-based) ---
        if file_path_lower.endswith('.csv'):
            print("Processing CSV: Treating each row as a single chunk.")
            # CSVLoader automatically makes 1 Document per Row.
            # We return these directly without further splitting.
            return self._load_from_csv()

        # --- CASE 2: PDF FILES (Smart, Semantic) ---
        elif file_path_lower.endswith('.pdf'):
            print("Processing PDF: Applying Semantic Chunking.")
            documents = self._load_from_pdf()
            
            if not documents:
                print("No PDF documents loaded, skipping chunking.")
                return []
            
            print("Starting semantic chunking for PDF (this may take a moment)...")
            # Uses the embedding model for context-aware splitting
            text_splitter = SemanticChunker(embed_model) 
            chunks = text_splitter.split_documents(documents)
            print(f"Split PDF into {len(chunks)} semantic chunks.")
            return chunks

        # --- ERROR CASE ---
        else:
            raise ValueError(f"Unsupported file type: {self.file_path}. Only .pdf and .csv are supported.")