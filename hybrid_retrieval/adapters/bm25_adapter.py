"""
BM25 Adapter for Hybrid Retrieval System
Wraps the existing Multilingual BM25 retriever to provide a unified interface
"""
import sys
import os
import shutil
import tempfile
import json
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..multilingual_bm25 import (
    MultilingualBM25Retriever,
    load_hknews_documents,
    create_temp_documents
)

from typing import List, Dict, Any, Tuple
import pickle


class BM25Adapter:
    """Adapter for BM25 retriever to work with hybrid system"""
    
    def __init__(self, docs_directory: str = "data/documents"):
        """
        Initialize BM25 adapter
        
        Args:
            docs_directory: Path to directory containing JSON documents
        """
        docs_path = Path(docs_directory)
        documents: List[Dict[str, Any]] = []

        # Load all .json documents from the folder
        for fn in docs_path.glob("*.json"):
            try:
                with open(fn, "r", encoding="utf-8") as f:
                    doc = json.load(f)

                # Basic cleanup / defaults
                doc.setdefault("id", fn.stem)
                doc.setdefault("title", "")
                doc.setdefault("main_content", doc.get("content", ""))

                documents.append(doc)
            except Exception as e:
                print(f"⚠️ Failed to load {fn}: {e}")

        self.documents = documents
        
        if not documents:
            print("Warning: No documents found in HKNews directory")
            self.retriever = None
            self.temp_dir = None
        else:
            print(f"Loaded {len(documents)} documents")
            
            # Create temporary directory with documents
            self.temp_dir = create_temp_documents(documents)
            
            # Initialize retriever with temp directory
            self.retriever = MultilingualBM25Retriever(self.temp_dir)
            print("BM25 retriever initialized successfully")
        
        self.name = "BM25"
    
    def __del__(self):
        """Clean up temporary directory when adapter is destroyed"""
        if hasattr(self, 'temp_dir') and self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except:
                pass
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve documents using BM25
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of tuples (document_dict, score)
        """
        if self.retriever is None:
            print("BM25 retriever not initialized")
            return []
            
        try:
            # Use the search method
            results = self.retriever.search(query, top_k=top_k)
            
            # Standardize the output format
            standardized_results = []
            
            # Handle different possible result formats
            if isinstance(results, list):
                for idx, result in enumerate(results):
                    if isinstance(result, tuple) and len(result) == 2:
                        # Expected format: (doc_info, score)
                        doc_info, score = result
                    elif isinstance(result, dict):
                        # Result is a dictionary
                        doc_info = result
                        score = result.get('score', 1.0 / (idx + 1))
                    else:
                        # Unknown format
                        doc_info = {'content': str(result)}
                        score = 1.0 / (idx + 1)
                    
                    # Create standardized document
                    doc_dict = {
                        'id': doc_info.get('id', f"bm25_{idx}"),
                        'content': doc_info.get('content', doc_info.get('text', '')),
                        'title': doc_info.get('title', ''),
                        'metadata': {
                            'language': doc_info.get('language', 'unknown'),
                            'source': doc_info.get('source_file', doc_info.get('source', '')),
                            'date': doc_info.get('date', ''),
                            'keywords': doc_info.get('keywords', []),
                            'topics': doc_info.get('topics', [])
                        },
                        'retriever': self.name
                    }
                    standardized_results.append((doc_dict, float(score)))
            
            return standardized_results
            
        except Exception as e:
            print(f"Error in BM25 adapter: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_name(self) -> str:
        """Return the name of this retriever"""
        return self.name
    
    def get_document_count(self) -> int:
        """Return number of loaded documents"""
        return len(self.documents) if hasattr(self, 'documents') and self.documents else 0