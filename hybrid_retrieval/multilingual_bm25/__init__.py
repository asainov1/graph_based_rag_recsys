"""
Multilingual BM25 Retriever Package

A retrieval system that supports cross-lingual BM25 search for English and German documents.
"""

from .retriever import MultilingualBM25Retriever
from .data_loader import load_hknews_documents, create_temp_documents
from .utils import save_results, run_interactive_search

__version__ = "1.0.0"
__author__ = "Alikhan"

__all__ = [
    "MultilingualBM25Retriever",
    "load_hknews_documents", 
    "create_temp_documents",
    "save_results",
    "run_interactive_search"
]