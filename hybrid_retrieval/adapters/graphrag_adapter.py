"""
GraphRAG Adapter for Hybrid Retrieval System
Wraps the existing GraphRAG retriever to provide a unified interface
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ..graphrag.graph_rag_retriever import GraphRAGRetriever
from typing import List, Dict, Any, Tuple


class GraphRAGAdapter:
    """Adapter for GraphRAG retriever to work with hybrid system"""
    
    def __init__(self, graph_path: str = "graph.pickle", 
                 entity_chunks_path: str = "entity_chunks.pickle"):
        """
        Initialize GraphRAG adapter
        
        Args:
            graph_path: Path to pickled knowledge graph
            entity_chunks_path: Path to pickled entity chunks
        """
        self.retriever = GraphRAGRetriever(
            graph_path=graph_path,
            entity_chunks_path=entity_chunks_path
        )
        self.name = "GraphRAG"
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        """
        Retrieve documents using GraphRAG
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of tuples (document_dict, score)
        """
        try:
            # Call the GraphRAG retriever
            results = self.retriever.retrieve(query, top_k=top_k)
            
            # Debug print to understand format
            if results and len(results) > 0:
                print(f"GraphRAG result type: {type(results[0])}, length: {len(results[0]) if hasattr(results[0], '__len__') else 'N/A'}")
            
            standardized_results = []
            
            # Handle different possible return formats
            for idx, result in enumerate(results):
                score = 0.0
                doc_data = {}
                
                # Check if result is a tuple with multiple elements
                if isinstance(result, tuple):
                    if len(result) >= 3:  # (doc, score, metadata) or similar
                        doc_data = result[0] if isinstance(result[0], dict) else {'content': str(result[0])}
                        score = float(result[1]) if len(result) > 1 else 1.0 / (idx + 1)
                    elif len(result) == 2:  # Expected (doc, score) format
                        doc_data = result[0] if isinstance(result[0], dict) else {'content': str(result[0])}
                        score = float(result[1])
                    else:
                        # Single element tuple
                        doc_data = result[0] if isinstance(result[0], dict) else {'content': str(result[0])}
                        score = 1.0 / (idx + 1)
                elif isinstance(result, dict):
                    # Result is already a dictionary
                    doc_data = result
                    score = result.get('score', 1.0 / (idx + 1))
                else:
                    # Result is some other format - convert to string
                    doc_data = {'content': str(result)}
                    score = 1.0 / (idx + 1)
                
                # Ensure we have content
                if 'content' not in doc_data:
                    doc_data['content'] = str(doc_data)
                
                # Create standardized document format
                doc_dict = {
                    'id': doc_data.get('id', f"graphrag_{idx}"),
                    'content': doc_data.get('content', ''),
                    'title': doc_data.get('title', ''),
                    'metadata': doc_data.get('metadata', {}),
                    'source': doc_data.get('source', 'GraphRAG'),
                    'retriever': self.name
                }
                
                standardized_results.append((doc_dict, score))
            
            return standardized_results
            
        except Exception as e:
            print(f"Error in GraphRAG adapter: {e}")
            return []
    
    def get_name(self) -> str:
        """Return the name of this retriever"""
        return self.name

    def get_document_count(self) -> int:
        """Return number of documents/chunks known to GraphRAG"""
        try:
            return len(self.retriever.entity_chunks)
        except Exception as e:
            print(f"⚠️ Could not count GraphRAG documents: {e}")
            return 0