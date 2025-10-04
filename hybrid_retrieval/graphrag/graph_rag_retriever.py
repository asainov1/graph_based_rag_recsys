import networkx as nx
import pickle
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict
import re
from langchain_huggingface import HuggingFaceEmbeddings

class GraphRAGRetriever:
    """
    Implements Local-to-Global retrieval strategy using a knowledge graph.
    
    This retriever:
    1. Uses vector similarity for initial (local) retrieval
    2. Expands results through graph traversal (global)
    3. Re-ranks combined results
    """
    
    def __init__(self, graph_path="graph.pickle", entity_chunks_path="entity_chunks.pickle"):
        """
        Initialize the GraphRAG retriever.
        
        Args:
            graph_path: Path to pickled knowledge graph
            entity_chunks_path: Path to pickled entity chunks
        """
        # Load knowledge graph
        self._load_graph(graph_path)
        
        # Load entity chunks
        self._load_entity_chunks(entity_chunks_path)
        
        # Initialize embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-large",
            model_kwargs={'device': 'cpu'}
        )
        
        # Cache for embeddings
        self.embedding_cache = {}
        
        # Generate embeddings for all articles
        self._generate_article_embeddings()
    
    def _load_graph(self, graph_path):
        """Load knowledge graph from pickle file."""
        print(f"Loading knowledge graph from {graph_path}...")
        try:
            with open(graph_path, "rb") as f:
                self.graph = pickle.load(f)
            print(f"Loaded graph with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
        except Exception as e:
            print(f"Error loading graph: {e}")
            self.graph = nx.Graph()
    
    def _load_entity_chunks(self, entity_chunks_path):
        """Load entity chunks from pickle file."""
        print(f"Loading entity chunks from {entity_chunks_path}...")
        try:
            with open(entity_chunks_path, "rb") as f:
                self.entity_chunks = pickle.load(f)
            print(f"Loaded {len(self.entity_chunks)} entity chunks")
        except Exception as e:
            print(f"Error loading entity chunks: {e}")
            self.entity_chunks = {}
    
    def _generate_article_embeddings(self):
        """Generate embeddings for all articles in the graph."""
        print("Generating article embeddings...")
        article_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'article']
        
        for article_node in article_nodes:
            # Get article text
            article_data = self.graph.nodes[article_node]
            article_text = article_data.get('title', '')
            
            # Skip if no text
            if not article_text:
                continue
            
            # Generate embedding
            try:
                embedding = self.embedding_model.embed_query(article_text)
                self.embedding_cache[article_node] = embedding
            except Exception as e:
                print(f"Error generating embedding for {article_node}: {e}")
        
        print(f"Generated embeddings for {len(self.embedding_cache)} articles")
    
    def detect_language(self, query):
        """
        Detect query language using simple heuristics.
        
        Args:
            query: The user query
            
        Returns:
            Language code ('en' or 'de')
        """
        # Simple heuristic: Check for German-specific characters
        german_chars = "äöüßÄÖÜ"
        if any(c in query for c in german_chars):
            return "de"
        
        # Default to English
        return "en"
    
    def retrieve(self, query, top_k=5):
        """
        Retrieve documents using Local-to-Global strategy.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        print(f"Processing query: {query}")
        
        # Detect query language
        language = self.detect_language(query)
        print(f"Detected language: {language}")
        
        # Phase 1: Local Retrieval (Vector Similarity)
        local_results = self._local_retrieval(query, top_k=top_k)
        print(f"Local retrieval found {len(local_results)} results")
        
        # Phase 2: Global Expansion (Graph Traversal)
        expanded_results = self._global_expansion(local_results, query, top_k=top_k*2)
        print(f"Global expansion found {len(expanded_results)} results")
        
        # Phase 3: Re-ranking
        final_results = self._rerank_results(expanded_results, query)
        print(f"Returning {min(top_k, len(final_results))} final results")
        
        return final_results[:top_k]
    
    def _local_retrieval(self, query, top_k=5):
        """
        Perform initial retrieval using vector similarity.
        
        Args:
            query: User query
            top_k: Number of results to return
            
        Returns:
            List of retrieved documents
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Calculate similarity with all articles
        similarities = {}
        for article_id, article_embedding in self.embedding_cache.items():
            similarity = self._cosine_similarity(query_embedding, article_embedding)
            similarities[article_id] = similarity
        
        # Sort by similarity (descending)
        sorted_articles = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        # Get top_k articles
        local_results = []
        for article_id, score in sorted_articles[:top_k]:
            article_data = self.graph.nodes[article_id]
            
            # Skip if no title
            if 'title' not in article_data:
                continue
            
            result = {
                "id": article_id,
                "title": article_data.get('title', ''),
                "language": article_data.get('language', ''),
                "text": article_data.get('title', ''),  # Using title as placeholder text
                "score": float(score),
                "retrieval_method": "local"
            }
            
            local_results.append(result)
        
        return local_results
    
    def _global_expansion(self, local_results, query, max_hops=2, top_k=10):
        """
        Expand results through graph traversal.
        
        Args:
            local_results: Initial locally retrieved results
            query: User query
            max_hops: Maximum number of hops in the graph
            top_k: Maximum number of results to return
            
        Returns:
            List of expanded results
        """
        expanded_results = local_results.copy()
        visited_nodes = set(r["id"] for r in local_results)
        
        # Query embedding for similarity calculation
        query_embedding = self.embedding_model.embed_query(query)
        
        # For each local result, expand through the graph
        for local_result in local_results:
            article_id = local_result["id"]
            
            # Start BFS from this article
            current_level = [article_id]
            for hop in range(max_hops):
                next_level = []
                
                for node_id in current_level:
                    # Get neighbors
                    for neighbor in self.graph.neighbors(node_id):
                        if neighbor in visited_nodes:
                            continue
                        
                        visited_nodes.add(neighbor)
                        
                        # Handle different node types differently
                        node_type = self.graph.nodes[neighbor].get('type')
                        
                        if node_type == 'article':
                            # Another article node
                            article_data = self.graph.nodes[neighbor]
                            
                            # Skip if no title
                            if 'title' not in article_data:
                                continue
                            
                            # Calculate score (using cached embedding if available)
                            score = 0.5  # Default score for expanded results
                            if neighbor in self.embedding_cache:
                                article_embedding = self.embedding_cache[neighbor]
                                score = self._cosine_similarity(query_embedding, article_embedding)
                                score = score / (hop + 2)  # Reduce score based on hop distance
                            
                            result = {
                                "id": neighbor,
                                "title": article_data.get('title', ''),
                                "language": article_data.get('language', ''),
                                "text": article_data.get('title', ''),
                                "score": float(score),
                                "retrieval_method": "graph_expansion",
                                "expansion_level": hop + 1
                            }
                            
                            expanded_results.append(result)
                            
                        elif node_type == 'entity':
                            # Entity node - get its contexts
                            entity_data = self.graph.nodes[neighbor]
                            entity_text = entity_data.get('text', '')
                            
                            # Get entity chunks if available
                            if neighbor in self.entity_chunks:
                                contexts = self.entity_chunks[neighbor]
                                
                                for context in contexts:
                                    article_id = context.get('article_id')
                                    article_node_id = f"article_{article_id}"
                                    
                                    # Skip if already visited
                                    if article_node_id in visited_nodes:
                                        continue
                                    
                                    visited_nodes.add(article_node_id)
                                    
                                    # Get article data
                                    if article_node_id in self.graph.nodes:
                                        article_data = self.graph.nodes[article_node_id]
                                        
                                        # Calculate score
                                        score = 0.4  # Default score for entity expansion
                                        
                                        result = {
                                            "id": article_node_id,
                                            "title": article_data.get('title', ''),
                                            "language": article_data.get('language', ''),
                                            "text": context.get('text', ''),
                                            "score": float(score),
                                            "retrieval_method": "entity_expansion",
                                            "expansion_level": hop + 1,
                                            "entity": entity_text
                                        }
                                        
                                        expanded_results.append(result)
                        
                        # Add to next level for BFS
                        next_level.append(neighbor)
                
                # Update for next hop
                current_level = next_level
        
        return expanded_results
    
    def _rerank_results(self, results, query):
        """
        Re-rank results based on query relevance.
        
        Args:
            results: List of retrieved results
            query: User query
            
        Returns:
            Re-ranked list of results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.embed_query(query)
        
        # Calculate relevance scores
        for result in results:
            # Text to compare - use full text if available, otherwise title
            text = result.get("text", result.get("title", ""))
            
            if text:
                # Generate text embedding
                text_embedding = self.embedding_model.embed_query(text)
                
                # Calculate similarity
                similarity = self._cosine_similarity(query_embedding, text_embedding)
                
                # Adjust score based on retrieval method and expansion level
                if result.get("retrieval_method") == "local":
                    # Local results get a boost
                    adjusted_score = similarity * 1.2
                elif "expansion_level" in result:
                    # Penalize based on expansion level
                    level = result["expansion_level"]
                    adjusted_score = similarity / (level + 1)
                else:
                    adjusted_score = similarity
                
                # Update score
                result["score"] = float(adjusted_score)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return results
    
    def _cosine_similarity(self, a, b):
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0
            
        return dot_product / (norm_a * norm_b)