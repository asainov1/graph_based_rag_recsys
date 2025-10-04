"""
Complete Multi-Model Re-ranking implementation for Step 2.2 requirement
"""
from sentence_transformers import CrossEncoder
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import json


class MultiModelReranker:
    """Complete reranking system with multiple models and approaches"""
    
    def __init__(self):
        """Initialize multiple reranking models"""
        print("Initializing Multi-Model Reranker System...")
        
        # Cross-encoder models
        self.models = {}
        
        # Model 1: TinyBERT (fast, lightweight) - already working
        try:
            self.models['tinybert'] = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
            print("TinyBERT model loaded (fast)")
        except Exception as e:
            print(f"✗ TinyBERT failed: {e}")
        
        # Model 2: MiniLM (balanced performance)
        try:
            self.models['minilm'] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("MiniLM model loaded (balanced)")
        except Exception as e:
            print(f"✗ MiniLM failed: {e}")
            
        # Model 3: BGE Reranker (state-of-the-art)
        try:
            self.models['bge'] = CrossEncoder('BAAI/bge-reranker-base')
            print("BGE reranker loaded (sota)")
        except Exception as e:
            print(f"✗ BGE failed: {e}")
        
        # Commercial API support
        self.cohere_api_key = "NXoFgAsWJ4nXzhpuZqBQ9R8gaIIWWbmxqcc0c7tt"
        
        print(f"Loaded {len(self.models)} cross-encoder models")
        self.available_methods = list(self.models.keys()) + ['ensemble', 'summary_fusion', 'keyword_boost']
        if self.cohere_api_key:
            self.available_methods.append('cohere')
    
    def set_cohere_key(self, api_key: str):
        """Set Cohere API key for commercial reranking"""
        self.cohere_api_key = api_key
        if 'cohere' not in self.available_methods:
            self.available_methods.append('cohere')
        print("Cohere API key configured")
    
    def rerank_single_model(self, query: str, results: List[Tuple[Dict, float]], 
                           model_name: str = 'tinybert', top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Rerank using a single cross-encoder model"""
        
        if model_name not in self.models:
            print(f"Model {model_name} not available. Available: {list(self.models.keys())}")
            return results[:top_k]
        
        model = self.models[model_name]
        documents = [doc for doc, _ in results]
        
        # Prepare query-document pairs
        pairs = []
        for doc in documents:
            content = self._extract_content(doc)
            if len(content) > 512:
                content = content[:512]
            pairs.append([query, content])
        
        # Get scores and rerank
        scores = model.predict(pairs)
        scored_results = [(documents[i], float(scores[i])) for i in range(len(documents))]
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Add metadata
        for doc, score in scored_results[:top_k]:
            doc['rerank_score'] = score
            doc['rerank_method'] = f'cross_encoder_{model_name}'
        
        return scored_results[:top_k]
    
    def rerank_ensemble(self, query: str, results: List[Tuple[Dict, float]], 
                       top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Ensemble reranking using multiple models with RRF fusion"""
        
        if len(self.models) < 2:
            print("Need at least 2 models for ensemble. Using single model.")
            return self.rerank_single_model(query, results, list(self.models.keys())[0], top_k)
        
        # Get rankings from each model
        model_rankings = {}
        
        for model_name in self.models.keys():
            try:
                ranked = self.rerank_single_model(query, results, model_name, len(results))
                # Create rank mapping: doc_id -> rank
                model_rankings[model_name] = {}
                for rank, (doc, _) in enumerate(ranked):
                    doc_id = doc.get('id', str(hash(str(doc))))
                    model_rankings[model_name][doc_id] = rank
            except Exception as e:
                print(f"Model {model_name} failed in ensemble: {e}")
        
        if not model_rankings:
            return results[:top_k]
        
        # Calculate RRF scores
        doc_rrf_scores = {}
        documents_dict = {}
        
        for doc, _ in results:
            doc_id = doc.get('id', str(hash(str(doc))))
            documents_dict[doc_id] = doc
            doc_rrf_scores[doc_id] = 0
        
        # RRF formula: score = sum(1/(k + rank)) for each model
        k = 60  # RRF constant
        for doc_id in doc_rrf_scores.keys():
            for model_name, rankings in model_rankings.items():
                rank = rankings.get(doc_id, len(results))  # Default to last if not found
                doc_rrf_scores[doc_id] += 1 / (k + rank)
        
        # Sort by RRF score
        sorted_docs = sorted(doc_rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create final ranking
        reranked = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            doc = documents_dict[doc_id].copy()
            doc['rerank_score'] = rrf_score
            doc['rerank_method'] = f'ensemble_rrf_{len(model_rankings)}models'
            reranked.append((doc, rrf_score))
        
        return reranked
    
    def rerank_summary_fusion(self, query: str, results: List[Tuple[Dict, float]], 
                             top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Summary-based reranking using content overlap"""
        
        query_words = set(query.lower().split())
        scored_results = []
        
        for doc, original_score in results:
            content = self._extract_content(doc).lower()
            
            # Calculate different relevance signals
            content_words = set(content.split())
            title_words = set(doc.get('title', '').lower().split())
            
            # 1. Query-content overlap
            content_overlap = len(query_words.intersection(content_words))
            title_overlap = len(query_words.intersection(title_words))
            
            # 2. Term frequency scoring
            query_term_freq = sum(content.count(word) for word in query_words)
            
            # 3. Combined score
            overlap_score = (content_overlap + 2 * title_overlap) / len(query_words) if query_words else 0
            freq_score = min(query_term_freq / 10, 1.0)  # Normalize
            
            # Weighted combination
            summary_score = 0.4 * overlap_score + 0.3 * freq_score + 0.3 * original_score
            
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = summary_score
            doc_copy['rerank_method'] = 'summary_fusion'
            
            scored_results.append((doc_copy, summary_score))
        
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]
    
    def rerank_keyword_boost(self, query: str, results: List[Tuple[Dict, float]], 
                            top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Keyword-based boosting for research-specific terms"""
        
        # Define research-related boost terms
        research_terms = {
            'research', 'study', 'analysis', 'investigation', 'experiment',
            'artificial intelligence', 'machine learning', 'AI', 'ML',
            'professor', 'researcher', 'scientist', 'laboratory', 'institute'
        }
        
        scored_results = []
        query_lower = query.lower()
        
        for doc, original_score in results:
            content = self._extract_content(doc).lower()
            title = doc.get('title', '').lower()
            
            # Calculate boost factor
            boost_factor = 1.0
            
            # Boost for research terms
            for term in research_terms:
                if term in content or term in title:
                    boost_factor += 0.1
            
            # Extra boost for query terms in title
            query_words = query_lower.split()
            title_matches = sum(1 for word in query_words if word in title)
            if title_matches > 0:
                boost_factor += 0.2 * title_matches
            
            # Apply boost
            boosted_score = original_score * boost_factor
            
            doc_copy = doc.copy()
            doc_copy['rerank_score'] = boosted_score
            doc_copy['rerank_method'] = f'keyword_boost_{boost_factor:.2f}x'
            
            scored_results.append((doc_copy, boosted_score))
        
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results[:top_k]
    
    def rerank_cohere(self, query: str, results: List[Tuple[Dict, float]], 
                     top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Rerank using Cohere API (requires API key)"""
        
        if not self.cohere_api_key:
            print("Cohere API key not set. Use set_cohere_key() first.")
            return results[:top_k]
        
        try:
            import cohere
            co = cohere.Client(self.cohere_api_key)
            
            documents = [doc for doc, _ in results]
            docs_text = [self._extract_content(doc) for doc in documents]
            
            response = co.rerank(
                model="rerank-english-v2.0",
                query=query,
                documents=docs_text,
                top_k=min(top_k, len(docs_text))
            )
            
            # Reorder based on Cohere ranking
            reranked = []
            for result in response.results:
                doc = documents[result.index].copy()
                doc['rerank_score'] = result.relevance_score
                doc['rerank_method'] = 'cohere_api'
                reranked.append((doc, result.relevance_score))
            
            return reranked
            
        except ImportError:
            print("Cohere library not installed. Install with: pip install cohere")
            return results[:top_k]
        except Exception as e:
            print(f"Cohere reranking failed: {e}")
            return results[:top_k]
    
    def rerank(self, query: str, results: List[Tuple[Dict, float]], 
              method: str = 'tinybert', top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Main reranking method with multiple approaches"""
        
        if not results:
            return []
        
        print(f"Reranking {len(results)} documents using method: {method}")
        
        if method in self.models:
            return self.rerank_single_model(query, results, method, top_k)
        elif method == 'ensemble':
            return self.rerank_ensemble(query, results, top_k)
        elif method == 'summary_fusion':
            return self.rerank_summary_fusion(query, results, top_k)
        elif method == 'keyword_boost':
            return self.rerank_keyword_boost(query, results, top_k)
        elif method == 'cohere':
            return self.rerank_cohere(query, results, top_k)
        else:
            print(f"Unknown method: {method}. Available: {self.available_methods}")
            return self.rerank_single_model(query, results, list(self.models.keys())[0], top_k)
    
    def compare_methods(self, query: str, results: List[Tuple[Dict, float]], 
                       methods: List[str] = None, top_k: int = 5) -> Dict[str, List[Tuple[Dict, float]]]:
        """Compare multiple reranking methods"""
        
        if methods is None:
            methods = self.available_methods[:4]  # Test first 4 methods
        
        comparison = {}
        
        for method in methods:
            if method in self.available_methods:
                try:
                    reranked = self.rerank(query, results, method, top_k)
                    comparison[method] = reranked
                except Exception as e:
                    print(f"Method {method} failed: {e}")
        
        return comparison
    
    def _extract_content(self, doc: Dict) -> str:
        """Extract text content from document"""
        content_fields = ['content', 'text', 'chunk_text', 'body']
        for field in content_fields:
            if field in doc and doc[field]:
                return str(doc[field])
        return doc.get('title', str(doc))
    
    def get_available_methods(self) -> List[str]:
        """Get list of available reranking methods"""
        return self.available_methods.copy()


# Backward compatibility - keep SimpleReranker
class SimpleReranker:
    """Simple wrapper for backward compatibility"""
    
    def __init__(self, model_name: str = 'cross-encoder/ms-marco-TinyBERT-L-2-v2'):
        self.multi_reranker = MultiModelReranker()
        self.default_method = 'tinybert'
    
    def rerank(self, query: str, results: List[Tuple[Dict, float]], top_k: int = 10) -> List[Tuple[Dict, float]]:
        return self.multi_reranker.rerank(query, results, self.default_method, top_k)


# Test function
if __name__ == "__main__":
    # Test the multi-model reranker
    reranker = MultiModelReranker()
    
    query = "artificial intelligence research at ETH"
    sample_results = [
        ({"content": "ETH Zurich artificial intelligence research in drug discovery", "title": "AI Drug Research", "id": "1"}, 0.5),
        ({"content": "Weather forecast for Zurich this weekend", "title": "Weather Update", "id": "2"}, 0.3),
        ({"content": "ETH computer science department machine learning", "title": "CS ML Department", "id": "3"}, 0.4),
        ({"content": "Professor Andreas Krause leads AI research group", "title": "AI Professor", "id": "4"}, 0.6),
    ]
    
    print(f"Available methods: {reranker.get_available_methods()}")
    
    # Test multiple methods
    comparison = reranker.compare_methods(query, sample_results, top_k=3)
    
    for method, results in comparison.items():
        print(f"\n=== {method.upper()} RESULTS ===")
        for i, (doc, score) in enumerate(results):
            print(f"{i+1}. {doc['title']} (Score: {score:.3f}, Method: {doc.get('rerank_method', 'unknown')})")