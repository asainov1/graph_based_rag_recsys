"""
Score Fusion Algorithms for Hybrid Retrieval
Implements different methods to combine scores from multiple retrievers
"""
from typing import List, Dict, Any, Tuple
import numpy as np
from collections import defaultdict


def reciprocal_rank_fusion(results_list: List[List[Tuple[Dict, float]]], 
                          k: int = 60) -> List[Tuple[Dict, float]]:
    """
    Reciprocal Rank Fusion (RRF) - combines rankings from multiple retrievers
    
    Args:
        results_list: List of results from each retriever
        k: Constant for RRF formula (default 60)
    
    Returns:
        Fused and re-ranked results
    """
    # Store RRF scores for each document
    rrf_scores = defaultdict(float)
    doc_data = {}
    
    # Calculate RRF score for each document
    for retriever_results in results_list:
        for rank, (doc, score) in enumerate(retriever_results):
            # Create a unique doc ID based on content
            doc_id = doc.get('id', str(hash(doc.get('content', ''))))
            
            # RRF formula: 1 / (k + rank)
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
            # Store document data (overwrite with latest)
            doc_data[doc_id] = doc
    
    # Sort by RRF score
    sorted_results = sorted(
        [(doc_data[doc_id], score) for doc_id, score in rrf_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results


def weighted_fusion(results_list: List[List[Tuple[Dict, float]]], 
                   weights: List[float]) -> List[Tuple[Dict, float]]:
    """
    Weighted score combination
    
    Args:
        results_list: List of results from each retriever
        weights: Weight for each retriever (should sum to 1.0)
    
    Returns:
        Fused and re-ranked results
    """
    # Normalize weights
    weights = np.array(weights) / np.sum(weights)
    
    # Store weighted scores
    weighted_scores = defaultdict(float)
    doc_data = {}
    
    for retriever_idx, retriever_results in enumerate(results_list):
        for doc, score in retriever_results:
            doc_id = doc.get('id', str(hash(doc.get('content', ''))))
            weighted_scores[doc_id] += score * weights[retriever_idx]
            doc_data[doc_id] = doc
    
    # Sort by weighted score
    sorted_results = sorted(
        [(doc_data[doc_id], score) for doc_id, score in weighted_scores.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    return sorted_results


def normalize_scores(results: List[Tuple[Dict, float]]) -> List[Tuple[Dict, float]]:
    """Normalize scores to [0, 1] range"""
    if not results:
        return results
    
    scores = [score for _, score in results]
    min_score = min(scores)
    max_score = max(scores)
    
    if max_score == min_score:
        return [(doc, 1.0) for doc, _ in results]
    
    normalized = []
    for doc, score in results:
        norm_score = (score - min_score) / (max_score - min_score)
        normalized.append((doc, norm_score))
    
    return normalized