"""
Evaluation script for testing reranking performance on 25 benchmark questions
"""
import json
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from hybrid_retrieval.adapters.bm25_adapter import BM25Adapter
from hybrid_retrieval.adapters.dense_adapter import DenseAdapter
from hybrid_retrieval.adapters.graphrag_adapter import GraphRAGAdapter
from hybrid_retrieval.hybrid_retriever import HybridRetriever


def load_benchmark_questions(file_path: str = "hybrid_retrieval/benchmark_qa.json"):
    """Load the 25 benchmark questions"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('questions', [])
    except FileNotFoundError:
        print(f"Benchmark file not found: {file_path}")
        return []


def evaluate_single_question(question_data: Dict, baseline_retriever: HybridRetriever, 
                            reranking_retriever: HybridRetriever, methods: List[str]):
    """Evaluate reranking on a single question"""
    
    question = question_data['question']
    question_id = question_data['id']
    
    print(f"\nQ{question_id}: {question}")
    
    # Get baseline results
    print("Getting baseline results...")
    baseline_results = baseline_retriever.retrieve(question, top_k=10)
    
    # Test each reranking method
    reranking_results = {}
    
    for method in methods:
        print(f"Testing {method}...")
        try:
            start_time = time.time()
            results = reranking_retriever.retrieve(
                question, top_k=10, rerank_method=method
            )
            timing = time.time() - start_time
            
            reranking_results[method] = {
                'results': results,
                'timing': timing
            }
        except Exception as e:
            print(f"Error with {method}: {e}")
            reranking_results[method] = {'results': [], 'timing': 0}
    
    return {
        'question_id': question_id,
        'question': question,
        'baseline': baseline_results,
        'reranked': reranking_results
    }


def calculate_metrics(baseline_results: List, reranked_results: List, 
                     relevant_docs: List[Dict]) -> Dict[str, float]:
    """Calculate evaluation metrics"""
    
    # Create relevance mapping from benchmark
    relevance_map = {}
    for doc in relevant_docs:
        doc_id = doc.get('doc_id', '')
        relevance_map[doc_id] = doc.get('relevance_score', 0.0)
    
    def get_relevance_scores(results: List, k: int = 5) -> List[float]:
        scores = []
        for i, (doc, _) in enumerate(results[:k]):
            # Try different ID fields
            doc_id = doc.get('id', doc.get('doc_id', ''))
            title = doc.get('title', '')
            
            # Check if document is relevant
            relevance = 0.0
            if doc_id in relevance_map:
                relevance = relevance_map[doc_id]
            elif any(title.startswith(rel_doc.get('title', '')[:50]) 
                    for rel_doc in relevant_docs):
                relevance = 0.5  # Partial match
            
            scores.append(relevance)
        return scores
    
    # Calculate metrics for both baseline and reranked
    baseline_scores = get_relevance_scores(baseline_results)
    reranked_scores = get_relevance_scores(reranked_results)
    
    def precision_at_k(scores: List[float]) -> float:
        return sum(1 for s in scores if s > 0) / len(scores) if scores else 0
    
    def dcg_at_k(scores: List[float]) -> float:
        return sum(s / np.log2(i + 2) for i, s in enumerate(scores))
    
    def ndcg_at_k(scores: List[float]) -> float:
        dcg = dcg_at_k(scores)
        ideal_scores = sorted(scores, reverse=True)
        idcg = dcg_at_k(ideal_scores)
        return dcg / idcg if idcg > 0 else 0
    
    return {
        'baseline_precision_5': precision_at_k(baseline_scores),
        'reranked_precision_5': precision_at_k(reranked_scores),
        'baseline_ndcg_5': ndcg_at_k(baseline_scores),
        'reranked_ndcg_5': ndcg_at_k(reranked_scores),
        'precision_improvement': precision_at_k(reranked_scores) - precision_at_k(baseline_scores),
        'ndcg_improvement': ndcg_at_k(reranked_scores) - ndcg_at_k(baseline_scores)
    }


def main():
    """Main evaluation function"""
    
    print("="*80)
    print("RERANKING EVALUATION ON 25 BENCHMARK QUESTIONS")
    print("="*80)
    
    # Load benchmark questions
    questions = load_benchmark_questions()
    if not questions:
        print("No benchmark questions found!")
        return
    
    print(f"Loaded {len(questions)} benchmark questions")
    
    # Initialize retrievers
    print("\nInitializing retrievers...")
    try:
        bm25 = BM25Adapter()
        dense = DenseAdapter()
        graphrag = GraphRAGAdapter()
    except Exception as e:
        print(f"Error initializing retrievers: {e}")
        return
    
    # Create baseline and reranking retrievers
    baseline_retriever = HybridRetriever(
        bm25_adapter=bm25,
        dense_adapter=dense,
        graphrag_adapter=graphrag,
        use_reranking=False
    )
    
    reranking_retriever = HybridRetriever(
        bm25_adapter=bm25,
        dense_adapter=dense,
        graphrag_adapter=graphrag,
        use_reranking=True
    )
    
    # Get available reranking methods
    available_methods = reranking_retriever.reranker.get_available_methods()
    test_methods = available_methods[:3]  # Test first 3 methods
    
    print(f"Testing methods: {test_methods}")
    
    # Evaluate subset of questions (first 5 for demo)
    test_questions = questions[:5]
    results = []
    
    for question_data in test_questions:
        result = evaluate_single_question(
            question_data, baseline_retriever, reranking_retriever, test_methods
        )
        results.append(result)
        
        # Calculate metrics if relevant documents available
        if 'relevant_documents' in question_data:
            for method in test_methods:
                if method in result['reranked']:
                    metrics = calculate_metrics(
                        result['baseline'],
                        result['reranked'][method]['results'],
                        question_data['relevant_documents']
                    )
                    result['reranked'][method]['metrics'] = metrics
    
    # Print summary
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    for result in results:
        print(f"\nQ{result['question_id']}: {result['question'][:60]}...")
        
        for method in test_methods:
            if method in result['reranked'] and 'metrics' in result['reranked'][method]:
                metrics = result['reranked'][method]['metrics']
                timing = result['reranked'][method]['timing']
                
                print(f"  {method:15s}: "
                      f"P@5={metrics['precision_improvement']:+.3f}, "
                      f"NDCG@5={metrics['ndcg_improvement']:+.3f}, "
                      f"Time={timing:.2f}s")
    
    # Save results
    output_file = "hybrid_retrieval/evaluation_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("\nNext steps:")
    print("1. Expand to all 25 questions")
    print("2. Add more detailed metrics")
    print("3. Create visualization plots")
    print("4. Write qualitative analysis")


if __name__ == "__main__":
    main()