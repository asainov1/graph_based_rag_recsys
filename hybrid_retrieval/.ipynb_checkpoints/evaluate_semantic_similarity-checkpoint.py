import json
import sys
import os
import time
import numpy as np
from typing import Dict, List, Tuple, Any
from sentence_transformers import SentenceTransformer, util

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

from hybrid_retrieval.adapters.bm25_adapter import BM25Adapter
from hybrid_retrieval.adapters.dense_adapter import DenseAdapter
from hybrid_retrieval.adapters.graphrag_adapter import GraphRAGAdapter
from hybrid_retrieval.hybrid_retriever import HybridRetriever

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def load_benchmark_questions(file_path: str = "hybrid_retrieval/benchmark_qa.json"):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data.get('questions', [])
    except FileNotFoundError:
        print(f"Benchmark file not found: {file_path}")
        return []

def semantic_similarity(a: str, b: str) -> float:
    emb1 = model.encode(a, convert_to_tensor=True)
    emb2 = model.encode(b, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2)[0])

def evaluate_with_ground_truth(results: List, true_answer: str, top_k: int = 5) -> Dict[str, float]:
    top_results = results[:top_k]
    scores = [semantic_similarity(doc['content'], true_answer) for doc, _ in top_results]
    avg_sim = sum(scores) / len(scores) if scores else 0
    precision = sum(s >= 0.6 for s in scores) / len(scores) if scores else 0
    return {"avg_similarity": avg_sim, "precision@5": precision, "scores": scores}

def evaluate_single_question(question_data: Dict, baseline: HybridRetriever, reranker: HybridRetriever, methods: List[str]):
    q = question_data['question']
    qid = question_data['id']
    true_answer = question_data.get("answer", "")
    print(f"\nQ{qid}: {q}")

    baseline_results = baseline.retrieve(q, top_k=10)
    reranking_results = {}

    for method in methods:
        print(f"Testing {method}...")
        try:
            start = time.time()
            reranked = reranker.retrieve(q, top_k=10, rerank_method=method)
            elapsed = time.time() - start
            reranking_results[method] = {"results": reranked, "timing": elapsed}
        except Exception as e:
            print(f"Error with {method}: {e}")
            reranking_results[method] = {"results": [], "timing": 0}

    return {
        "question_id": qid,
        "question": q,
        "answer": true_answer,
        "baseline": baseline_results,
        "reranked": reranking_results
    }

def main():
    print("="*80)
    print("SEMANTIC SIMILARITY EVALUATION BASED ON GROUND TRUTH ANSWERS")
    print("="*80)

    questions = load_benchmark_questions()
    if not questions:
        print("No benchmark questions found!")
        return

    print(f"Loaded {len(questions)} benchmark questions")

    try:
        bm25 = BM25Adapter()
        dense = DenseAdapter()
        graphrag = GraphRAGAdapter()
    except Exception as e:
        print(f"Retriever initialization error: {e}")
        return

    baseline = HybridRetriever(bm25, dense, graphrag, use_reranking=False)
    reranker = HybridRetriever(bm25, dense, graphrag, use_reranking=True)

    methods = reranker.reranker.get_available_methods()[:3]

    results = []
    for q in questions:
        result = evaluate_single_question(q, baseline, reranker, methods)
        answer = q.get("answer", "")

        result["baseline_eval"] = evaluate_with_ground_truth(result["baseline"], answer)

        for m in methods:
            if m in result["reranked"]:
                reranked_docs = result["reranked"][m]["results"]
                result["reranked"][m]["evaluation"] = evaluate_with_ground_truth(reranked_docs, answer)

        results.append(result)

    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    for r in results:
        print(f"\nQ{r['question_id']}: {r['question'][:60]}...")
        base = r["baseline_eval"]
        print(f"  Baseline     : AvgSim={base['avg_similarity']:.3f}, P@5={base['precision@5']:.2f}")
        for m in methods:
            if "evaluation" in r["reranked"][m]:
                ev = r["reranked"][m]["evaluation"]
                print(f"  {m:12s}: AvgSim={ev['avg_similarity']:.3f}, P@5={ev['precision@5']:.2f}, Time={r['reranked'][m]['timing']:.2f}s")

    with open("hybrid_retrieval/semantic_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()