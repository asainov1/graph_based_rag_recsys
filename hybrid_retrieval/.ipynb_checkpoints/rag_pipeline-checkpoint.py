"""Test script for hybrid retrieval system"""
import sys
import os
from typing import List, Tuple, Dict
from transformers import pipeline

# Add parent directory to path and change working directory to project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

from hybrid_retrieval.adapters.bm25_adapter import BM25Adapter
from hybrid_retrieval.adapters.dense_adapter import DenseAdapter
from hybrid_retrieval.adapters.graphrag_adapter import GraphRAGAdapter
from hybrid_retrieval.hybrid_retriever import HybridRetriever

def test_reranking_with_sample_data():
    """Test reranking with sample data when real data is missing"""
    print(f"\n{'='*80}")
    print("TESTING RERANKING WITH SAMPLE DATA")
    print(f"{'='*80}")
    
    try:
        from hybrid_retrieval.reranking import SimpleReranker
        
        # Create sample retrieval results in your format
        sample_results = [
            ({"content": "ETH Zurich conducts advanced artificial intelligence research in machine learning and neural networks.", 
              "title": "AI Research at ETH", "retriever": "BM25", "metadata": {}}, 0.5),
            ({"content": "The computer science department at ETH focuses on robotics, AI systems, and autonomous vehicles.", 
              "title": "CS Department Focus", "retriever": "Dense", "metadata": {}}, 0.4),
            ({"content": "ETH researchers published breakthrough papers in quantum computing and cryptography.", 
              "title": "Quantum Research", "retriever": "GraphRAG", "metadata": {}}, 0.3),
            ({"content": "Zurich weather forecast shows sunny conditions for the weekend.", 
              "title": "Weather Update", "retriever": "BM25", "metadata": {}}, 0.2),
            ({"content": "ETH hosts international conference on sustainable technology and green energy solutions.", 
              "title": "Sustainability Conference", "retriever": "Dense", "metadata": {}}, 0.6),
        ]
        
        query = "How do alpine plants respond to climate change?"
        print(f"Query: '{query}'")
        print(f"Testing with {len(sample_results)} sample documents")
        
        # Test reranking
        reranker = SimpleReranker()
        reranked_results = reranker.rerank(query, sample_results, top_k=3)
        
        print(f"\n=== COMPARISON ===")
        print("BEFORE RERANKING (by fusion score):")
        sorted_original = sorted(sample_results, key=lambda x: x[1], reverse=True)
        for i, (doc, score) in enumerate(sorted_original):
            print(f"  {i+1}. Score: {score:.3f} - {doc['title']}")
        
        print(f"\nAFTER RERANKING (by relevance):")
        for i, (doc, score) in enumerate(reranked_results):
            print(f"  {i+1}. Rerank: {score:.3f} - {doc['title']}")
            print(f"     Original score: {[x[1] for x in sample_results if x[0]['title'] == doc['title']][0]:.3f}")
        
        print("\nReranking test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Reranking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_retrieval(adapters_dict):
    """Test basic hybrid retrieval without reranking"""
    print(f"\n{'='*80}")
    print("TESTING BASIC HYBRID RETRIEVAL (NO RERANKING)")
    print(f"{'='*80}")
    
    # Create hybrid retriever without reranking
    hybrid = HybridRetriever(
        bm25_adapter=adapters_dict.get('bm25'),
        dense_adapter=adapters_dict.get('dense'),
        graphrag_adapter=adapters_dict.get('graphrag'),
        fusion_method="rrf",
        use_reranking=False  # No reranking
    )
    
    # Test query
    query = "How do alpine plants respond to climate change?"
    print(f"\nTest Query: '{query}'")
    
    try:
        results = hybrid.retrieve(query, top_k=5)
        hybrid.print_results(results)
        return results
    except Exception as e:
        print(f"Error during basic retrieval: {e}")
        return []


def test_reranking_retrieval(adapters_dict):
    """Test hybrid retrieval with reranking"""
    print(f"\n{'='*80}")
    print("TESTING HYBRID RETRIEVAL WITH RERANKING")
    print(f"{'='*80}")
    
    # Create hybrid retriever with reranking
    hybrid_rerank = HybridRetriever(
        bm25_adapter=adapters_dict.get('bm25'),
        dense_adapter=adapters_dict.get('dense'),
        graphrag_adapter=adapters_dict.get('graphrag'),
        fusion_method="rrf",
        use_reranking=True  # Enable reranking
    )
    
    # Test query
    query = "query"
    print(f"\nTest Query: '{query}'")
    
    try:
        results = hybrid_rerank.retrieve(query, top_k=5, rerank_candidates=15)
        hybrid_rerank.print_results(results)
        return results
    except Exception as e:
        print(f"Error during reranking retrieval: {e}")
        return []

def generate_answer_from_documents(
    query: str, 
    documents: List[Tuple[Dict, float]],  # Fixed: Remove extra bracket
    top_k: int = 3) -> str:
    """Generate an answer from top retrieved documents using a language model."""
    print("\n🔍 Generating answer from top documents...")
    
    # Use top-k documents (sorted by score)
    top_docs = [doc[0]['content'] for doc in documents[:top_k] if 'content' in doc[0]]
    if not top_docs:
        return "No relevant documents found to generate an answer."
    
    # Combine context
    context = "\n".join(top_docs)

    # Load HuggingFace generation pipeline
    generator = pipeline("text-generation", model="gpt2", max_new_tokens=150)

    # Construct prompt
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    
    # Generate answer
    result = generator(prompt, do_sample=True, temperature=0.7)[0]["generated_text"]
    
    # Extract the answer after the 'Answer:' part
    answer = result.split("Answer:")[-1].strip()
    return answer
    
def main():
    print("Initializing Hybrid Retrieval System...")
    
    # Initialize adapters with error handling
    adapters_dict = {}
    
    try:
        print("\n1. Initializing BM25...")
        bm25 = BM25Adapter()
        adapters_dict['bm25'] = bm25
        print("BM25 initialized successfully")
    except Exception as e:
        print(f"BM25 initialization failed: {e}")
    
    try:
        print("\n2. Initializing Dense/ChromaDB...")
        dense = DenseAdapter()
        adapters_dict['dense'] = dense
        print("Dense retriever initialized successfully")
    except Exception as e:
        print(f"Dense retriever initialization failed: {e}")
    
    try:
        print("\n3. Initializing GraphRAG...")
        graphrag = GraphRAGAdapter()
        adapters_dict['graphrag'] = graphrag
        print("GraphRAG initialized successfully")
    except Exception as e:
        print(f"GraphRAG initialization failed: {e}")
    
    # Check if any real data is available
    total_docs = 0
    for name, adapter in adapters_dict.items():
        try:
            # Try to get document count if method exists
            if hasattr(adapter, 'get_document_count'):
                count = adapter.get_document_count()
                total_docs += count
                print(f"{name}: {count} documents")
        except:
            pass
    
    print(f"\n4. Available retrievers: {list(adapters_dict.keys())}")
    print(f"Total documents available: {total_docs}")
    
    # If no real data available, test reranking with samples
    if not adapters_dict:
        print("\nNo real data found. Testing reranking with sample data...")
        test_reranking_with_sample_data()
        return
    
    # Test basic retrieval first
    basic_results = test_basic_retrieval(adapters_dict)

     # Test with reranking
    rerank_results = test_reranking_retrieval(adapters_dict)
    
    # Test with reranking if basic retrieval found results
    if basic_results and rerank_results:
        print(f"\n{'='*80}")
        print("COMPARISON SUMMARY")
        print(f"{'='*80}")
        print(f"Basic retrieval: {len(basic_results)} results")
        print(f"With reranking: {len(rerank_results)} results")
        
        if rerank_results:
            top_rerank_score = rerank_results[0][1]
            print(f"Top reranked result score: {top_rerank_score:.4f}")
            
            top_doc = rerank_results[0][0]
            if 'rerank_score' in top_doc:
                print(f"Rerank score: {top_doc['rerank_score']:.4f}")
                
            # Show improvement
            if basic_results:
                print(f"\nTop document improvement:")
                print(f"  Before reranking: {basic_results[0][0].get('title', 'No title')[:80]}...")
                print(f"  After reranking:  {rerank_results[0][0].get('title', 'No title')[:80]}...")
                
                basic_score = basic_results[0][1]
                rerank_score = rerank_results[0][1]
                improvement = rerank_score / basic_score if basic_score > 0 else float('inf')
                print(f"  Score improvement: {basic_score:.4f} → {rerank_score:.4f} ({improvement:.1f}x better)")

        # Generate answer from reranked results
        if rerank_results:
            top_answer = generate_answer_from_documents(
                query="How do alpine plants respond to climate change?",
                documents=rerank_results,
                top_k=3
            )
            print("\n📣 Generated Answer:")
            print("-" * 40)
            print(top_answer)
        
if __name__ == "__main__":
    main()