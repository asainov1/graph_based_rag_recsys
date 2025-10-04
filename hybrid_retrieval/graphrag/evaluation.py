import matplotlib.pyplot as plt
import numpy as np
import os

class GraphRAGEvaluator:
    def __init__(self):
        """Initialize the evaluator"""
        pass
    
    def evaluate_retrieval(self, retriever, test_queries, ground_truth, k_values=[1, 3, 5, 10]):
        """Evaluate retrieval performance using standard metrics"""
        results = {
            "query": [],
            "precision@k": {k: [] for k in k_values},
            "recall@k": {k: [] for k in k_values},
            "mrr": []
        }
        
        for query_id, query in enumerate(test_queries):
            query_text = query["question"]
            relevant_docs = ground_truth.get(query_id, [])
            
            # Get retrieval results
            retrieved_docs = retriever.retrieve(query_text, top_k=max(k_values))
            retrieved_ids = [doc["id"] for doc in retrieved_docs]
            
            # Calculate metrics
            results["query"].append(query_text)
            
            # MRR calculation
            mrr = 0
            for rank, doc_id in enumerate(retrieved_ids):
                if doc_id in relevant_docs:
                    mrr = 1.0 / (rank + 1)
                    break
            results["mrr"].append(mrr)
            
            # Precision and Recall at different k values
            for k in k_values:
                retrieved_at_k = retrieved_ids[:k]
                relevant_retrieved = [doc_id for doc_id in retrieved_at_k if doc_id in relevant_docs]
                
                # Precision@k
                precision_at_k = len(relevant_retrieved) / k if k > 0 else 0
                results["precision@k"][k].append(precision_at_k)
                
                # Recall@k
                recall_at_k = len(relevant_retrieved) / len(relevant_docs) if len(relevant_docs) > 0 else 0
                results["recall@k"][k].append(recall_at_k)
        
        # Calculate averages
        avg_results = {
            "avg_precision@k": {k: sum(results["precision@k"][k]) / len(test_queries) for k in k_values},
            "avg_recall@k": {k: sum(results["recall@k"][k]) / len(test_queries) for k in k_values},
            "avg_mrr": sum(results["mrr"]) / len(test_queries)
        }
        
        return results, avg_results
    
    def visualize_comparison(self, avg_results_dict, output_dir="evaluation_results"):
        """Create comparative visualizations of retriever performance"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        retrievers = list(avg_results_dict.keys())
        k_values = list(avg_results_dict[retrievers[0]]["avg_precision@k"].keys())
        
        # Precision@k
        plt.figure(figsize=(12, 6))
        for retriever in retrievers:
            precision_values = [avg_results_dict[retriever]["avg_precision@k"][k] for k in k_values]
            plt.plot(k_values, precision_values, marker='o', label=retriever)
        
        plt.title('Precision@k Comparison')
        plt.xlabel('k')
        plt.ylabel('Precision')
        plt.legend()
        plt.grid(True)
        precision_file = os.path.join(output_dir, 'precision_comparison.png')
        plt.savefig(precision_file)
        plt.close()
        
        # Recall@k
        plt.figure(figsize=(12, 6))
        for retriever in retrievers:
            recall_values = [avg_results_dict[retriever]["avg_recall@k"][k] for k in k_values]
            plt.plot(k_values, recall_values, marker='o', label=retriever)
        
        plt.title('Recall@k Comparison')
        plt.xlabel('k')
        plt.ylabel('Recall')
        plt.legend()
        plt.grid(True)
        recall_file = os.path.join(output_dir, 'recall_comparison.png')
        plt.savefig(recall_file)
        plt.close()
        
        # MRR
        plt.figure(figsize=(8, 6))
        mrr_values = [avg_results_dict[retriever]["avg_mrr"] for retriever in retrievers]
        plt.bar(retrievers, mrr_values)
        plt.title('Mean Reciprocal Rank (MRR) Comparison')
        plt.ylabel('MRR')
        plt.xticks(rotation=45)
        mrr_file = os.path.join(output_dir, 'mrr_comparison.png')
        plt.savefig(mrr_file, bbox_inches='tight')
        plt.close()
        
        return [precision_file, recall_file, mrr_file]
    
    def evaluate_hybrid_retrieval(self, graph_retriever, other_retrievers, test_queries, ground_truth):
        """Compare GraphRAG with other retrievers and hybrid approaches"""
        # Evaluate each retriever individually
        retrieval_results = {}
        
        # GraphRAG
        graph_results, graph_avg = self.evaluate_retrieval(graph_retriever, test_queries, ground_truth)
        retrieval_results["GraphRAG"] = graph_avg
        
        # Other retrievers
        for name, retriever in other_retrievers.items():
            results, avg = self.evaluate_retrieval(retriever, test_queries, ground_truth)
            retrieval_results[name] = avg
        
        # Create a simple hybrid retriever (combine results)
        class SimpleHybridRetriever:
            def __init__(self, retrievers, weights=None):
                self.retrievers = retrievers
                self.weights = weights or {name: 1.0 for name in retrievers}
                # Normalize weights
                total = sum(self.weights.values())
                self.weights = {name: w/total for name, w in self.weights.items()}
            
            def retrieve(self, query, top_k=5):
                all_results = {}
                
                # Get results from each retriever
                for name, retriever in self.retrievers.items():
                    results = retriever.retrieve(query, top_k=top_k*2)  # Get more results to allow for filtering
                    for result in results:
                        doc_id = result["id"]
                        score = result.get("score", 0.0) * self.weights[name]
                        
                        if doc_id in all_results:
                            all_results[doc_id]["score"] += score
                        else:
                            result["score"] = score
                            all_results[doc_id] = result
                
                # Convert to list and sort by score
                combined_results = list(all_results.values())
                combined_results.sort(key=lambda x: x.get("score", 0), reverse=True)
                
                return combined_results[:top_k]
        
        # Create hybrid retrievers with different weightings
        hybrid_retrievers = {}
        
        if other_retrievers:
            # Equal weights
            equal_hybrid = SimpleHybridRetriever(
                {"GraphRAG": graph_retriever, **other_retrievers}
            )
            
            # Graph-heavy weights
            graph_weights = {"GraphRAG": 2.0}
            graph_weights.update({name: 1.0 for name in other_retrievers})
            graph_heavy = SimpleHybridRetriever(
                {"GraphRAG": graph_retriever, **other_retrievers}, 
                weights=graph_weights
            )
            
            # Evaluate hybrid retrievers
            hybrid_retrievers["Hybrid (Equal)"] = equal_hybrid
            hybrid_retrievers["Hybrid (Graph-heavy)"] = graph_heavy
            
            for name, retriever in hybrid_retrievers.items():
                results, avg = self.evaluate_retrieval(retriever, test_queries, ground_truth)
                retrieval_results[name] = avg
        
        # Visualize comparison
        vis_files = self.visualize_comparison(retrieval_results)
        
        return retrieval_results, vis_files

if __name__ == "__main__":
    # Simple test with mock data
    import networkx as nx
    from graphrag.graph_rag_retriever import GraphRAGRetriever
    
    # Create a mock graph
    G = nx.Graph()
    G.add_node("article_1", type="article", title="Test Article 1")
    G.add_node("article_2", type="article", title="Test Article 2")
    G.add_node("PERSON_Smith", type="entity", label="PERSON", text="Smith")
    G.add_edge("article_1", "PERSON_Smith", relationship="contains")
    G.add_edge("article_2", "PERSON_Smith", relationship="contains")
    
    # Mock ground truth
    test_queries = [
        {"question": "Who is Smith?"},
        {"question": "What is article 1 about?"}
    ]
    
    ground_truth = {
        0: ["article_1", "article_2"],
        1: ["article_1"]
    }
    
    # Mock retriever that just returns article_1 for any query
    class MockRetriever:
        def retrieve(self, query, top_k=5):
            return [
                {"id": "article_1", "text": "Test Article 1", "score": 0.9},
                {"id": "article_2", "text": "Test Article 2", "score": 0.5}
            ][:top_k]
    
    # Create evaluator and run evaluation
    evaluator = GraphRAGEvaluator()
    mock_retriever = MockRetriever()
    
    results, avg = evaluator.evaluate_retrieval(mock_retriever, test_queries, ground_truth)
    
    print("Evaluation Results:")
    print(f"MRR: {avg['avg_mrr']}")
    print(f"Precision@1: {avg['avg_precision@k'][1]}")
    print(f"Recall@3: {avg['avg_recall@k'][3]}")
    