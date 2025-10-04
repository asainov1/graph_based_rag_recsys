import pickle
import time
import os
from data_loader import ETHNewsDataLoader
from entity_extractor import EnhancedEntityExtractor
from graph_builder import GraphBuilder
from graph_rag_retriever import GraphRAGRetriever
from graph_visualizer import GraphVisualizer

def test_graphrag(graph_path=None, entity_chunks_path=None):
    """
    Test the GraphRAG retriever.
    
    Args:
        graph_path: Path to saved graph pickle file
        entity_chunks_path: Path to saved entity chunks pickle file
    """
    start_time = time.time()
    
    # Find the most recent graph and entity chunks files if not specified
    if graph_path is None or entity_chunks_path is None:
        # Find all graph pickle files
        graph_files = [f for f in os.listdir('.') if f.startswith('graph_') and f.endswith('.pickle')]
        chunk_files = [f for f in os.listdir('.') if f.startswith('entity_chunks_') and f.endswith('.pickle')]
        
        # Sort by creation time (most recent first)
        graph_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        chunk_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        
        if graph_files and chunk_files:
            graph_path = graph_files[0]
            entity_chunks_path = chunk_files[0]
            print(f"Using most recent files: {graph_path} and {entity_chunks_path}")
        else:
            # Use default paths if no files found
            graph_path = "graph.pickle"
            entity_chunks_path = "entity_chunks.pickle"
    
    # Check if graph and entity chunks exist, otherwise create them
    if not os.path.exists(graph_path) or not os.path.exists(entity_chunks_path):
        print("Graph or entity chunks not found. Creating new ones...")
        from test_graph_builder import test_graph_builder
        _, _, graph_path, entity_chunks_path = test_graph_builder(article_limit=100)
    else:
        print(f"Loading existing graph from {graph_path} and entity chunks from {entity_chunks_path}")
    
    # Initialize GraphRAG retriever
    retriever = GraphRAGRetriever(graph_path=graph_path, entity_chunks_path=entity_chunks_path)
    
    # Test with sample queries
    test_queries = [
        "What research is happening at ETH Zurich?",
        "Who is Professor Smith?",
        "Tell me about climate research",
        "What are the latest developments in artificial intelligence at ETH?",
        "Information about sustainable energy at ETH",
        "How does ETH support faculty from abroad?",
        "What programs are available for doctoral students at ETH?",
        "Recent publications from ETH researchers"
    ]
    
    # Create directory for query results
    results_dir = "query_results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Create visualizer
    visualizer = GraphVisualizer()
    
    # Process each query
    for query in test_queries:
        query_start = time.time()
        print(f"\nQuery: {query}")
        
        # Run retrieval
        results = retriever.retrieve(query, top_k=5)
        
        # Display results
        print(f"Retrieved {len(results)} results in {time.time() - query_start:.2f} seconds:")
        
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print(f"ID: {result['id']}")
            print(f"Title: {result['title']}")
            print(f"Score: {result.get('score', 0):.4f}")
            print(f"Retrieval Method: {result.get('retrieval_method', 'unknown')}")
            if 'expansion_level' in result:
                print(f"Expansion Level: {result['expansion_level']}")
            if 'entity' in result:
                print(f"Entity: {result['entity']}")
            
            # Print text snippet
            text = result.get("text", "")
            text_snippet = text[:150] + "..." if len(text) > 150 else text
            print(f"Text: {text_snippet}")
        
        # Create safe filename from query
        safe_filename = query.replace(" ", "_").replace("?", "").replace("/", "_")[:30]
        vis_filename = os.path.join(results_dir, f"query_{safe_filename}.html")
        
        # Visualize query results
        try:
            visualization_file = visualizer.visualize_query_path(
                retriever.graph, 
                results, 
                output_file=vis_filename
            )
            print(f"Query path visualization saved to {visualization_file}")
        except Exception as e:
            print(f"Error creating visualization: {e}")
    
    print(f"\nTotal testing time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    test_graphrag()