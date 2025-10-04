import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from .retriever import MultilingualBM25Retriever


def save_results(query: str, results: List[Dict], output_dir: str = "bm25_query_results") -> Path:
    """
    Save search results to a JSON file.
    
    Args:
        query: The search query
        results: List of search results
        output_dir: Directory to save results (default: "bm25_query_results")
        
    Returns:
        Path to the saved results file
    """
    results_dir = Path(output_dir)
    os.makedirs(results_dir, exist_ok=True)
    
    # Create filename from query
    slug = re.sub(r'[^\w\-]', '', re.sub(r'\s+', '-', query.lower()))[:50]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{slug}_{timestamp}.json"
    
    output_data = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "results_count": len(results),
        "results": results
    }
    
    output_path = results_dir / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_path}")
    return output_path


def run_interactive_search(retriever: MultilingualBM25Retriever) -> None:
    """
    Run an interactive search session with the retriever.
    
    Args:
        retriever: Initialized MultilingualBM25Retriever instance
    """
    print("\n=== BM25 Search Tool Ready ===")
    print("Enter your search queries below. Type 'exit' to quit.")
    
    while True:
        query = input("\nEnter search query: ")
        
        if query.lower() in ['exit', 'quit']:
            print("Exiting search tool")
            break
        
        if not query.strip():
            print("Please enter a valid query")
            continue
        
        # Get number of results
        try:
            k_str = input("Number of results to return (default: 5): ")
            top_k = int(k_str) if k_str.strip() else 5
        except ValueError:
            print("Invalid number, using default: 5")
            top_k = 5
        
        # Perform search
        results = retriever.search(query, top_k=top_k)
        
        # Display results
        print(f"\nFound {len(results)} results for '{query}':")
        for i, result in enumerate(results):
            print(f"\n{i+1}. [{result['language'].upper()}] {result['title']} (Score: {result['score']:.4f})")
            print(f"Summary: {result.get('summary', 'N/A')}")
            print(f"Snippet: {result.get('content_snippet', 'N/A')[:200]}...")
        
        # Option to save results
        save_option = input("\nSave these results? (y/n): ")
        if save_option.lower() in ['y', 'yes']:
            save_results(query, results)