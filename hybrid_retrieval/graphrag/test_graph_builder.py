from data_loader import ETHNewsDataLoader
from entity_extractor import EnhancedEntityExtractor
from graph_builder import GraphBuilder
import networkx as nx
import pickle
import time

def test_graph_builder(article_limit=500):
    """
    Build a knowledge graph from the ETH News dataset.
    
    Args:
        article_limit: Maximum number of articles to process (set to None for all)
    """
    start_time = time.time()
    
    # Load articles
    print(f"Loading up to {article_limit} articles...")
    loader = ETHNewsDataLoader()
    articles = loader.load_articles(limit=article_limit)
    
    print(f"Loaded {len(articles)} articles in {time.time() - start_time:.2f} seconds")
    
    # Process entities
    entity_start = time.time()
    print("\nProcessing entities...")
    extractor = EnhancedEntityExtractor()
    entities = extractor.process_entities(articles)
    normalized_entities = extractor.normalize_entities(entities)
    
    print(f"Processed entities in {time.time() - entity_start:.2f} seconds")
    
    # Build knowledge graph
    graph_start = time.time()
    print("\nBuilding knowledge graph...")
    builder = GraphBuilder()
    G, entity_chunks = builder.create_knowledge_graph(articles, normalized_entities)
    
    print(f"Built knowledge graph in {time.time() - graph_start:.2f} seconds")
    
    # Analyze the graph
    print("\nGraph Analysis:")
    print(f"Total nodes: {len(G.nodes())}")
    print(f"Total edges: {len(G.edges())}")
    
    # Count node types
    article_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'article']
    entity_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'entity']
    print(f"Article nodes: {len(article_nodes)}")
    print(f"Entity nodes: {len(entity_nodes)}")
    
    # Count edge types
    contains_edges = [e for e in G.edges(data=True) if e[2].get('relationship') == 'contains']
    cooccurs_edges = [e for e in G.edges(data=True) if e[2].get('relationship') == 'co-occurs']
    temporal_edges = [e for e in G.edges(data=True) if e[2].get('relationship') == 'temporal']
    topic_edges = [e for e in G.edges(data=True) if e[2].get('relationship') == 'shared_topic']
    
    print(f"Contains edges: {len(contains_edges)}")
    print(f"Co-occurrence edges: {len(cooccurs_edges)}")
    print(f"Temporal edges: {len(temporal_edges)}")
    print(f"Topic edges: {len(topic_edges)}")
    
    # Analyze entity chunks
    print(f"\nEntity chunks collected: {len(entity_chunks)}")
    
    # Show sample entity with its contexts
    if entity_chunks:
        sample_entity = list(entity_chunks.keys())[0]
        print(f"\nSample entity: {sample_entity}")
        print(f"Appears in {len(entity_chunks[sample_entity])} articles")
        print("Sample context:")
        print(entity_chunks[sample_entity][0]["text"][:200] + "...")
    
    # Save graph and entity chunks
    save_start = time.time()
    print("\nSaving knowledge graph and entity chunks...")
    
    # Create different filenames based on size
    graph_filename = f"graph_{len(articles)}_articles.pickle"
    chunks_filename = f"entity_chunks_{len(articles)}_articles.pickle"
    
    # Save graph
    with open(graph_filename, "wb") as f:
        pickle.dump(G, f)
    
    # Save entity chunks
    with open(chunks_filename, "wb") as f:
        pickle.dump(entity_chunks, f)
    
    print(f"Saved to {graph_filename} and {chunks_filename} in {time.time() - save_start:.2f} seconds")
    print(f"\nTotal processing time: {time.time() - start_time:.2f} seconds")
    
    return G, entity_chunks, graph_filename, chunks_filename

if __name__ == "__main__":
    # Process more articles (default: 500)
    test_graph_builder(article_limit=500)