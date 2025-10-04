import networkx as nx
from typing import List, Dict, Any, Tuple
import re
from collections import defaultdict

class GraphBuilder:
    """Builds a knowledge graph from articles and their entities."""
    
    def __init__(self):
        """Initialize graph builder."""
        pass
    
    def create_knowledge_graph(self, articles: List[Dict], processed_entities: Dict[str, Dict]) -> Tuple[nx.Graph, Dict]:
        """
        Create a knowledge graph from articles and their entities.
        
        Args:
            articles: List of processed article dictionaries
            processed_entities: Dictionary mapping article IDs to categorized entities
            
        Returns:
            Tuple of (knowledge graph, entity_chunks dictionary)
        """
        print("Building knowledge graph...")
        
        # Initialize graph
        G = nx.Graph()
        
        # Dictionary to store entity chunks for later retrieval
        entity_chunks = {}
        
        # Statistics for monitoring
        stats = {
            "article_nodes": 0,
            "entity_nodes": 0,
            "article_entity_edges": 0,
            "entity_entity_edges": 0,
            "temporal_edges": 0,
            "topic_edges": 0
        }
        
        # First, add article nodes
        for article in articles:
            article_id = article["id"]
            article_node_id = f"article_{article_id}"
            
            # Add article node with metadata
            G.add_node(article_node_id, 
                      type="article",
                      title=article.get("title", ""),
                      language=article.get("language", ""),
                      date=article.get("date", ""),
                      source=article.get("source", ""))
            
            stats["article_nodes"] += 1
        
        # Add entity nodes and connect to articles
        entity_to_articles = defaultdict(list)
        all_entities = set()
        
        for article_id, categories in processed_entities.items():
            article_node_id = f"article_{article_id}"
            
            # Process each category of entities
            for category, entities in categories.items():
                for entity_text in entities:
                    # Create a unique entity ID
                    entity_id = f"{category}_{self._clean_entity_id(entity_text)}"
                    
                    # Add entity node if not exists
                    if not G.has_node(entity_id):
                        G.add_node(entity_id,
                                  type="entity",
                                  category=category,
                                  text=entity_text)
                        
                        stats["entity_nodes"] += 1
                        all_entities.add(entity_id)
                    
                    # Connect entity to article
                    G.add_edge(article_node_id, entity_id,
                              relationship="contains",
                              weight=1)
                    
                    stats["article_entity_edges"] += 1
                    
                    # Store which articles contain this entity
                    entity_to_articles[entity_id].append(article_node_id)
                    
                    # Store entity context for retrieval
                    article = next((a for a in articles if a["id"] == article_id), None)
                    if article:
                        context = self._extract_entity_context(entity_text, article["main_content"])
                        
                        if entity_id not in entity_chunks:
                            entity_chunks[entity_id] = []
                        
                        entity_chunks[entity_id].append({
                            "article_id": article_id,
                            "text": context,
                            "language": article["language"]
                        })
        
        # Add entity-entity connections based on co-occurrence
        for entity_id, article_list in entity_to_articles.items():
            for other_entity_id in all_entities:
                if entity_id != other_entity_id:
                    # Check if they co-occur in any articles
                    common_articles = set(article_list).intersection(set(entity_to_articles[other_entity_id]))
                    
                    if common_articles:
                        # Entities co-occur in some articles
                        if not G.has_edge(entity_id, other_entity_id):
                            G.add_edge(entity_id, other_entity_id,
                                      relationship="co-occurs",
                                      weight=len(common_articles))
                            
                            stats["entity_entity_edges"] += 1
        
        # Add temporal edges between articles
        self._add_temporal_edges(G, articles, stats)
        
        # Add topic-based edges
        self._add_topic_edges(G, articles, stats)
        
        print(f"Knowledge graph built successfully:")
        print(f"- {stats['article_nodes']} article nodes")
        print(f"- {stats['entity_nodes']} entity nodes")
        print(f"- {stats['article_entity_edges']} article-entity edges")
        print(f"- {stats['entity_entity_edges']} entity-entity edges")
        print(f"- {stats['temporal_edges']} temporal edges")
        print(f"- {stats['topic_edges']} topic edges")
        
        return G, entity_chunks
    
    def _clean_entity_id(self, entity_text: str) -> str:
        """Create a clean entity ID from text."""
        # Replace spaces and special characters
        clean_id = re.sub(r'[^a-zA-Z0-9]', '_', entity_text)
        # Convert to lowercase and limit length
        clean_id = clean_id.lower()[:50]
        return clean_id
    
    def _extract_entity_context(self, entity: str, content: str, context_size: int = 200) -> str:
        """Extract context around entity mention."""
        if not content or not entity:
            return ""
        
        # Find entity in content
        position = content.find(entity)
        if position == -1:
            return ""
        
        # Get context window
        start = max(0, position - context_size // 2)
        end = min(len(content), position + len(entity) + context_size // 2)
        
        # Extract context
        context = content[start:end]
        
        # Add ellipsis if needed
        if start > 0:
            context = "..." + context
        if end < len(content):
            context = context + "..."
        
        return context
    
    def _add_temporal_edges(self, G: nx.Graph, articles: List[Dict], stats: Dict) -> None:
        """Add temporal edges between articles based on dates."""
        # Group articles by language and sort by date
        en_articles = sorted([a for a in articles if a["language"] == "en"], 
                           key=lambda x: x.get("date", ""))
        de_articles = sorted([a for a in articles if a["language"] == "de"], 
                           key=lambda x: x.get("date", ""))
        
        # Connect articles in chronological order within each language
        for article_list in [en_articles, de_articles]:
            for i in range(len(article_list) - 1):
                curr_article = article_list[i]
                next_article = article_list[i + 1]
                
                curr_node_id = f"article_{curr_article['id']}"
                next_node_id = f"article_{next_article['id']}"
                
                # Only connect if both have valid dates
                if curr_article.get("date") and next_article.get("date"):
                    G.add_edge(curr_node_id, next_node_id,
                              relationship="temporal",
                              weight=1)
                    
                    stats["temporal_edges"] += 1
    
    def _add_topic_edges(self, G: nx.Graph, articles: List[Dict], stats: Dict) -> None:
        """Add edges between articles with shared topics."""
        # Create mapping of topics to articles
        topic_to_articles = defaultdict(list)
        
        for article in articles:
            article_id = article["id"]
            article_node_id = f"article_{article_id}"
            
            # Get topics
            topics = article.get("topics", [])
            
            for topic in topics:
                topic_str = str(topic).lower()
                topic_to_articles[topic_str].append(article_node_id)
        
        # Connect articles with shared topics
        for topic, article_list in topic_to_articles.items():
            if len(article_list) > 1:  # Only if topic is shared by multiple articles
                for i in range(len(article_list)):
                    for j in range(i + 1, len(article_list)):
                        article1 = article_list[i]
                        article2 = article_list[j]
                        
                        # Add edge if not already connected
                        if not G.has_edge(article1, article2):
                            G.add_edge(article1, article2,
                                      relationship="shared_topic",
                                      topic=topic,
                                      weight=1)
                            
                            stats["topic_edges"] += 1