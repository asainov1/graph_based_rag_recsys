import spacy
from typing import List, Dict, Any, Set, Tuple
import re

class EnhancedEntityExtractor:
    """Entity extractor that leverages pre-extracted entities from ETH News articles."""
    
    def __init__(self):
        """Initialize with spaCy models for fallback extraction."""
        # Load spaCy models for fallback extraction
        try:
            self.nlp_en = spacy.load("en_core_web_lg")
            print("Loaded English language model (large)")
        except:
            try:
                self.nlp_en = spacy.load("en_core_web_md")
                print("Loaded English language model (medium)")
            except:
                self.nlp_en = spacy.load("en_core_web_sm")
                print("Loaded English language model (small)")
        
        try:
            self.nlp_de = spacy.load("de_core_news_lg")
            print("Loaded German language model (large)")
        except:
            try:
                self.nlp_de = spacy.load("de_core_news_md")
                print("Loaded German language model (medium)")
            except:
                self.nlp_de = spacy.load("de_core_news_sm")
                print("Loaded German language model (small)")
    
    def process_entities(self, articles: List[Dict]) -> Dict[str, Dict]:
        """
        Process entities from articles and categorize them.
        
        This uses pre-extracted entities from the JSON files and
        enhances them with categorization and normalization.
        
        Args:
            articles: List of processed article dictionaries
            
        Returns:
            Dictionary mapping article IDs to structured entities
        """
        print("Processing entities from articles...")
        
        all_entities = {}
        entity_stats = {
            "total_articles": len(articles),
            "with_entities": 0,
            "entity_counts": {
                "PERSON": 0,
                "ORG": 0,
                "GPE": 0,
                "LOC": 0,
                "MISC": 0
            }
        }
        
        # Process entities from each article
        for article in articles:
            article_id = article["id"]
            
            # Get pre-extracted entities
            named_entities = article.get("named_entities", [])
            
            if named_entities:
                entity_stats["with_entities"] += 1
                
                # Process and categorize entities
                structured_entities = self._categorize_entities(named_entities, article)
                all_entities[article_id] = structured_entities
                
                # Update statistics
                for category, entities in structured_entities.items():
                    entity_stats["entity_counts"][category] += len(entities)
            else:
                # Create empty structure for articles without entities
                all_entities[article_id] = {
                    "PERSON": [],
                    "ORG": [],
                    "GPE": [],
                    "LOC": [],
                    "MISC": []
                }
        
        # Print statistics
        print(f"Entity processing complete:")
        print(f"- {entity_stats['with_entities']} of {entity_stats['total_articles']} articles had pre-extracted entities")
        print(f"- Categorized entities: {entity_stats['entity_counts']}")
        
        return all_entities
    
    def _categorize_entities(self, entities: List[str], article: Dict) -> Dict[str, List[str]]:
        """
        Categorize entities into standard NER categories.
        
        Args:
            entities: List of entity strings
            article: Full article dictionary for context
            
        Returns:
            Dictionary mapping entity categories to lists of entities
        """
        categorized = {
            "PERSON": [],
            "ORG": [],
            "GPE": [],  # Countries, cities, states
            "LOC": [],  # Non-GPE locations
            "MISC": []  # Other entities
        }
        
        # Common organization keywords
        org_keywords = ["ETH", "University", "Institut", "Center", "Department", "Lab", 
                        "Association", "Universität", "Zentrum", "Bibliothek"]
        
        # Common person title prefixes
        person_prefixes = ["Prof.", "Dr.", "Professor", "Professor.", "Profesor"]
        
        # Common location keywords
        location_keywords = ["Zurich", "Switzerland", "Zürich", "Building", "Campus", 
                             "Schweiz", "Hönggerberg"]
        
        # Process each entity
        for entity in entities:
            entity_text = str(entity).strip()
            
            if not entity_text:
                continue
            
            # Apply heuristic categorization rules
            if any(entity_text.startswith(prefix) for prefix in person_prefixes) or \
               re.search(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', entity_text):  # Name pattern like "John Smith"
                categorized["PERSON"].append(entity_text)
            elif any(keyword in entity_text for keyword in org_keywords):
                categorized["ORG"].append(entity_text)
            elif any(keyword in entity_text for keyword in location_keywords):
                if "Building" in entity_text or "Campus" in entity_text:
                    categorized["LOC"].append(entity_text)
                else:
                    categorized["GPE"].append(entity_text)
            else:
                # Default category
                categorized["MISC"].append(entity_text)
        
        return categorized
    
    def normalize_entities(self, all_entities: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        Normalize entity names to handle variants of the same entity.
        
        Args:
            all_entities: Dictionary mapping article IDs to categorized entities
            
        Returns:
            Dictionary with normalized entity names
        """
        print("Normalizing entity names...")
        
        # Common entity mappings
        mappings = {
            "ETH Zurich": ["ETH", "ETH Zürich", "ETHZ", "ETH Zentrum"],
            "Zurich": ["Zürich", "Zurich"],
            "Switzerland": ["Schweiz", "Switzerland"]
        }
        
        normalized_entities = {}
        
        for article_id, categories in all_entities.items():
            normalized = {
                "PERSON": [],
                "ORG": [],
                "GPE": [],
                "LOC": [],
                "MISC": []
            }
            
            # Normalize each category
            for category, entities in categories.items():
                for entity in entities:
                    # Check if this entity matches any mapping
                    normalized_entity = entity
                    for standard_form, variants in mappings.items():
                        if entity in variants:
                            normalized_entity = standard_form
                            break
                    
                    # Add to normalized list if not already present
                    if normalized_entity not in normalized[category]:
                        normalized[category].append(normalized_entity)
            
            normalized_entities[article_id] = normalized
        
        return normalized_entities