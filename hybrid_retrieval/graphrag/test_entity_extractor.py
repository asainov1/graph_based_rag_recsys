from data_loader import ETHNewsDataLoader
from entity_extractor import EnhancedEntityExtractor

def test_entity_extraction():
    # Load articles
    loader = ETHNewsDataLoader()
    articles = loader.load_articles(limit=20)
    
    # Process entities
    extractor = EnhancedEntityExtractor()
    entities = extractor.process_entities(articles)
    normalized_entities = extractor.normalize_entities(entities)
    
    # Print sample results
    print("\nSample Entity Categorization:")
    for i, article_id in enumerate(list(entities.keys())[:5]):
        article = next(a for a in articles if a["id"] == article_id)
        print(f"\nArticle: {article['title'][:50]}...")
        print(f"Original entities: {article['named_entities'][:3]}...")
        
        # Show categorized entities
        print("Categorized entities:")
        for category, cat_entities in entities[article_id].items():
            if cat_entities:
                print(f"  - {category}: {cat_entities[:3]}")
        
        # Show any normalization changes
        print("Normalized entities:")
        for category, norm_entities in normalized_entities[article_id].items():
            if norm_entities:
                print(f"  - {category}: {norm_entities[:3]}")

if __name__ == "__main__":
    test_entity_extraction()