from data_loader import ETHNewsDataLoader

def test_data_loader():
    # Create loader with default HKNews path
    loader = ETHNewsDataLoader()
    
    # Load a limited number of articles for testing
    articles = loader.load_articles(limit=10)
    
    # Print summary of loaded articles
    print("\nArticle Summary:")
    for idx, article in enumerate(articles):
        print(f"\nArticle {idx+1}: {article['id']}")
        print(f"Title: {article['title']}")
        print(f"Language: {article['language']}")
        print(f"Named Entities: {article['named_entities'][:5]}...")

if __name__ == "__main__":
    test_data_loader()