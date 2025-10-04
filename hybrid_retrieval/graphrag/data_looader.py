import json
from pathlib import Path
from typing import List, Dict, Any

class ETHNewsDataLoader:
    """Loads and processes ETH News articles from JSON files."""
    
    def __init__(self, base_dir: str = "HKNews"):
        """
        Initialize with path to HKNews directory.
        
        Args:
            base_dir: Path to the HKNews directory
        """
        self.base_dir = Path(base_dir)
        self.articles = []
    
    def load_articles(self, limit: int = None) -> List[Dict[str, Any]]:
        """
        Load articles from JSON files.
        
        Args:
            limit: Optional limit on number of articles to load
            
        Returns:
            List of processed article dictionaries
        """
        print(f"Loading articles from {self.base_dir}...")
        
        # Check if directory exists
        if not self.base_dir.exists():
            print(f"Error: Directory {self.base_dir} not found")
            return []
        
        articles = []
        total_loaded = 0
        skipped = 0
        
        # Get language directories (en_*, de_*)
        lang_dirs = [d for d in self.base_dir.iterdir() if d.is_dir()]
        
        for lang_dir in lang_dirs:
            # Extract language code
            lang_prefix = lang_dir.name.split('_')[0].lower()
            
            # Only process English and German
            if lang_prefix not in ['en', 'de']:
                continue
                
            # Process all year directories
            year_dirs = [d for d in lang_dir.iterdir() if d.is_dir()]
            for year_dir in year_dirs:
                # Process all month directories
                month_dirs = [d for d in year_dir.iterdir() if d.is_dir()]
                for month_dir in month_dirs:
                    # Process all JSON files in this month
                    json_files = [f for f in month_dir.iterdir() if f.suffix.lower() == '.json']
                    
                    for json_file in json_files:
                        # Check if we've reached the limit
                        if limit and total_loaded >= limit:
                            break
                            
                        try:
                            with open(json_file, 'r', encoding='utf-8') as f:
                                article_data = json.load(f)
                            
                            # Skip articles with empty content
                            main_content = article_data.get('main_content', '')
                            if not main_content:
                                skipped += 1
                                continue
                            
                            # Create a unique ID from file path components
                            article_id = f"{lang_prefix}_{year_dir.name}_{month_dir.name}_{json_file.stem}"
                            
                            # Process the article
                            processed_article = self._process_article(article_data, article_id, lang_prefix, json_file)
                            articles.append(processed_article)
                            total_loaded += 1
                            
                            # Print progress every 100 articles
                            if total_loaded % 100 == 0:
                                print(f"Loaded {total_loaded} articles so far...")
                            
                        except Exception as e:
                            print(f"Error processing {json_file}: {e}")
        
        print(f"Successfully loaded {total_loaded} articles ({skipped} skipped due to empty content)")
        self.articles = articles
        return articles
    
    def _process_article(self, article_data: Dict, article_id: str, language: str, file_path: Path) -> Dict:
        """
        Process an article from raw JSON data.
        
        Args:
            article_data: Raw article data from JSON
            article_id: Unique article ID
            language: Article language code
            file_path: Path to the source file
            
        Returns:
            Processed article dictionary
        """
        # Extract core fields
        processed = {
            "id": article_id,
            "file_path": str(file_path),
            "language": language,
            "title": article_data.get("title", ""),
            "date": article_data.get("date", ""),
            "source": article_data.get("source", ""),
            "main_content": article_data.get("main_content", ""),
            "summary": article_data.get("summary", ""),
            # Use pre-extracted entities if available, otherwise empty list
            "named_entities": article_data.get("named_entities", []),
            "topics": article_data.get("topics", []),
            "keywords": article_data.get("keywords", [])
        }
        
        return processed