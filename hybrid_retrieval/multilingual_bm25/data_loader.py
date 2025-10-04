import os
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any


def load_hknews_documents(source_dir: str = "HKNews/") -> List[Dict[str, Any]]:
    """
    Load documents from the HKNews directory structure.
    
    Expected structure:
    HKNews/
    ├── en_documents/
    │   ├── 2023/
    │   │   ├── 01/
    │   │   │   ├── doc1.json
    │   │   │   └── doc2.json
    │   │   └── 02/
    │   └── 2024/
    └── de_documents/
        └── ...
    
    Args:
        source_dir: Path to the HKNews directory (default: "HKNews/")
        
    Returns:
        List of processed documents with standardized structure
    """
    source_path = Path(source_dir)
    documents = []
    
    print(f"Loading documents from {source_path}...")
    
    if not source_path.exists():
        print(f"Error: Directory {source_path} not found")
        return documents
    
    # Find all language directories
    language_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    total_docs = 0
    en_docs = 0
    de_docs = 0
    
    for lang_dir in language_dirs:
        # Extract language from directory name (e.g., 'en_documents' -> 'en')
        language = lang_dir.name.split('_')[0].lower()
        
        if language not in ['en', 'de']:
            print(f"Skipping directory with unknown language prefix: {lang_dir.name}")
            continue
        
        # Navigate through year directories
        for year_dir in lang_dir.iterdir():
            if not year_dir.is_dir():
                continue
                
            # Navigate through month directories
            for month_dir in year_dir.iterdir():
                if not month_dir.is_dir():
                    continue
                
                # Process all JSON files in this month
                for json_file in month_dir.glob("*.json"):
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            doc_data = json.load(f)
                        
                        # Validate that document has main content
                        main_content = doc_data.get('main_content', '')
                        if not main_content:
                            print(f"Skipping document with empty content: {json_file}")
                            continue
                        
                        # Create standardized document structure
                        doc_id = json_file.stem
                        processed_doc = {
                            'id': doc_id,
                            'language': language,
                            'title': doc_data.get('title', ''),
                            'main_content': main_content,
                            'summary': doc_data.get('summary', ''),
                            'keywords': doc_data.get('keywords', []),
                            'topics': doc_data.get('topics', []),
                            # Preserve original metadata if needed
                            'source_file': str(json_file),
                            'year': year_dir.name,
                            'month': month_dir.name
                        }
                        
                        documents.append(processed_doc)
                        total_docs += 1
                        
                        if language == 'en':
                            en_docs += 1
                        else:
                            de_docs += 1
                    
                    except Exception as e:
                        print(f"Error loading {json_file}: {e}")
    
    print(f"Successfully loaded {total_docs} documents ({en_docs} English, {de_docs} German)")
    return documents


def create_temp_documents(documents: List[Dict[str, Any]]) -> str:
    """
    Create temporary JSON files from document list for the retriever.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        Path to temporary directory containing JSON files
    """
    temp_dir = tempfile.mkdtemp()
    print(f"Creating temporary documents in {temp_dir}")
    
    for i, doc in enumerate(documents):
        file_path = os.path.join(temp_dir, f"doc_{i+1}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
    
    print(f"Created {len(documents)} document files")
    return temp_dir


def load_documents_from_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Alternative loader for documents already in a flat directory structure.
    
    Args:
        directory: Path to directory containing JSON documents
        
    Returns:
        List of documents
    """
    documents = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Directory {directory} not found")
        return documents
    
    print(f"Loading documents from flat directory: {directory}")
    
    for json_file in dir_path.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                doc = json.load(f)
            
            # Ensure required fields exist
            if 'language' in doc and doc.get('main_content'):
                documents.append(doc)
            else:
                print(f"Skipping invalid document: {json_file}")
        
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    print(f"Loaded {len(documents)} documents from directory")
    return documents


def validate_document_structure(doc: Dict[str, Any]) -> bool:
    """
    Validate that a document has the required structure for the retriever.
    
    Args:
        doc: Document dictionary
        
    Returns:
        True if document is valid, False otherwise
    """
    required_fields = ['language', 'main_content']
    optional_fields = ['title', 'summary', 'keywords', 'topics']
    
    # Check required fields
    for field in required_fields:
        if field not in doc or not doc[field]:
            print(f"Document missing required field: {field}")
            return False
    
    # Check language is supported
    if doc['language'].lower() not in ['en', 'de']:
        print(f"Unsupported language: {doc['language']}")
        return False
    
    # Ensure optional fields are present (can be empty)
    for field in optional_fields:
        if field not in doc:
            doc[field] = [] if field in ['keywords', 'topics'] else ""
    
    return True


def filter_valid_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter documents to keep only those with valid structure.
    
    Args:
        documents: List of document dictionaries
        
    Returns:
        List of valid documents
    """
    valid_docs = []
    
    for doc in documents:
        if validate_document_structure(doc):
            valid_docs.append(doc)
    
    print(f"Kept {len(valid_docs)} valid documents out of {len(documents)} total")
    return valid_docs