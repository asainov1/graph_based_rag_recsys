import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from pathlib import Path
import json
import re

# For language detection
from langdetect import detect

# For translation
from deep_translator import GoogleTranslator

# For BM25 retrieval
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class MultilingualBM25Retriever:
    """
    A retrieval system that:
    1. Indexes documents in their original language (EN or DE)
    2. Detects query language
    3. Translates the query to the other language
    4. Searches both EN and DE documents using BM25
    """
    
    def __init__(self, docs_directory: str = "data/documents"):
        """
        Initialize the retriever.
        
        Args:
            docs_directory: Path to the directory containing JSON documents
        """
        self.docs_directory = docs_directory
        
        # Storage for documents by language
        self.documents = {
            "en": [],
            "de": []
        }
        
        # BM25 indexes for each language
        self.bm25_indexes = {
            "en": None,
            "de": None
        }
        
        # Tokenized corpus for each language
        self.tokenized_corpus = {
            "en": [],
            "de": []
        }
        
        # Load stopwords
        self.stopwords = {
            "en": set(stopwords.words('english')),
            "de": set(stopwords.words('german'))
        }
        
        # Load documents and build indexes
        self._load_documents()
        self._build_indexes()
    
    def _load_documents(self):
        """Load JSON documents from directory and separate by language"""
        print(f"Loading documents from {self.docs_directory}...")
        
        # Get all JSON files in the directory
        json_files = [f for f in os.listdir(self.docs_directory) if f.endswith('.json')]
        
        for file_name in json_files:
            file_path = os.path.join(self.docs_directory, file_name)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    
                language = doc.get("language", "").lower()
                if language in ["en", "de"]:
                    self.documents[language].append(doc)
                else:
                    print(f"Skipping document with unknown language: {language} - {file_path}")
            
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        print(f"Loaded {len(self.documents['en'])} English documents and {len(self.documents['de'])} German documents")
    
    def _preprocess_text(self, text: str, language: str) -> List[str]:
        """
        Enhanced preprocessing with better handling of technical terms 
        
        Args:
            text: Text to preprocess
            language: Language of the text ('en' or 'de')
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Normalize text 
        normalized_text = text.lower()
        
        # Special handling for technical terms that should be preserved intact
        # This helps with retrieval precision for technical documents
        technical_terms = []
        
        # Find technical terms like 'AI', 'KI', 'ETH', etc.
        tech_pattern = r'\b[A-Z]{2,}\b'  # Uppercase acronyms
        for match in re.finditer(tech_pattern, text):
            term = match.group().lower()
            technical_terms.append(term)
        
        # Find hyphenated technical terms
        hyphenated_words = re.findall(r'\w+-\w+', normalized_text)
        for hyphenated in hyphenated_words:
            # Add both the hyphenated version and the space-separated version
            normalized_text += " " + hyphenated.replace('-', ' ')
        
        # Tokenize
        tokens = word_tokenize(normalized_text)
        
        # Remove stopwords but keep alphanumeric tokens
        filtered_tokens = []
        for token in tokens:
            # Keep tokens that aren't pure stopwords
            if token not in self.stopwords.get(language, set()):
                # Keep alphanumeric tokens
                if any(c.isalnum() for c in token):
                    filtered_tokens.append(token)
        
        # Add the technical terms back to ensure they're included
        filtered_tokens.extend(technical_terms)
        
        return filtered_tokens
    
    def _build_indexes(self):
        """Build BM25 indexes with improved parameters for better relevance"""
        print("Building BM25 indexes...")
        
        # Process English documents
        for doc in self.documents["en"]:
            # Extract fields
            title = doc.get("title", "")
            content = doc.get("main_content", "")
            summary = doc.get("summary", "")
            keywords = doc.get("keywords", [])
            topics = doc.get("topics", [])
            
            # Improved weighting approach:
            # 1. Topic matching is critical, especially for technical subjects
            # 2. Title is very important for relevance
            # 3. Keywords are good signals
            # 4. Main content needs proper weight for longer documents
            weighted_parts = []
            
            # Add title with higher weight (6x)
            for _ in range(6):
                weighted_parts.append(title)
            
            # Topics have high weight (5x) - these help with topical relevance
            if topics:
                topics_text = " ".join(str(t) for t in topics)
                for _ in range(5):
                    weighted_parts.append(topics_text)
            
            # Keywords also have very high weight (5x)
            if keywords:
                keyword_text = " ".join(str(k) for k in keywords)
                for _ in range(5):
                    weighted_parts.append(keyword_text)
            
            # Add summary (3x)
            for _ in range(3):
                weighted_parts.append(summary)
            
            # Add content with proper weight (1x but it's typically longer)
            weighted_parts.append(content)
            
            # Join with spaces to create combined text
            combined_text = " ".join(weighted_parts)
            
            # Tokenize and add to corpus
            tokenized_doc = self._preprocess_text(combined_text, "en")
            self.tokenized_corpus["en"].append(tokenized_doc)
        
        # Process German documents with the same improved approach
        for doc in self.documents["de"]:
            title = doc.get("title", "")
            content = doc.get("main_content", "")
            summary = doc.get("summary", "")
            keywords = doc.get("keywords", [])
            topics = doc.get("topics", [])
            
            weighted_parts = []
            
            # Add title with high weight (6x)
            for _ in range(6):
                weighted_parts.append(title)
            
            # Topics have high weight (5x)
            if topics:
                topics_text = " ".join(str(t) for t in topics)
                for _ in range(5):
                    weighted_parts.append(topics_text)
            
            # Keywords also have very high weight (5x)
            if keywords:
                keyword_text = " ".join(str(k) for k in keywords)
                for _ in range(5):
                    weighted_parts.append(keyword_text)
            
            # Add summary (3x)
            for _ in range(3):
                weighted_parts.append(summary)
            
            # Add content
            weighted_parts.append(content)
            
            # Join with spaces
            combined_text = " ".join(weighted_parts)
            
            # Tokenize and add to corpus
            tokenized_doc = self._preprocess_text(combined_text, "de")
            self.tokenized_corpus["de"].append(tokenized_doc)
        
        # Use better BM25 parameters for small corpora
        # k1=2.0: Increase term frequency importance
        # b=0.25: Reduce document length normalization (works better for small corpora)
        if self.tokenized_corpus["en"]:
            print("Creating English BM25 index...")
            self.bm25_indexes["en"] = BM25Okapi(self.tokenized_corpus["en"], k1=2.0, b=0.25)
        
        if self.tokenized_corpus["de"]:
            print("Creating German BM25 index...")
            self.bm25_indexes["de"] = BM25Okapi(self.tokenized_corpus["de"], k1=2.0, b=0.25)
        
        print("BM25 indexes built successfully")
    
    def detect_language(self, query: str) -> str:
        """
        Detect the language of the query
        
        Args:
            query: User query
            
        Returns:
            Language code ('en' or 'de')
        """
        try:
            lang = detect(query)
            if lang == 'en':
                return 'en'
            elif lang == 'de':  # Only detect standard German
                return 'de'
            else:
                print(f"Language detected as {lang}, defaulting to English")
                return 'en'
        except:
            print("Language detection failed, defaulting to English")
            return 'en'
    
    def translate_query(self, query: str, source_lang: str, target_lang: str) -> str:
        """
        Translate query from source language to target language
        
        Args:
            query: Query to translate
            source_lang: Source language code ('en' or 'de')
            target_lang: Target language code ('en' or 'de')
            
        Returns:
            Translated query
        """
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            translated = translator.translate(query)
            return translated
        except Exception as e:
            print(f"Translation error: {e}")
            return query  # Return original query if translation fails
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Improved search with smarter cross-lingual balancing
        
        Args:
            query: User query
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents with scores
        """
        # Detect query language
        query_lang = self.detect_language(query)
        
        # Get the other language
        other_lang = "de" if query_lang == "en" else "en"
        
        print(f"Query language detected as {query_lang}")
        
        # Translate query to the other language
        translated_query = self.translate_query(query, query_lang, other_lang)
        
        print(f"Original query: {query}")
        print(f"Translated query: {translated_query}")
        
        # Search in original language - get more results for better filtering
        original_results = self._search_language(query, query_lang, top_k * 2)
        
        # Search in translated language
        translated_results = self._search_language(translated_query, other_lang, top_k * 2)
        
        # More intelligent balancing:
        # 1. Favor highly-relevant primary language results (score > 0.5)
        # 2. Include top secondary language results for diversity
        # 3. Ensure relevant content (especially high-scored matches)
        
        # Combine all results
        all_results = original_results + translated_results
        
        # Sort by score (descending)
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Top half from primary language (if available)
        primary_results = [r for r in all_results if r["language"] == query_lang]
        # Include some high-scored results first
        high_scored_primary = [r for r in primary_results if r["score"] > 0]
        
        # Include some secondary language results
        secondary_results = [r for r in all_results if r["language"] == other_lang]
        # Include high-scored cross-lingual results
        high_scored_secondary = [r for r in secondary_results if r["score"] > 0]
        
        # Balanced allocation algorithm:
        # 1. Start with high-scoring results from both languages (if available)
        # 2. Fill the rest with best scoring results from either language
        balanced_results = []
        
        # Add high-scored primary language results first (up to 60% of total)
        primary_count = min(len(high_scored_primary), int(top_k * 0.6))
        balanced_results.extend(high_scored_primary[:primary_count])
        
        # Add high-scored secondary language results (up to 40% of total)
        secondary_count = min(len(high_scored_secondary), int(top_k * 0.4))
        balanced_results.extend(high_scored_secondary[:secondary_count])
        
        # Fill remaining spots with the best remaining results
        remaining_slots = top_k - len(balanced_results)
        if remaining_slots > 0:
            # Get results that weren't already included
            remaining_results = [r for r in all_results if r not in balanced_results]
            balanced_results.extend(remaining_results[:remaining_slots])
        
        # Final sort by score
        balanced_results.sort(key=lambda x: x["score"], reverse=True)
        
        return balanced_results[:top_k]
    
    def _search_language(self, query: str, language: str, top_k: int) -> List[Dict]:
        """
        Enhanced search with topic and keyword boosting
        
        Args:
            query: Query (in the language specified)
            language: Language to search in ('en' or 'de')
            top_k: Number of top results to return
            
        Returns:
            List of retrieved documents with scores
        """
        results = []
        
        # Check if we have documents and an index for this language
        if not self.documents[language] or self.bm25_indexes[language] is None:
            return results
        
        # Step 1: Run an initial search with the original query to identify relevant topics
        query_lower = query.lower()
        tokenized_query = self._preprocess_text(query, language)
        
        if not tokenized_query:
            return results
        
        # Get initial BM25 scores
        initial_scores = self.bm25_indexes[language].get_scores(tokenized_query)
        
        # Get initial top results to extract topics for query expansion
        initial_top_indices = np.argsort(initial_scores)[::-1][:3]  # Get top 3 initial matches
        
        # Step 2: Extract topics and keywords from initial top results to expand the query
        expansion_terms = set()
        
        for idx in initial_top_indices:
            if initial_scores[idx] > 0:  # Only use positively scored documents
                doc = self.documents[language][idx]
                
                # Extract key information from the document
                topics = doc.get("topics", [])
                keywords = doc.get("keywords", [])
                
                # Add topics and keywords to expansion terms
                for topic in topics:
                    # Add individual words from topics
                    for word in str(topic).lower().split():
                        if len(word) > 3 and word not in self.stopwords.get(language, set()):
                            expansion_terms.add(word)
                
                # Add keywords
                for keyword in keywords:
                    expansion_terms.add(str(keyword).lower())
        
        # Step 3: Create expanded query
        boosted_query = query
        
        # Add expansion terms if we found any
        if expansion_terms:
            boosted_query += " " + " ".join(expansion_terms)
        
        # Always add ETH to the query for institutional relevance
        if "eth" not in query_lower:
            boosted_query += " ETH"
        
        # Step 4: Run the final search with the expanded query
        final_tokenized_query = self._preprocess_text(boosted_query, language)
        
        if not final_tokenized_query:
            return results
        
        # Get BM25 scores with expanded query
        scores = self.bm25_indexes[language].get_scores(final_tokenized_query)
        
        # Get top_k document indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Post-score adjustment: Boost documents based on query-document relevance metrics
        adjusted_scores = []
        for idx in top_indices:
            score = float(scores[idx])
            doc = self.documents[language][idx]
            
            # Extract document metadata
            title = doc.get("title", "").lower()
            topics = [str(t).lower() for t in doc.get("topics", [])]
            
            # Boost documents with title and topic matches to original query
            for query_term in query_lower.split():
                # Skip very short terms
                if len(query_term) <= 2:
                    continue
                    
                # Title match is a strong signal
                if query_term in title:
                    score += 0.5
                
                # Topic match is also a good signal
                for topic in topics:
                    if query_term in topic:
                        score += 0.3
            
            adjusted_scores.append((idx, score))
        
        # Sort by adjusted score
        adjusted_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Create result objects
        for idx, score in adjusted_scores:
            doc = self.documents[language][idx]
            
            result = {
                "document": doc,
                "score": score,
                "language": language,
                "title": doc.get("title", ""),
                "summary": doc.get("summary", ""),
                "content_snippet": self._get_content_snippet(doc.get("main_content", ""), final_tokenized_query)
            }
            
            results.append(result)
        
        return results[:top_k]
    
    def _get_content_snippet(self, content: str, query_tokens: List[str], max_length: int = 200) -> str:
        """
        Extract a relevant snippet from the content based on query tokens
        
        Args:
            content: Document content
            query_tokens: Tokenized query
            max_length: Maximum snippet length
            
        Returns:
            Content snippet
        """
        if not content or not query_tokens:
            return ""
        
        # Simple approach: Find first occurrence of any query token
        content_lower = content.lower()
        best_pos = len(content)
        
        for token in query_tokens:
            pos = content_lower.find(token)
            if pos != -1 and pos < best_pos:
                best_pos = pos
        
        # If no query tokens found, return the beginning of the content
        if best_pos == len(content):
            best_pos = 0
        
        # Find a good starting position (start of a sentence if possible)
        start = max(0, best_pos - 100)
        while start > 0 and content[start] not in ".!?\n":
            start -= 1
        
        if start > 0:
            start += 1  # Skip the punctuation
        
        # Find a good ending position (end of a sentence if possible)
        end = min(len(content), best_pos + max_length)
        while end < len(content) and content[end] not in ".!?\n":
            end += 1
        
        if end < len(content):
            end += 1  # Include the punctuation
        
        # Extract snippet
        snippet = content[start:end].strip()
        
        # Add ellipsis if needed
        if start > 0:
            snippet = "..." + snippet
        
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet