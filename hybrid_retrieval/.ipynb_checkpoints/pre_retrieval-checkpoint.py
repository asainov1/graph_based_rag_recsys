import re
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import chromadb
from collections import Counter

@dataclass
class QueryContext:
    """Enhanced query context with metadata and routing information."""
    original_query: str
    processed_query: str
    detected_language: str
    query_type: str
    entities: List[str]
    keywords: List[str]
    expansion_terms: List[str]
    routing_strategy: str
    confidence_score: float
    metadata_filters: Dict[str, Any]

class QueryType(Enum):
    """Different types of queries for routing decisions."""
    FACTUAL = "factual"           # Who, what, when questions
    TEMPORAL = "temporal"         # Time-based queries
    PERSON = "person"             # Person-related queries
    RESEARCH = "research"         # Research/academic queries
    EVENT = "event"               # Event-related queries
    GENERAL = "general"           # General search queries

class RoutingStrategy(Enum):
    """Different retrieval strategies for routing."""
    DENSE_VECTOR = "dense_vector"
    KEYWORD_SEARCH = "keyword_search"
    HYBRID = "hybrid"
    ENTITY_BASED = "entity_based"
    TEMPORAL_SEARCH = "temporal_search"

class LanguageDetector:
    """Simple multilingual query language detection."""
    
    GERMAN_PATTERNS = {
        'question_words': ['wer', 'was', 'wo', 'wann', 'wie', 'warum', 'welche', 'welcher'],
        'common_words': ['der', 'die', 'das', 'und', 'ist', 'ein', 'eine', 'von', 'zu', 'mit'],
        'university_terms': ['universität', 'hochschule', 'forschung', 'studenten', 'professor']
    }
    
    ENGLISH_PATTERNS = {
        'question_words': ['who', 'what', 'where', 'when', 'how', 'why', 'which'],
        'common_words': ['the', 'and', 'is', 'a', 'an', 'of', 'to', 'with', 'for'],
        'university_terms': ['university', 'research', 'students', 'professor', 'academic']
    }
    
    @classmethod
    def detect_language(cls, query: str) -> Tuple[str, float]:
        """Detect query language with confidence score."""
        query_lower = query.lower()
        
        german_score = 0
        english_score = 0
        
        # Check patterns
        for pattern_type, words in cls.GERMAN_PATTERNS.items():
            for word in words:
                if word in query_lower:
                    german_score += 2 if pattern_type == 'question_words' else 1
        
        for pattern_type, words in cls.ENGLISH_PATTERNS.items():
            for word in words:
                if word in query_lower:
                    english_score += 2 if pattern_type == 'question_words' else 1
        
        total_score = german_score + english_score
        if total_score == 0:
            return "en", 0.5  # Default to English with low confidence
        
        if german_score > english_score:
            confidence = german_score / total_score
            return "de", confidence
        else:
            confidence = english_score / total_score
            return "en", confidence

class QueryExpander:
    """Language-agnostic query expansion using metadata and patterns."""
    
    # Multilingual synonym mappings
    SYNONYMS = {
        'en': {
            'president': ['president', 'chief', 'head', 'leader', 'director'],
            'rector': ['rector', 'vice-chancellor', 'provost'],
            'student': ['student', 'pupil', 'scholar', 'undergraduate', 'graduate'],
            'research': ['research', 'study', 'investigation', 'project', 'work'],
            'professor': ['professor', 'faculty', 'academic', 'researcher', 'scientist'],
            'university': ['university', 'institution', 'college', 'school', 'eth'],
            'grant': ['grant', 'funding', 'award', 'scholarship', 'fellowship']
        },
        'de': {
            'präsident': ['präsident', 'leiter', 'chef', 'direktor'],
            'rektor': ['rektor', 'vizekanzler', 'prorektor'],
            'student': ['student', 'studentin', 'schüler', 'studierende'],
            'forschung': ['forschung', 'studie', 'untersuchung', 'projekt', 'arbeit'],
            'professor': ['professor', 'professorin', 'fakultät', 'forscher', 'wissenschaftler'],
            'universität': ['universität', 'hochschule', 'institution', 'eth'],
            'stipendium': ['stipendium', 'förderung', 'auszeichnung', 'fellowship']
        }
    }
    
    # Cross-language term mappings for true language-agnostic expansion
    CROSS_LANGUAGE_TERMS = {
        'president': ['president', 'präsident'],
        'rector': ['rector', 'rektor'],
        'student': ['student', 'studentin', 'studierende'],
        'research': ['research', 'forschung'],
        'professor': ['professor', 'professorin'],
        'university': ['university', 'universität', 'eth'],
        'grant': ['grant', 'stipendium', 'förderung']
    }
    
    def __init__(self, chromadb_client, collection_name: str):
        self.client = chromadb_client
        self.collection_name = collection_name
        self._load_metadata_terms()
    
    def _load_metadata_terms(self):
        """Load common terms from existing metadata for expansion."""
        try:
            collection = self.client.get_collection(self.collection_name)
            sample = collection.get(limit=1000, include=['metadatas'])
            
            # Extract common keywords and entities
            all_keywords = []
            all_entities = []
            
            for meta in sample['metadatas']:
                keywords = meta.get('keywords', '')
                if keywords:
                    all_keywords.extend([k.strip() for k in keywords.split(',')])
                
                entities = meta.get('ner_entities', '')
                if entities:
                    all_entities.extend([e.strip() for e in entities.split(',')])
            
            # Keep most common terms for expansion
            self.common_keywords = [k for k, count in Counter(all_keywords).most_common(100)]
            self.common_entities = [e for e, count in Counter(all_entities).most_common(100)]
            
        except Exception as e:
            print(f"Warning: Could not load metadata terms: {e}")
            self.common_keywords = []
            self.common_entities = []
    
    def expand_query(self, query: str, language: str, expansion_type: str = "moderate") -> List[str]:
        """Expand query with synonyms and related terms."""
        expansion_terms = []
        query_lower = query.lower()
        
        # 1. Synonym expansion
        synonyms = self.SYNONYMS.get(language, {})
        for word in query.split():
            word_clean = re.sub(r'[^\w]', '', word.lower())
            if word_clean in synonyms:
                expansion_terms.extend(synonyms[word_clean])
        
        # 2. Cross-language expansion (language-agnostic)
        for eng_term, multilingual_terms in self.CROSS_LANGUAGE_TERMS.items():
            if any(term in query_lower for term in multilingual_terms):
                expansion_terms.extend(multilingual_terms)
        
        # 3. Metadata-based expansion
        if expansion_type in ["aggressive", "moderate"]:
            # Add related entities if query contains entity-like terms
            for entity in self.common_entities:
                if any(word in entity.lower() for word in query.split() if len(word) > 3):
                    expansion_terms.append(entity)
        
        # 4. Keyword expansion from metadata
        if expansion_type == "aggressive":
            for keyword in self.common_keywords:
                if any(word in keyword.lower() for word in query.split() if len(word) > 4):
                    expansion_terms.append(keyword)
        
        # Remove duplicates and return unique terms
        unique_terms = list(set(expansion_terms))
        return [term for term in unique_terms if term.lower() not in query_lower]

class QueryClassifier:
    """Classify queries into different types for routing decisions."""
    
    QUERY_PATTERNS = {
        QueryType.FACTUAL: [
            r'\b(who|wer)\s+(is|was|were|ist|war)\b',
            r'\b(what|was)\s+(is|are|ist|sind)\b',
            r'\b(when|wann)\s+(did|was|ist)\b'
        ],
        QueryType.TEMPORAL: [
            r'\b(in|between|during|zwischen|während)\s+\d{4}\b',
            r'\b(since|until|bis|seit)\b',
            r'\b(year|years|jahr|jahre)\b'
        ],
        QueryType.PERSON: [
            r'\b(president|rector|professor|director|präsident|rektor|professorin?)\b',
            r'\b[A-Z][a-z]+\s+[A-Z][a-z]+\b'  # Name pattern
        ],
        QueryType.RESEARCH: [
            r'\b(research|study|project|grant|forschung|studie|projekt|stipendium)\b',
            r'\b(erc|nsf|snf|horizon)\b'
        ],
        QueryType.EVENT: [
            r'\b(conference|workshop|lecture|seminar|konferenz|workshop|vorlesung)\b',
            r'\b(award|prize|auszeichnung|preis)\b'
        ]
    }
    
    @classmethod
    def classify_query(cls, query: str) -> Tuple[QueryType, float]:
        """Classify query type with confidence score."""
        query_lower = query.lower()
        scores = {query_type: 0 for query_type in QueryType}
        
        for query_type, patterns in cls.QUERY_PATTERNS.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, query_lower))
                scores[query_type] += matches
        
        # Find best match
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        
        if max_score == 0:
            return QueryType.GENERAL, 0.5
        
        confidence = max_score / sum(scores.values()) if sum(scores.values()) > 0 else 0.5
        return best_type, confidence

class QueryRouter:
    """Route queries to appropriate retrieval strategies based on analysis."""
    
    ROUTING_RULES = {
        QueryType.FACTUAL: RoutingStrategy.ENTITY_BASED,
        QueryType.TEMPORAL: RoutingStrategy.TEMPORAL_SEARCH,
        QueryType.PERSON: RoutingStrategy.ENTITY_BASED,
        QueryType.RESEARCH: RoutingStrategy.HYBRID,
        QueryType.EVENT: RoutingStrategy.HYBRID,
        QueryType.GENERAL: RoutingStrategy.DENSE_VECTOR
    }
    
    @classmethod
    def route_query(cls, query_context: QueryContext) -> Dict[str, Any]:
        """Determine routing strategy and parameters."""
        query_type = QueryType(query_context.query_type)
        base_strategy = cls.ROUTING_RULES.get(query_type, RoutingStrategy.DENSE_VECTOR)
        
        routing_config = {
            'strategy': base_strategy.value,
            'collections': [query_context.routing_strategy],
            'filters': query_context.metadata_filters,
            'boost_factors': {}
        }
        
        # Add specific configurations based on query characteristics
        if base_strategy == RoutingStrategy.ENTITY_BASED:
            routing_config['boost_factors']['entities'] = 2.0
            routing_config['primary_fields'] = ['ner_entities', 'keywords']
        
        elif base_strategy == RoutingStrategy.TEMPORAL_SEARCH:
            routing_config['boost_factors']['date'] = 1.5
            routing_config['sort_by'] = 'date'
        
        elif base_strategy == RoutingStrategy.HYBRID:
            routing_config['vector_weight'] = 0.7
            routing_config['keyword_weight'] = 0.3
        
        # Language-specific routing
        if query_context.detected_language:
            routing_config['filters']['language'] = query_context.detected_language
        
        return routing_config

class QueryRewriter:
    """Rewrite queries for better retrieval performance."""
    
    REWRITE_PATTERNS = {
        # Convert questions to statements
        r'^(who is|who was|wer ist|wer war)\s+(.+)': r'\2',
        r'^(what is|what are|was ist|was sind)\s+(.+)': r'\2',
        r'^(when did|when was|wann war)\s+(.+)': r'\2',
        
        # Expand abbreviations
        r'\beth\b': 'ETH Zurich Swiss Federal Institute Technology',
        r'\berc\b': 'European Research Council ERC',
        r'\bsnf\b': 'Swiss National Science Foundation SNF',
        r'\bai\b': 'artificial intelligence machine learning',
        
        # Standardize terminology
        r'\buniversity\b': 'university institution ETH',
        r'\buniversität\b': 'universität institution ETH',
    }
    
    @classmethod
    def rewrite_query(cls, query: str, query_context: QueryContext) -> str:
        """Rewrite query for better retrieval."""
        rewritten = query
        
        # Apply pattern-based rewrites
        for pattern, replacement in cls.REWRITE_PATTERNS.items():
            rewritten = re.sub(pattern, replacement, rewritten, flags=re.IGNORECASE)
        
        # Add expansion terms
        if query_context.expansion_terms:
            # Add most relevant expansion terms
            top_expansions = query_context.expansion_terms[:3]
            rewritten += " " + " ".join(top_expansions)
        
        # Add entity context
        if query_context.entities:
            # Add key entities for context
            key_entities = [e for e in query_context.entities if len(e) > 2][:2]
            rewritten += " " + " ".join(key_entities)
        
        return rewritten.strip()

class PreRetrievalPipeline:
    """Main pre-retrieval pipeline orchestrating all components."""
    
    def __init__(self, chromadb_client, collection_name: str):
        self.client = chromadb_client
        self.collection_name = collection_name
        self.expander = QueryExpander(chromadb_client, collection_name)
    
    def process_query(self, query: str, expansion_type: str = "moderate") -> QueryContext:
        """Complete pre-retrieval processing pipeline."""
        
        # Step 1: Language Detection
        detected_language, lang_confidence = LanguageDetector.detect_language(query)
        
        # Step 2: Query Classification  
        query_type, type_confidence = QueryClassifier.classify_query(query)
        
        # Step 3: Entity and Keyword Extraction
        entities = self._extract_entities(query)
        keywords = self._extract_keywords(query)
        
        # Step 4: Query Expansion
        expansion_terms = self.expander.expand_query(query, detected_language, expansion_type)
        
        # Step 5: Metadata Filter Generation
        metadata_filters = self._generate_filters(query, detected_language, entities)
        
        # Create query context
        query_context = QueryContext(
            original_query=query,
            processed_query=query,  # Will be updated by rewriter
            detected_language=detected_language,
            query_type=query_type.value,
            entities=entities,
            keywords=keywords,
            expansion_terms=expansion_terms,
            routing_strategy=self.collection_name,
            confidence_score=min(lang_confidence, type_confidence),
            metadata_filters=metadata_filters
        )
        
        # Step 6: Query Rewriting
        query_context.processed_query = QueryRewriter.rewrite_query(query, query_context)
        
        return query_context
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entity-like terms from query."""
        # Simple entity extraction - can be enhanced with NER
        entities = []
        
        # Extract capitalized words (potential person/place names)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities.extend(capitalized)
        
        # Extract years
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        entities.extend(years)
        
        return entities
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query."""
        # Remove stop words and extract keywords
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'an', 'will', 'by',
                     'der', 'die', 'das', 'ist', 'und', 'ein', 'eine', 'zu', 'von', 'mit', 'auf'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _generate_filters(self, query: str, language: str, entities: List[str]) -> Dict[str, Any]:
        """Generate metadata filters based on query analysis."""
        filters = {}
        
        # Language filter
        if language in ['en', 'de']:
            filters['language'] = language
        
        # Date filter for temporal queries
        years = re.findall(r'\b(19|20)\d{2}\b', query)
        if years:
            # Filter by decade or specific year
            year = years[0]
            filters['date'] = {"$regex": f"{year[:3]}"}  # Match decade
        
        return filters

# Usage example and testing functions
def demo_preretrieval_pipeline(collection_name: str = "test_multilingual"):
    """Demonstrate the pre-retrieval pipeline."""
    
    print("🔄 PRE-RETRIEVAL STRATEGY DEMO")
    print("="*40)
    
    # Initialize
    client = chromadb.PersistentClient(path="./chroma_db")
    pipeline = PreRetrievalPipeline(client, collection_name)
    
    # Test queries
    test_queries = [
        "Who was the president of ETH in 2003?",
        "Wer waren die Rektoren der ETH zwischen 2017 und 2022?", 
        "ERC grants at ETH Zurich",
        "machine learning research projects",
        "student awards and scholarships"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Original Query: '{query}'")
        print("-" * 50)
        
        # Process query through pipeline
        context = pipeline.process_query(query, expansion_type="moderate")
        
        # Display results
        print(f"🌍 Language: {context.detected_language} (confidence: {context.confidence_score:.2f})")
        print(f"📊 Query Type: {context.query_type}")
        print(f"🎯 Entities: {context.entities}")
        print(f"🔑 Keywords: {context.keywords}")
        print(f"📈 Expansion Terms: {context.expansion_terms[:5]}")
        print(f"🔄 Rewritten Query: '{context.processed_query}'")
        print(f"🎛️  Metadata Filters: {context.metadata_filters}")
        
        # Get routing configuration
        routing_config = QueryRouter.route_query(context)
        print(f"🧭 Routing Strategy: {routing_config['strategy']}")
        
if __name__ == "__main__":
    # Run demonstration
    demo_preretrieval_pipeline()