from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]      # project root
sys.path.insert(0, str(BASE_DIR))                   # ensure project root on sys.path
# --- END robust path setup ---

import os, json, pickle
from typing import List, Dict
from pathlib import Path
from collections import defaultdict

import networkx as nx
# import your GraphBuilder from its location
# from your_module.graph_builder import GraphBuilder
from hybrid_retrieval.graphrag.graph_builder import GraphBuilder  # ✅ correct



DOCS_DIR = Path("data/documents")
OUT_DIR = Path("data/graphrag")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_PATH = OUT_DIR / "graph.pickle"
CHUNKS_PATH = OUT_DIR / "entity_chunks.pickle"

def load_articles() -> List[Dict]:
    articles = []
    for fn in DOCS_DIR.glob("*.json"):
        with open(fn, "r", encoding="utf-8") as f:
            doc = json.load(f)
        # Ensure required fields; derive id from filename if missing
        doc.setdefault("id", fn.stem)  # e.g., eth_ai_initiative_1
        doc.setdefault("language", "en")
        doc.setdefault("date", "")  # fill ISO string if you have it
        doc.setdefault("topics", doc.get("topics", []))
        doc.setdefault("title", doc.get("title", ""))
        doc.setdefault("main_content", doc.get("main_content", ""))
        articles.append(doc)
    return articles

def build_processed_entities_from_docs(articles: List[Dict]) -> Dict[str, Dict]:
    """
    Simple fallback: use topics/keywords as entities if you don’t have NER yet.
    You can replace this with spaCy NER later.
    """
    out: Dict[str, Dict] = {}
    for a in articles:
        cats = defaultdict(list)
        for t in a.get("topics", []):
            cats["TOPIC"].append(str(t))
        for k in a.get("keywords", []):
            cats["KEYWORD"].append(str(k))
        out[str(a["id"])] = dict(cats)
    return out

def main():
    articles = load_articles()
    if not articles:
        print("No docs in data/documents. Add JSONs first.")
        return

    processed_entities = build_processed_entities_from_docs(articles)

    gb = GraphBuilder()
    G, entity_chunks = gb.create_knowledge_graph(articles, processed_entities)

    # Save
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(entity_chunks, f)

    print(f"Graph saved to: {GRAPH_PATH}")
    print(f"Entity chunks saved to: {CHUNKS_PATH}")
    print(f"Nodes: {G.number_of_nodes()}  Edges: {G.number_of_edges()}")

if __name__ == "__main__":
    main()
