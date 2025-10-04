import os, json, uuid
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

DOCS_DIR = os.path.join("data", "documents")
COLLECTION = "ethz_news"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 outputs 384 dims

def load_docs(path: str) -> List[Dict]:
    docs = []
    for fn in os.listdir(path):
        if fn.endswith(".json"):
            with open(os.path.join(path, fn), "r", encoding="utf-8") as f:
                docs.append(json.load(f))
    return docs

def ensure_collection(client: QdrantClient, collection: str, vector_size: int):
    # recreate_collection drops existing; switch to create_collection if you want idempotent
    client.recreate_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

def main():
    # 1) Load docs
    docs = load_docs(DOCS_DIR)
    if not docs:
        print("No docs found in data/documents/. Add some JSONs first.")
        return

    # 2) Start client
    client = QdrantClient(host="localhost", port=6333)

    # 3) Ensure collection exists
    ensure_collection(client, COLLECTION, VECTOR_SIZE)

    # 4) Load embedding model
    model = SentenceTransformer(MODEL_NAME)

    # 5) Build points
    points: List[PointStruct] = []
    texts = []
    for doc in docs:
        text = f"{doc.get('title','')} {doc.get('summary','')} {doc.get('main_content','')}".strip()
        texts.append(text)

    embeddings = model.encode(texts, normalize_embeddings=True)

    for emb, doc in zip(embeddings, docs):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=doc,
            )
        )

    # 6) Upsert
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Ingested {len(points)} points into collection '{COLLECTION}'.")

if __name__ == "__main__":
    main()
