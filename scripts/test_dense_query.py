from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

COLLECTION = "ethz_news"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    client = QdrantClient(host="localhost", port=6333)
    model = SentenceTransformer(MODEL_NAME)

    query = "artificial intelligence research at ETH"
    query = "die deutsche Sprache ist gut" # testing 
    qvec = model.encode([query], normalize_embeddings=True)[0].tolist()

    res = client.search(
        collection_name=COLLECTION,
        query_vector=qvec,
        limit=5,
        with_payload=True,
        with_vectors=False,
    )
    for i, r in enumerate(res, 1):
        title = r.payload.get("title", "<no title>")
        print(f"{i}. score={r.score:.4f}  {title}")

if __name__ == "__main__":
    main()
