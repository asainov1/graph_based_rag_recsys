# dense_adapter.py
import os
from typing import List, Dict, Any, Tuple
from pathlib import Path

from chromadb.utils import embedding_functions as chroma_ef
import chromadb

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter as QFilter, PointStruct

from sentence_transformers import SentenceTransformer
import numpy as np

class DenseAdapter:
    def __init__(self,
                 backend=os.getenv("DENSE_BACKEND", "qdrant"),
                 qdrant_host=os.getenv("QDRANT_HOST", "localhost"),
                 qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
                 qdrant_collection=os.getenv("QDRANT_COLLECTION", "ethz_news"),
                 chroma_path=Path(__file__).resolve().parents[2] / "notebooks" / "chroma_db_fixed",
                 model_name=os.getenv("DENSE_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")):

        self.backend = backend.lower()
        self.model = SentenceTransformer(model_name)

        if self.backend == "qdrant":
            self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.collection = qdrant_collection
            # sanity check collection exists
            self.client.get_collection(self.collection)  # raises if missing
            print(f"✅ Qdrant connected: {qdrant_host}:{qdrant_port}, collection={self.collection}")

        elif self.backend == "chroma":
            self.client = chromadb.PersistentClient(path=str(chroma_path))
            self.collection = self.client.get_collection(
                name="ethz_news",
                embedding_function=chroma_ef.SentenceTransformerEmbeddingFunction(model_name=model_name),
            )
            print(f"✅ Chroma loaded at {chroma_path}, collection=ethz_news")
        else:
            raise ValueError("backend must be 'qdrant' or 'chroma'")

        self.name = f"Dense_{self.backend.capitalize()}"
    

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[0]

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Dict[str, Any], float]]:
        if self.backend == "qdrant":
            vec = self._embed(query).tolist()
            sr = self.client.search(
                collection_name=self.collection,
                query_vector=vec,
                limit=top_k,
                with_payload=True,
                with_vectors=False,
            )
            out = []
            for i, p in enumerate(sr):
                payload = p.payload or {}
                doc = {
                    "id": payload.get("id", f"q_{i}"),
                    "content": payload.get("content", ""),
                    "title": payload.get("title", ""),
                    "metadata": payload,
                    "retriever": self.name,
                    "distance": p.score  # cosine similarity if collection configured that way
                }
                # turn similarity into [0,1] score if needed
                score = float(p.score)
                out.append((doc, score))
            return out

        # chroma
        res = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = res["documents"][0] if res["documents"] else []
        metas = res["metadatas"][0] if res["metadatas"] else []
        dists = res["distances"][0] if res["distances"] else []
        out = []
        for i, (doc, m, d) in enumerate(zip(docs, metas, dists)):
            score = 1.0 / (1.0 + d)  # convert distance to similarity
            out.append(({
                "id": m.get("id", f"c_{i}"),
                "content": doc,
                "title": m.get("title", ""),
                "metadata": m,
                "retriever": self.name,
                "distance": d
            }, score))
        return out

    def get_document_count(self) -> int:
        if self.backend == "qdrant":
            info = self.client.get_collection(self.collection)
            return int(info.points_count or 0)
        return self.collection.count()
    def get_name(self) -> str:
        """Return the name of this retriever"""
        return self.name

