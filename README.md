# 🧠 Graph-Based RAG Recommender System

A **hybrid retrieval system** that combines **BM25**, **Dense Vector Search (Qdrant/Chroma)**, and **GraphRAG** for high-quality retrieval, fusion, and reranking.  
Designed for advanced question-answering, content generation, and retrieval-augmented generation (RAG) pipelines.

![Architecture Diagram](https://user-images.githubusercontent.com/00000000/architecture.png) <!-- optional: replace with your own image -->

---

## 🚀 Features

- 🔸 **BM25 Retrieval** — Multilingual keyword-based search with NLTK preprocessing  
- 🔸 **Dense Retrieval** — SentenceTransformer embeddings + Qdrant/Chroma backends  
- 🔸 **GraphRAG Retrieval** — Knowledge graph–based local and global expansion  
- 🔸 **Hybrid Fusion** — Reciprocal Rank Fusion (RRF) for combining retrievers  
- 🔸 **Multi-Model Reranking** — TinyBERT / MiniLM / BGE ensemble reranker  
- 🔸 **Groq LLM Evaluation** — Optional integration for fast answer generation

---

## 📁 Repository Structure

graph_based_rag_recsys/
├── agents/ # (Optional) multi-agent orchestration logic
├── benchmark/ # Benchmark QA datasets & evaluation utilities
├── data/
│ └── documents/ # Example ETH Zurich articles for BM25 & dense retrievers
├── docker/ # Dockerfiles & deployment configs
├── hybrid_retrieval/ # Core hybrid retriever, reranker, adapters
├── scripts/ # Data ingestion, graph building, Groq evaluation
│ ├── build_graphrag.py
│ ├── ingest_dense.py
│ ├── generate_answers_groq.py
│ └── test_dense_query.py
├── .env.example # Example environment variables (no secrets)
├── .gitignore
├── requirements.txt
├── requirements_min.txt
└── README.md

---

📊 Benchmarking
We provide benchmark/benchmark_qa.json with ETH Zurich–related questions to evaluate retrieval quality and answer accuracy.
You can run:
python scripts/generate_answers_groq.py
to retrieve documents, generate answers with Groq LLM, and compare against reference answers.

🧠 Reranking System
The reranking module supports:
tinybert (fast)
minilm (balanced)
bge (SOTA)
ensemble (combines all)
Reranking boosts precision by reordering top retrieved results using cross-encoder similarity.

📝 Roadmap
 🧭 Add more dense backends (Weaviate, Milvus)
 🧪 Add LangGraph-based multi-agent orchestration
 🖼️ Add Streamlit UI demo for interactive retrieval
 🧮 Add retrieval evaluation metrics (nDCG, Recall@k, MRR)
 
🤝 Contributing
Contributions are welcome!
To contribute:
Fork the repo
Create a new branch (feature/my-feature)
Commit changes (git commit -m "Add new feature")
Push and open a pull request

📜 License
This project is released under the MIT License.
© 2025 Alikhan Sainov

⭐ Support
If you found this project helpful, please consider starring the repo ⭐
Your support helps keep the project alive and visible to others!

📌 Acknowledgements
Qdrant for vector search
SentenceTransformers for embeddings
Groq for ultra-fast LLM inference
ETH Zurich datasets for evaluation examples

## ⚡️ Quickstart

### 1️⃣ Clone the repository
```bash
git clone https://github.com/asainov1/graph_based_rag_recsys.git
cd graph_based_rag_recsys
2️⃣ Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
3️⃣ Install dependencies
pip install -r requirements.txt
4️⃣ Start Qdrant (for dense retrieval)
docker run -p 6333:6333 qdrant/qdrant
5️⃣ Run hybrid retrieval tests
python -m hybrid_retrieval.test_hybrid
🧪 Example Usage
🔹 Test Dense Retriever
python scripts/test_dense_query.py
🔹 Build GraphRAG Knowledge Graph
python scripts/build_graphrag.py
🔹 Generate Answers with Groq LLM
python scripts/generate_answers_groq.py
