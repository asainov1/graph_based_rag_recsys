import json
from groq import Groq
from hybrid_retrieval.hybrid_retriever import HybridRetriever
from hybrid_retrieval.adapters.bm25_adapter import BM25Adapter
from hybrid_retrieval.adapters.dense_adapter import DenseAdapter
from hybrid_retrieval.adapters.graphrag_adapter import GraphRAGAdapter

# Initialize Groq
import os
from dotenv import load_dotenv  # ✅ for loading .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
assert GROQ_API_KEY, "Set GROQ_API_KEY in your environment"


# Initialize retrievers
hybrid = HybridRetriever(
    bm25_adapter=BM25Adapter(),
    dense_adapter=DenseAdapter(),
    graphrag_adapter=GraphRAGAdapter()
)

# Load benchmark questions
with open('hybrid_retrieval/benchmark_qa.json', 'r') as f:
    benchmark = json.load(f)

# Load Groq evaluation results
with open('groq_top5_scores.json', 'r') as f:
    groq_scores = json.load(f)

# Questions with good retrieval (score >= 0.5)
good_questions = []
for item in groq_scores:
    scores = []
    for result in item['results']:
        if 'groq_response' in result and 'Score:' in result['groq_response']:
            try:
                score = float(result['groq_response'].split('Score:')[1].strip())
                scores.append(score)
            except:
                pass
    if scores and max(scores) >= 0.5:
        # Find the benchmark question
        for q in benchmark['questions']:
            if q['question'] == item['question']:
                good_questions.append(q)
                break

print(f"Generating answers for {len(good_questions)} questions with good retrieval...")

results = []
for q in good_questions:  # Test with first 5
    print(f"\nQ{q['id']}: {q['question']}")
    
    # Retrieve documents
    docs = hybrid.retrieve(q['question'], top_k=5)
    
    # Create context from documents
    context = "\n\n".join([f"Document {i+1}: {doc['content'][:500]}..." 
                          for i, (doc, score) in enumerate(docs)])
    
    # Generate answer with Groq
    prompt = f"""Based on the following ETH Zurich documents, answer the question accurately.

Context:
{context}

Question: {q['question']}

Instructions:
- Answer based ONLY on the provided documents
- Be specific and accurate
- If the answer is not in the documents, say "I cannot find this information in the provided documents"
- Keep the answer concise (2-3 sentences)

Answer:"""

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200
    )
    
    answer = response.choices[0].message.content
    print(f"Generated answer: {answer[:100]}...")
    
    results.append({
        "question_id": q['id'],
        "question": q['question'],
        "generated_answer": answer,
        "reference_answer": q.get('answer', '')
    })

# Save results
with open('groq_generated_answers.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} generated answers to groq_generated_answers.json")