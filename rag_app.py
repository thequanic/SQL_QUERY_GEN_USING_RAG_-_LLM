import json
import faiss
import numpy as np
import requests
import ollama
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load schema documents
with open("schema_docs.json", "r") as f:
    docs = json.load(f)

texts = [d["text"] for d in docs]

# Create embeddings
embeddings = model.encode(texts)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# ---- RAG Retrieval ----
def retrieve_context(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    
    results = [texts[i] for i in indices[0]]
    return "\n".join(results)

# ---- LLM Call (Ollama) ----
# def call_llm(prompt):
#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={
#             "model": "llama3",
#             "prompt": prompt,
#             "stream": False
#         }
#     )
#     return response.json()["response"]

# ---- LLM Call (Ollama) ----
def call_llm(prompt):
    response=ollama.chat(
        model="llama3",
        messages=[{"role":"user","content":prompt}]
    )
    return response['message']['content']

# ---- Prompt Template ----
def build_prompt(query, context):
    return f"""
You are an expert SQL generator.

Use the following database schema and rules:
{context}

Rules:
- Only generate SELECT queries
- Use correct joins
- Do not hallucinate tables or columns

User Question:
{query}

Output format:
SQL Query:
<query>

Explanation:
<explanation>
"""

# ---- Main Pipeline ----
def generate_sql(query):
    context = retrieve_context(query)
    prompt = build_prompt(query, context)
    
    response = call_llm(prompt)
    return response

# ---- Run Example ----
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    
    result = generate_sql(user_query)
    
    print("\n=== RESULT ===")
    print(result)