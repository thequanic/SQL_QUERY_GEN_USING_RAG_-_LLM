import json
import faiss
import numpy as np
import ollama
import streamlit as st
from sentence_transformers import SentenceTransformer

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="RAG SQL Generator", layout="wide")

st.title("🧠 RAG-based SQL Generator")
st.caption("Ask questions → Get SQL + Explanation")

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ------------------ LOAD DATA ------------------
@st.cache_resource
def load_index():
    with open("schema_docs.json", "r") as f:
        docs = json.load(f)

    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))

    return index, texts

index, texts = load_index()

# ------------------ RAG ------------------
def retrieve_context(query, k=3):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k)
    results = [texts[i] for i in indices[0]]
    return "\n".join(results)

# ------------------ LLM ------------------
def call_llm(prompt):
    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )
    return response['message']['content']

# ------------------ PROMPT ------------------
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

# ------------------ MAIN PIPELINE ------------------
def generate_sql(query):
    context = retrieve_context(query)
    prompt = build_prompt(query, context)
    response = call_llm(prompt)
    return response

# ------------------ CHAT UI ------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Ask your data question...")

if user_input:
    # Store user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Generating SQL..."):
            result = generate_sql(user_input)
            st.markdown(result)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": result})