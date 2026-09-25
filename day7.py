import os
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. Sample Knowledge Base (Documents)
documents = [
    "FastAPI is a modern, fast web framework for building APIs with Python.",
    "Gemini 2.5 Flash is Google's lightweight and efficient multimodal AI model.",
    "Retrieval-Augmented Generation (RAG) improves LLM responses using external context.",
    "Python is a high-level programming language widely used in AI and Web Development."
]

def get_embedding(text: str) -> list:
    """Generates a vector embedding for a given input text."""
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def cosine_similarity(a, b):
    """Calculates cosine similarity between two vector arrays."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 2. Pre-compute Document Embeddings
print("Generating embeddings for documents...")
doc_embeddings = [get_embedding(doc) for doc in documents]

# 3. User Query Search
query = "What is RAG in AI?"
print(f"\nUser Query: '{query}'")

query_embedding = get_embedding(query)

# 4. Calculate Similarities
similarities = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]

# 5. Get Top Relevant Document
best_idx = np.argmax(similarities)

print("\n--- Search Results ---")
for idx, sim in enumerate(similarities):
    print(f"Doc {idx + 1} Score: {sim:.4f} -> {documents[idx]}")

print(f"\nMost Relevant Context for RAG:\n>>> {documents[best_idx]}")