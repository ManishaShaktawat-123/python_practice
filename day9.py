import os
import asyncio
import numpy as np
from typing import List, Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

load_dotenv()

app = FastAPI(title="Day 9: Production RAG with Chunking & Threshold Filtering")

# Initialize Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 1. Raw Documents
raw_documents = [
    {
        "doc_id": "doc_1",
        "content": "FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints. It is built on top of Starlette for web parts and Pydantic for data parts."
    },
    {
        "doc_id": "doc_2",
        "content": "Retrieval-Augmented Generation (RAG) is a technique that optimizes the output of a Large Language Model by referencing an authoritative knowledge base outside its training data sources before generating a response."
    },
    {
        "doc_id": "doc_3",
        "content": "Gemini 2.5 Flash is designed for high-frequency, low-latency tasks where speed and cost-efficiency are critical. It supports multimodal inputs including text, images, audio, and video."
    }
]

# 2. Text Chunking
def chunk_text(text: str, chunk_size: int = 120, overlap: int = 30) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
    return chunks

chunk_store: List[Dict] = []
for doc in raw_documents:
    text_chunks = chunk_text(doc["content"])
    for idx, chunk_str in enumerate(text_chunks):
        chunk_store.append({
            "chunk_id": f"{doc['doc_id']}_c{idx}",
            "doc_id": doc["doc_id"],
            "text": chunk_str
        })

def get_embedding_sync(text: str) -> list:
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

# Pre-computed embeddings store
chunk_embeddings: List[list] = []

@app.on_event("startup")
async def startup_event():
    """Server start hone par safely embeddings pre-compute karega"""
    loop = asyncio.get_running_loop()
    global chunk_embeddings
    chunk_embeddings = []
    for c in chunk_store:
        emb = await loop.run_in_executor(None, get_embedding_sync, c["text"])
        chunk_embeddings.append(emb)

# Request & Response Schemas
class RAGQueryRequest(BaseModel):
    query: str
    similarity_threshold: float = 0.50

class RAGQueryResponse(BaseModel):
    query: str
    matched_chunk_id: str
    retrieved_context: str
    similarity_score: float
    answer: str

@app.post("/rag/v2/ask", response_model=RAGQueryResponse)
async def ask_rag_v2(request: RAGQueryRequest):
    try:
        loop = asyncio.get_running_loop()

        # 1. User Query Embedding
        query_emb = await loop.run_in_executor(None, get_embedding_sync, request.query)

        # 2. Vector Search
        scores = [cosine_similarity(query_emb, c_emb) for c_emb in chunk_embeddings]
        best_idx = int(np.argmax(scores))
        best_score = round(scores[best_idx], 4)

        # 3. Threshold check
        if best_score < request.similarity_threshold:
            raise HTTPException(
                status_code=404, 
                detail=f"No relevant context found above threshold ({request.similarity_threshold}). Highest score: {best_score}"
            )

        matched_chunk = chunk_store[best_idx]

        prompt = f"""
You are a precise technical assistant. Answer the user question strictly based ONLY on the context provided below.

Context:
"{matched_chunk['text']}"

Question:
{request.query}
"""

        def generate_sync():
            return client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

        response = await loop.run_in_executor(None, generate_sync)

        return RAGQueryResponse(
            query=request.query,
            matched_chunk_id=matched_chunk["chunk_id"],
            retrieved_context=matched_chunk["text"],
            similarity_score=best_score,
            answer=response.text.strip()
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))