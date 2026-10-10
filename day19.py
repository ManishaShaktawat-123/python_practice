import os
import time
import logging
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
import chromadb

# 1. Load Environment Variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Initialize Gemini Client & ChromaDB
client = genai.Client(api_key=GEMINI_API_KEY)
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Collections: One for RAG documents, one specifically for Semantic Cache
rag_collection = chroma_client.get_or_create_collection(name="day19_rag_collection")
cache_collection = chroma_client.get_or_create_collection(name="day19_semantic_cache")

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("semantic_cache_rag")

app = FastAPI(
    title="Day 19 - Semantic Caching & RAG Optimization Pipeline",
    version="1.0"
)

# Request Models
class IngestRequest(BaseModel):
    documents: List[str] = Field(..., description="Documents to populate knowledge base.")

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query to process with semantic cache check.")
    top_k: int = Field(default=3, description="Number of context matches.")
    similarity_threshold: float = Field(default=0.15, description="Maximum distance for cache hit (lower means closer match).")

# Helper Functions
def get_embedding(text: str) -> List[float]:
    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text
    )
    return response.embedding

# Endpoints
@app.post("/ingest")
def ingest_documents(payload: IngestRequest):
    """Ingest documents into the primary RAG database."""
    embeddings = [get_embedding(doc) for doc in payload.documents]
    ids = [str(uuid.uuid4()) for _ in payload.documents]
    metadatas = [{"source": "manual_ingest"} for _ in payload.documents]
    
    rag_collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=payload.documents,
        metadatas=metadatas
    )
    return {"status": "Success", "indexed_count": len(payload.documents)}

@app.post("/query-with-cache")
def query_with_semantic_cache(payload: QueryRequest):
    """
    Checks semantic cache first. If a semantically identical query exists within the 
    threshold, returns cached response instantly. Otherwise, executes full RAG pipeline 
    and caches the result.
    """
    start_time = time.time()
    query_emb = get_embedding(payload.query)
    
    # 1. Check Semantic Cache
    cache_results = cache_collection.query(
        query_embeddings=[query_emb],
        n_results=1
    )
    
    distances = cache_results.get("distances", [[]])[0]
    cached_docs = cache_results.get("documents", [[]])[0]
    cached_metadatas = cache_results.get("metadatas", [[]])[0]
    
    # 2. Cache Hit Verification
    if distances and distances[0] <= payload.similarity_threshold:
        elapsed = time.time() - start_time
        logger.info(f"Cache HIT for query: '{payload.query}' with distance {distances[0]:.4f}")
        return {
            "source": "cache",
            "execution_time_seconds": round(elapsed, 4),
            "similarity_distance": round(distances[0], 4),
            "query": payload.query,
            "answer": cached_docs[0],
            "context_used": cached_metadatas[0].get("context", "")
        }
    
    # 3. Cache Miss: Execute Standard RAG Pipeline
    logger.info(f"Cache MISS for query: '{payload.query}'. Executing RAG retrieval.")
    
    rag_res = rag_collection.query(
        query_embeddings=[query_emb],
        n_results=payload.top_k
    )
    retrieved_docs = rag_res.get("documents", [[]])[0]
    context = " ".join(retrieved_docs) if retrieved_docs else "No context found."
    
    prompt = f"Context:\n{context}\n\nQuestion: {payload.query}\nProvide a concise answer based on context."
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt
    )
    answer = response.text
    
    # 4. Store result in Semantic Cache
    cache_collection.upsert(
        ids=[str(uuid.uuid4())],
        embeddings=[query_emb],
        documents=[answer],
        metadatas=[{"original_query": payload.query, "context": context}]
    )
    
    elapsed = time.time() - start_time
    return {
        "source": "rag_pipeline",
        "execution_time_seconds": round(elapsed, 4),
        "query": payload.query,
        "answer": answer,
        "context_used": context
    }