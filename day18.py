import os
import time
import asyncio
import logging
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv

from fastapi import FastAPI, BackgroundTasks
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
collection = chroma_client.get_or_create_collection(name="async_rag_collection")

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("async_rag")

app = FastAPI(
    title="Day 18 - Async RAG & Concurrent Vector Pipeline",
    version="1.0"
)

# Request Models
class BatchIngestRequest(BaseModel):
    documents: List[str] = Field(..., description="List of raw text documents for asynchronous ingestion.")

class MultiQueryRequest(BaseModel):
    queries: List[str] = Field(..., description="List of parallel queries to retrieve context for.")
    top_k: int = Field(default=3, description="Number of matches per query.")

# Helper Functions
async def async_get_embedding(text: str) -> List[float]:
    """Asynchronously generate embeddings by offloading blocking SDK calls to threadpool."""
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
    )
    return response.embedding

async def async_generate_answer(prompt: str) -> str:
    """Asynchronously generate text completion via Gemini."""
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
    )
    return response.text

# Endpoints
@app.post("/ingest-batch-async", status_code=202)
async def ingest_batch_async(payload: BatchIngestRequest, background_tasks: BackgroundTasks):
    """
    Asynchronously process and vector-index multiple documents concurrently 
    without blocking API execution thread.
    """
    async def process_documents():
        start_time = time.time()
        logger.info(f"Starting background processing for {len(payload.documents)} documents.")
        
        # Concurrent embedding tasks
        embedding_tasks = [async_get_embedding(doc) for doc in payload.documents]
        embeddings = await asyncio.gather(*embedding_tasks)
        
        ids = [str(uuid.uuid4()) for _ in payload.documents]
        metadatas = [{"source": "async_batch", "doc_id": i} for i in range(len(payload.documents))]
        
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=payload.documents,
            metadatas=metadatas
        )
        elapsed = time.time() - start_time
        logger.info(f"Successfully indexed {len(payload.documents)} documents in {elapsed:.2f}s.")

    background_tasks.add_task(process_documents)
    return {
        "status": "Accepted",
        "message": f"Processing {len(payload.documents)} documents in background task.",
        "task": "batch_ingest"
    }

@app.post("/multi-query-async")
async def multi_query_retrieval(payload: MultiQueryRequest):
    """
    Executes multiple retrieval and generation pipelines concurrently via asyncio.gather.
    """
    start_time = time.time()
    
    # 1. Concurrently fetch query embeddings
    embedding_tasks = [async_get_embedding(q) for q in payload.queries]
    query_embeddings = await asyncio.gather(*embedding_tasks)
    
    # 2. Retrieve vector contexts
    retrieval_results = []
    for query_emb in query_embeddings:
        res = collection.query(
            query_embeddings=[query_emb],
            n_results=payload.top_k
        )
        docs = res.get("documents", [[]])[0]
        retrieval_results.append(" ".join(docs) if docs else "No relevant context found.")

    # 3. Concurrently execute LLM generations
    generation_tasks = []
    for query, context in zip(payload.queries, retrieval_results):
        prompt = f"Context:\n{context}\n\nQuestion: {query}\nProvide a concise answer based on context."
        generation_tasks.append(async_generate_answer(prompt))
    
    answers = await asyncio.gather(*generation_tasks)
    elapsed = time.time() - start_time

    return {
        "execution_time_seconds": round(elapsed, 3),
        "results": [
            {"query": q, "retrieved_context": c, "answer": a}
            for q, c, a in zip(payload.queries, retrieval_results, answers)
        ]
    }