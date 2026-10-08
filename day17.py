import os
import time
import uuid
import logging
import json
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google import genai
import chromadb

load_dotenv()

# Setup Structured JSON Logging
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module
        }
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "duration_ms"):
            log_data["duration_ms"] = record.duration_ms
        return json.dumps(log_data)

logger = logging.getLogger("rag_observability")
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

app = FastAPI(title="Day 17: Production Observability & Tracing RAG Pipeline")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="day17_observable_rag")


# Middleware for Request Correlation & Execution Timing
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id
    start_time = time.time()

    response = await call_next(request)

    process_time = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-MS"] = str(process_time)

    logger.info(
        f"Path: {request.url.path} | Status: {response.status_code}",
        extra={"request_id": request_id, "duration_ms": process_time}
    )
    return response


def get_embedding_sync(text: str) -> list:
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values


@app.on_event("startup")
async def startup_event():
    """Populate baseline data if empty"""
    loop = asyncio.get_running_loop()
    if collection.count() == 0:
        documents = [
            {"id": "doc_1", "text": "Structured logging in JSON allows centralized log aggregation in Elasticsearch or CloudWatch."},
            {"id": "doc_2", "text": "Request correlation IDs enable end-to-end tracing across distributed microservices."}
        ]
        ids, embeddings, docs_text = [], [], []
        for doc in documents:
            emb = await loop.run_in_executor(None, get_embedding_sync, doc["text"])
            ids.append(doc["id"])
            embeddings.append(emb)
            docs_text.append(doc["text"])

        collection.add(ids=ids, embeddings=embeddings, documents=docs_text)


class ObservedRAGRequest(BaseModel):
    query: str


@app.post("/rag/v9/observed-search")
async def observed_rag_search(req: ObservedRAGRequest, request: Request):
    req_id = getattr(request.state, "request_id", "unknown")
    try:
        loop = asyncio.get_running_loop()

        # Step 1: Retrieval Timing
        t0 = time.time()
        query_emb = await loop.run_in_executor(None, get_embedding_sync, req.query)
        
        def search_db():
            return collection.query(query_embeddings=[query_emb], n_results=1)

        res = await loop.run_in_executor(None, search_db)
        retrieval_ms = round((time.time() - t0) * 1000, 2)

        if not res.get("documents") or len(res["documents"][0]) == 0:
            raise HTTPException(status_code=404, detail="No relevant context found.")

        context = res["documents"][0][0]

        # Step 2: Generation Timing
        t1 = time.time()
        prompt = f"Answer strictly using this context:\n{context}\n\nQuery: {req.query}"

        def generate_sync():
            return client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

        gen_res = await loop.run_in_executor(None, generate_sync)
        generation_ms = round((time.time() - t1) * 1000, 2)

        return {
            "request_id": req_id,
            "query": req.query,
            "answer": gen_res.text.strip(),
            "performance_metrics": {
                "retrieval_latency_ms": retrieval_ms,
                "generation_latency_ms": generation_ms,
                "total_pipeline_ms": round(retrieval_ms + generation_ms, 2)
            }
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Execution Error: {str(e)}", extra={"request_id": req_id})
        raise HTTPException(status_code=500, detail=str(e))