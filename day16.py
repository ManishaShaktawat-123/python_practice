import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
import chromadb

load_dotenv()

app = FastAPI(title="Day 16: Dynamic Knowledge Ingestion Pipeline")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="day16_dynamic_ingestion")


def get_embedding_sync(text: str) -> list:
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values


def chunk_text(text: str, chunk_size: int = 150, overlap: int = 30) -> List[str]:
    """Splits text into overlapping chunks for preserving semantic continuity"""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks if chunks else [text]


# Request & Response Schemas
class IngestDocumentRequest(BaseModel):
    document_id: str = Field(..., example="doc_ai_roadmap")
    title: str = Field(..., example="Production AI Engineering")
    raw_content: str = Field(..., example="Building scalable RAG pipelines requires structured chunking...")
    chunk_size: int = Field(default=50, ge=20, le=500)
    overlap: int = Field(default=10, ge=0, le=100)


class IngestDocumentResponse(BaseModel):
    document_id: str
    total_chunks_created: int
    status: str
    sample_chunks: List[str]


class QueryIngestedRequest(BaseModel):
    query: str
    top_k: int = 2


@app.post("/ingest/document", response_model=IngestDocumentResponse)
async def ingest_document(request: IngestDocumentRequest):
    try:
        loop = asyncio.get_running_loop()

        # Step 1: Dynamic Chunking
        chunks = chunk_text(request.raw_content, chunk_size=request.chunk_size, overlap=request.overlap)

        ids = [f"{request.document_id}_chunk_{idx}" for idx in range(len(chunks))]
        metadatas = [
            {
                "document_id": request.document_id,
                "title": request.title,
                "chunk_index": idx
            }
            for idx in range(len(chunks))
        ]

        # Step 2: Batch Embedding Generation
        embeddings = []
        for chunk in chunks:
            emb = await loop.run_in_executor(None, get_embedding_sync, chunk)
            embeddings.append(emb)

        # Step 3: Upsert into ChromaDB
        def upsert_db():
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )

        await loop.run_in_executor(None, upsert_db)

        return IngestDocumentResponse(
            document_id=request.document_id,
            total_chunks_created=len(chunks),
            status="Successfully ingested and vector-indexed",
            sample_chunks=chunks[:2]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/query")
async def query_ingested_knowledge(request: QueryIngestedRequest):
    try:
        loop = asyncio.get_running_loop()

        query_emb = await loop.run_in_executor(None, get_embedding_sync, request.query)

        def search_db():
            return collection.query(
                query_embeddings=[query_emb],
                n_results=request.top_k
            )

        res = await loop.run_in_executor(None, search_db)

        if not res.get("documents") or len(res["documents"][0]) == 0:
            raise HTTPException(status_code=404, detail="No matching ingested context found.")

        return {
            "query": request.query,
            "retrieved_chunks": res["documents"][0],
            "metadata": res["metadatas"][0] if res.get("metadatas") else []
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))