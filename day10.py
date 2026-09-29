import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
import chromadb

load_dotenv()

app = FastAPI(title="Day 10: ChromaDB Vector DB Integration")

# 1. Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# 2. Initialize ChromaDB Persistent Storage
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="rag_documents")

def get_embedding_sync(text: str) -> list:
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

@app.on_event("startup")
async def startup_event():
    """Startup par ChromaDB collection populate karega agar empty hai"""
    loop = asyncio.get_running_loop()
    
    if collection.count() == 0:
        documents = [
            {
                "id": "doc_1",
                "text": "FastAPI is a modern, high-performance web framework for building APIs with Python 3.8+ based on standard Python type hints.",
                "metadata": {"category": "web_framework", "author": "tiangolo"}
            },
            {
                "id": "doc_2",
                "text": "ChromaDB is an open-source AI application database designed for developer productivity and embedding retrieval.",
                "metadata": {"category": "vector_db", "type": "open_source"}
            },
            {
                "id": "doc_3",
                "text": "Gemini 3.5 Flash-Lite provides high-speed, cost-efficient generation for real-time RAG pipelines.",
                "metadata": {"category": "llm", "vendor": "google"}
            }
        ]
        
        ids = []
        embeddings = []
        docs_text = []
        metadatas = []

        for doc in documents:
            emb = await loop.run_in_executor(None, get_embedding_sync, doc["text"])
            ids.append(doc["id"])
            embeddings.append(emb)
            docs_text.append(doc["text"])
            metadatas.append(doc["metadata"])

        # Collection me vectors, payload, aur metadata store kar rahe hain
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=docs_text,
            metadatas=metadatas
        )

# Request & Response Schemas
class VectorSearchRequest(BaseModel):
    query: str
    n_results: int = 2

class VectorSearchResponse(BaseModel):
    query: str
    retrieved_docs: List[Dict[str, Any]]

@app.post("/rag/v3/search", response_model=VectorSearchResponse)
async def search_vector_db(request: VectorSearchRequest):
    try:
        loop = asyncio.get_running_loop()

        # 1. Query Embedding generate karna
        query_emb = await loop.run_in_executor(None, get_embedding_sync, request.query)

        # 2. ChromaDB Vector Query Executer
        def run_chroma_query():
            return collection.query(
                query_embeddings=[query_emb],
                n_results=request.n_results
            )

        results = await loop.run_in_executor(None, run_chroma_query)

        # 3. Format Response
        formatted_docs = []
        if results and results.get("documents") and len(results["documents"][0]) > 0:
            for i in range(len(results["documents"][0])):
                formatted_docs.append({
                    "id": results["ids"][0][i],
                    "document": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": round(float(results["distances"][0][i]), 4) if results.get("distances") else None
                })

        return VectorSearchResponse(
            query=request.query,
            retrieved_docs=formatted_docs
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))