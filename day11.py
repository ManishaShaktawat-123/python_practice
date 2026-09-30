import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
import chromadb

load_dotenv()

app = FastAPI(title="Day 11: ChromaDB RAG with Metadata Filtering")

# 1. Initialize Client & Vector DB Persistent Store
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="day11_rag_docs")

def get_embedding_sync(text: str) -> list:
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

@app.on_event("startup")
async def startup_event():
    """Startup par documents populate karega agar empty hai"""
    loop = asyncio.get_running_loop()
    
    if collection.count() == 0:
        documents = [
            {
                "id": "doc_1",
                "text": "FastAPI provides automatic OpenAPI documentation and high performance using Starlette and Pydantic.",
                "metadata": {"category": "framework", "level": "backend"}
            },
            {
                "id": "doc_2",
                "text": "ChromaDB allows filtering search queries using metadata conditions like category and author.",
                "metadata": {"category": "database", "level": "storage"}
            },
            {
                "id": "doc_3",
                "text": "Gemini 3.5 Flash-Lite excels at fast contextual responses in retrieval augmented pipelines.",
                "metadata": {"category": "llm", "level": "ai"}
            }
        ]
        
        ids, embeddings, docs_text, metadatas = [], [], [], []

        for doc in documents:
            emb = await loop.run_in_executor(None, get_embedding_sync, doc["text"])
            ids.append(doc["id"])
            embeddings.append(emb)
            docs_text.append(doc["text"])
            metadatas.append(doc["metadata"])

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=docs_text,
            metadatas=metadatas
        )

# Request & Response Schemas
class FilteredRAGRequest(BaseModel):
    query: str
    category_filter: Optional[str] = None  # Example: "database", "framework", "llm"

class FilteredRAGResponse(BaseModel):
    query: str
    category_filter_used: Optional[str]
    retrieved_context: str
    answer: str

@app.post("/rag/v4/ask", response_model=FilteredRAGResponse)
async def ask_with_metadata_filter(request: FilteredRAGRequest):
    try:
        loop = asyncio.get_running_loop()

        # 1. Embed Query
        query_emb = await loop.run_in_executor(None, get_embedding_sync, request.query)

        # 2. Build Metadata Filter
        where_clause = {"category": request.category_filter} if request.category_filter else None

        # 3. Search ChromaDB
        def query_db():
            return collection.query(
                query_embeddings=[query_emb],
                n_results=1,
                where=where_clause
            )

        results = await loop.run_in_executor(None, query_db)

        # Handle case where no documents match the filter
        if not results["documents"] or len(results["documents"][0]) == 0:
            raise HTTPException(
                status_code=404,
                detail=f"No matching documents found for filter: {request.category_filter}"
            )

        retrieved_text = results["documents"][0][0]

        # 4. Generate Answer via Gemini
        prompt = f"""
Answer the question strictly using ONLY the context provided below.

Context:
"{retrieved_text}"

Question:
{request.query}
"""

        def generate_sync():
            return client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

        response = await loop.run_in_executor(None, generate_sync)

        return FilteredRAGResponse(
            query=request.query,
            category_filter_used=request.category_filter,
            retrieved_context=retrieved_text,
            answer=response.text.strip()
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))