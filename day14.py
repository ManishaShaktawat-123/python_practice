import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
import chromadb

load_dotenv()

app = FastAPI(title="Day 14: Contextual Compression & Window Retrieval")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="day14_window_rag")


def get_embedding_sync(text: str) -> list:
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values


@app.on_event("startup")
async def startup_event():
    """Context window data with surrounding sentence metadata populate karein"""
    loop = asyncio.get_running_loop()
    
    if collection.count() == 0:
        # Document split into sentence-level chunks with window metadata
        passages = [
            {
                "id": "sent_1",
                "text": "FastAPI leverages Pydantic for validation.",
                "window": "Modern web APIs require robust schemas. FastAPI leverages Pydantic for validation. It ensures high runtime performance and strict type checking."
            },
            {
                "id": "sent_2",
                "text": "Sentence-window retrieval isolates key targets.",
                "window": "Standard chunking introduces irrelevant noise. Sentence-window retrieval isolates key targets. Surrounding sentences are appended dynamically during context generation."
            },
            {
                "id": "sent_3",
                "text": "Gemini 3.5 Flash-Lite synthesizes compressed context.",
                "window": "RAG tokens must be kept minimal for low latency. Gemini 3.5 Flash-Lite synthesizes compressed context. This lowers hallucination rates in production systems."
            }
        ]

        ids, embeddings, docs_text, metadatas = [], [], [], []

        for p in passages:
            emb = await loop.run_in_executor(None, get_embedding_sync, p["text"])
            ids.append(p["id"])
            embeddings.append(emb)
            docs_text.append(p["text"])
            metadatas.append({"window_context": p["window"]})

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=docs_text,
            metadatas=metadatas
        )


# Request/Response Schemas
class WindowRAGRequest(BaseModel):
    query: str
    top_k: int = 1


class WindowRAGResponse(BaseModel):
    query: str
    matched_sentence: str
    reconstructed_window: str
    compressed_answer: str


@app.post("/rag/v7/window-retrieval", response_model=WindowRAGResponse)
async def window_retrieval_rag(request: WindowRAGRequest):
    try:
        loop = asyncio.get_running_loop()

        # 1. Embed Query
        query_emb = await loop.run_in_executor(None, get_embedding_sync, request.query)

        # 2. Search Sentence Match
        def search_db():
            return collection.query(
                query_embeddings=[query_emb],
                n_results=request.top_k
            )

        res = await loop.run_in_executor(None, search_db)

        if not res.get("documents") or len(res["documents"][0]) == 0:
            raise HTTPException(status_code=404, detail="No matching sentence found.")

        matched_sentence = res["documents"][0][0]
        reconstructed_window = res["metadatas"][0][0]["window_context"]

        # 3. Context Compression & Answer Generation
        prompt = f"""
Strictly answer the query using only the provided context window. Compress the response to be direct and concise.

Context Window:
{reconstructed_window}

User Query:
{request.query}
"""

        def generate_sync():
            return client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

        response = await loop.run_in_executor(None, generate_sync)

        return WindowRAGResponse(
            query=request.query,
            matched_sentence=matched_sentence,
            reconstructed_window=reconstructed_window,
            compressed_answer=response.text.strip()
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))