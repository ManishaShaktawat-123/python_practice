import os
import asyncio
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
import chromadb

load_dotenv()

app = FastAPI(title="Day 12: Hybrid Search & Context Reranking")

# Initialize Gemini & ChromaDB Persistent Storage
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="day12_hybrid_docs")

def get_embedding_sync(text: str) -> list:
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

@app.on_event("startup")
async def startup_event():
    """Startup par hybrid knowledge base populate karein"""
    loop = asyncio.get_running_loop()
    
    if collection.count() == 0:
        documents = [
            {
                "id": "doc_1",
                "text": "FastAPI uses Pydantic for data validation and Starlette for high-performance async routing.",
                "metadata": {"source": "tech_stack", "code": "FASTAPI_404"}
            },
            {
                "id": "doc_2",
                "text": "ChromaDB stores high-dimensional embeddings and supports vector similarity search via HNSW indexing.",
                "metadata": {"source": "tech_stack", "code": "CHROMA_200"}
            },
            {
                "id": "doc_3",
                "text": "Gemini 3.5 Flash-Lite delivers low-latency generative responses optimized for real-time RAG pipelines.",
                "metadata": {"source": "llm", "code": "GEMINI_35"}
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
class HybridRAGRequest(BaseModel):
    query: str
    top_k: int = 2

class HybridRAGResponse(BaseModel):
    query: str
    selected_context: str
    ranked_candidates: List[Dict[str, Any]]
    answer: str

def keyword_match_score(query: str, text: str) -> float:
    """Basic lexical/keyword matching relevance score"""
    query_tokens = set(query.lower().split())
    text_tokens = set(text.lower().split())
    matches = query_tokens.intersection(text_tokens)
    return len(matches) / max(len(query_tokens), 1)

@app.post("/rag/v5/hybrid-search", response_model=HybridRAGResponse)
async def hybrid_search_rag(request: HybridRAGRequest):
    try:
        loop = asyncio.get_running_loop()

        # 1. Generate Query Vector
        query_emb = await loop.run_in_executor(None, get_embedding_sync, request.query)

        # 2. ChromaDB Semantic Vector Query
        def run_vector_search():
            return collection.query(
                query_embeddings=[query_emb],
                n_results=collection.count()
            )

        vector_res = await loop.run_in_executor(None, run_vector_search)

        if not vector_res.get("documents") or len(vector_res["documents"][0]) == 0:
            raise HTTPException(status_code=404, detail="No documents found in knowledge base.")

        # 3. Hybrid Reranking Logic (Combining Semantic Distance + Keyword Matching)
        candidates = []
        for i in range(len(vector_res["documents"][0])):
            doc_text = vector_res["documents"][0][i]
            dist = vector_res["distances"][0][i] if vector_res.get("distances") else 1.0
            
            # Semantic score inversely proportional to distance
            semantic_score = 1.0 / (1.0 + dist)
            lexical_score = keyword_match_score(request.query, doc_text)
            
            # Weighted Hybrid Score (60% Semantic + 40% Lexical)
            final_score = (0.6 * semantic_score) + (0.4 * lexical_score)
            
            candidates.append({
                "id": vector_res["ids"][0][i],
                "document": doc_text,
                "metadata": vector_res["metadatas"][0][i] if vector_res.get("metadatas") else {},
                "score": round(final_score, 4)
            })

        # Sort candidate context based on highest hybrid score
        candidates.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = candidates[:request.top_k]
        
        context_str = "\n".join([f"- {c['document']}" for c in top_candidates])

        # 4. Generate Answer via Gemini
        prompt = f"""
Answer the user request strictly based on the retrieved context below.

Context:
{context_str}

User Request:
{request.query}
"""

        def generate_sync():
            return client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

        response = await loop.run_in_executor(None, generate_sync)

        return HybridRAGResponse(
            query=request.query,
            selected_context=context_str,
            ranked_candidates=top_candidates,
            answer=response.text.strip()
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))