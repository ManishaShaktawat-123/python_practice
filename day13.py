import os
import asyncio
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
import chromadb

load_dotenv()

app = FastAPI(title="Day 13: Query Rewriting & Expansion RAG Pipeline")

# Initialize Gemini & ChromaDB Persistent Storage
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="day13_expanded_rag")


def get_embedding_sync(text: str) -> list:
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values


@app.on_event("startup")
async def startup_event():
    """Startup par ChromaDB collection populate karein agar empty hai"""
    loop = asyncio.get_running_loop()
    
    if collection.count() == 0:
        documents = [
            {
                "id": "doc_1",
                "text": "FastAPI leverages Pydantic for schema validation and Starlette for high-concurrency async request handling."
            },
            {
                "id": "doc_2",
                "text": "Query expansion generates multiple semantic variations of a single query to maximize vector store retrieval recall."
            },
            {
                "id": "doc_3",
                "text": "Gemini 3.5 Flash-Lite delivers ultra-low latency response generation optimized for production RAG systems."
            }
        ]
        
        ids, embeddings, docs_text = [], [], []

        for doc in documents:
            emb = await loop.run_in_executor(None, get_embedding_sync, doc["text"])
            ids.append(doc["id"])
            embeddings.append(emb)
            docs_text.append(doc["text"])

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=docs_text
        )


# Schemas
class ExpansionRAGRequest(BaseModel):
    query: str


class ExpansionRAGResponse(BaseModel):
    original_query: str
    expanded_queries: List[str]
    retrieved_documents: List[str]
    answer: str


def generate_query_variations(query: str) -> List[str]:
    """Generates 3 rewritten variations of the input query for better vector recall"""
    prompt = f"""
Given the search query below, generate 3 alternative ways to ask the same question for information retrieval.
Return ONLY 3 lines, one variation per line. Do not include numbers or bullet points.

Query: {query}
"""
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )
    lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
    return lines[:3]


@app.post("/rag/v6/query-expansion", response_model=ExpansionRAGResponse)
async def query_expansion_rag(request: ExpansionRAGRequest):
    try:
        loop = asyncio.get_running_loop()

        # Step 1: Generate Query Variations via Gemini
        variations = await loop.run_in_executor(None, generate_query_variations, request.query)
        all_queries = [request.query] + variations

        # Step 2: Multi-Vector Retrieval across all generated queries
        all_retrieved_docs = set()
        
        for q in all_queries:
            q_emb = await loop.run_in_executor(None, get_embedding_sync, q)
            
            def search_db(emb):
                return collection.query(query_embeddings=[emb], n_results=1)
                
            res = await loop.run_in_executor(None, search_db, q_emb)
            if res.get("documents") and len(res["documents"][0]) > 0:
                all_retrieved_docs.add(res["documents"][0][0])

        docs_list = list(all_retrieved_docs)
        if not docs_list:
            raise HTTPException(status_code=404, detail="No matching context found.")

        # Step 3: Context Assembly & Answer Generation
        context_str = "\n".join([f"- {doc}" for doc in docs_list])
        
        prompt = f"""
Answer the user's query based ONLY on the context provided below.

Context:
{context_str}

User Query:
{request.query}
"""

        def generate_sync():
            return client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=prompt
            )

        response = await loop.run_in_executor(None, generate_sync)

        return ExpansionRAGResponse(
            original_query=request.query,
            expanded_queries=variations,
            retrieved_documents=docs_list,
            answer=response.text.strip()
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))