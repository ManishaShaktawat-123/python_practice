import os
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai

load_dotenv()

app = FastAPI(title="Day 8: Full RAG Pipeline Service")

# Initialize Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Knowledge Base
knowledge_base = [
    "FastAPI is a modern, fast web framework for building APIs with Python.",
    "Gemini 2.5 Flash is Google's lightweight and efficient multimodal AI model.",
    "Retrieval-Augmented Generation (RAG) improves LLM responses using external context.",
    "Python is a high-level programming language widely used in AI and Web Development."
]

def get_embedding(text: str) -> list:
    """Generates vector embeddings using gemini-embedding-001."""
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Pre-compute document embeddings at startup
doc_embeddings = [get_embedding(doc) for doc in knowledge_base]

class QueryRequest(BaseModel):
    query: str

class RAGResponse(BaseModel):
    query: str
    retrieved_context: str
    similarity_score: float
    answer: str

@app.post("/rag/ask", response_model=RAGResponse)
async def ask_rag(request: QueryRequest):
    try:
        # 1. Embed user query
        query_emb = get_embedding(request.query)

        # 2. Vector search
        similarities = [cosine_similarity(query_emb, doc_emb) for doc_emb in doc_embeddings]
        best_idx = int(np.argmax(similarities))
        retrieved_context = knowledge_base[best_idx]
        best_score = float(similarities[best_idx])

        # 3. Prompt Construction
        prompt = f"""
You are a helpful AI Assistant. Answer the user question strictly based ONLY on the provided context below.

Context:
"{retrieved_context}"

Question:
{request.query}
"""

        # 4. Async Generation Call (Prevents FastAPI Network Disconnect / Event Loop Blocking)
        response = await client.aio.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return RAGResponse(
            query=request.query,
            retrieved_context=retrieved_context,
            similarity_score=round(best_score, 4),
            answer=response.text
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))