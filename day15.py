import os
import asyncio
import json
from typing import List, Dict, Any
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
import chromadb

load_dotenv()

app = FastAPI(title="Day 15: Automated RAG Evaluation & Triad Metrics")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="day15_eval_docs")


def get_embedding_sync(text: str) -> list:
    response = client.models.embed_content(
        model="models/gemini-embedding-001",
        contents=text
    )
    return response.embeddings[0].values


@app.on_event("startup")
async def startup_event():
    """Knowledge base setup for evaluation testing"""
    loop = asyncio.get_running_loop()
    
    if collection.count() == 0:
        documents = [
            {
                "id": "eval_doc_1",
                "text": "RAG Triad evaluates RAG pipelines across three axes: Context Relevance, Faithfulness (Groundedness), and Answer Relevance."
            },
            {
                "id": "eval_doc_2",
                "text": "Faithfulness checks if the generated answer contains information that is strictly derived from the retrieved context without hallucination."
            },
            {
                "id": "eval_doc_3",
                "text": "Answer Relevance calculates how well the generated response directly answers the user original prompt."
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


# Request & Response Schemas
class EvalRAGRequest(BaseModel):
    query: str


class EvaluationScores(BaseModel):
    context_relevance: float
    groundedness: float
    answer_relevance: float
    overall_score: float


class EvalRAGResponse(BaseModel):
    query: str
    retrieved_context: str
    generated_answer: str
    evaluations: EvaluationScores


def evaluate_rag_triad(query: str, context: str, answer: str) -> Dict[str, float]:
    """LLM-as-a-Judge approach to score the RAG Triad"""
    eval_prompt = f"""
You are an AI Evaluation Judge. Rate the following RAG output on a scale from 0.0 to 1.0 for three metrics:

1. context_relevance: Is the context useful for answering the query?
2. groundedness: Is the answer derived ONLY from the context without assumptions or hallucinations?
3. answer_relevance: Does the answer directly and completely address the query?

Respond strictly in valid JSON format with keys: "context_relevance", "groundedness", "answer_relevance".
Do not add markdown backticks or extra text outside JSON.

Query: {query}
Context: {context}
Answer: {answer}
"""
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=eval_prompt
    )
    
    clean_text = response.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(clean_text)


@app.post("/rag/v8/evaluate", response_model=EvalRAGResponse)
async def evaluate_rag_pipeline(request: EvalRAGRequest):
    try:
        loop = asyncio.get_running_loop()

        # Step 1: Retrieval
        query_emb = await loop.run_in_executor(None, get_embedding_sync, request.query)

        def search_db():
            return collection.query(
                query_embeddings=[query_emb],
                n_results=2
            )

        res = await loop.run_in_executor(None, search_db)

        if not res.get("documents") or len(res["documents"][0]) == 0:
            raise HTTPException(status_code=404, detail="No matching documents found.")

        retrieved_docs = res["documents"][0]
        context_str = "\n".join([f"- {d}" for d in retrieved_docs])

        # Step 2: Generation
        generation_prompt = f"""
Answer the user's query strictly based on the context below.

Context:
{context_str}

Query:
{request.query}
"""

        def generate_sync():
            return client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=generation_prompt
            )

        gen_response = await loop.run_in_executor(None, generate_sync)
        answer_str = gen_response.text.strip()

        # Step 3: LLM-as-a-Judge Evaluation (Triad Scoring)
        scores = await loop.run_in_executor(
            None, evaluate_rag_triad, request.query, context_str, answer_str
        )

        c_rel = float(scores.get("context_relevance", 0.0))
        g_val = float(scores.get("groundedness", 0.0))
        a_rel = float(scores.get("answer_relevance", 0.0))
        overall = round((c_rel + g_val + a_rel) / 3.0, 2)

        return EvalRAGResponse(
            query=request.query,
            retrieved_context=context_str,
            generated_answer=answer_str,
            evaluations=EvaluationScores(
                context_relevance=c_rel,
                groundedness=g_val,
                answer_relevance=a_rel,
                overall_score=overall
            )
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))