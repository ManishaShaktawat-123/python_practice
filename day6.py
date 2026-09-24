import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable not found.")

client = genai.Client(api_key=api_key)

app = FastAPI(title="Day 6 - Gemini Structured Output API")

# Define Response Schema using Pydantic
class AnalysisResponse(BaseModel):
    summary: str = Field(description="Brief summary of the input text")
    sentiment: str = Field(description="Sentiment: Positive, Negative, or Neutral")
    key_points: list[str] = Field(description="List of 3 main key points extracted")
    action_items: list[str] = Field(description="List of recommended action items")

class AnalysisRequest(BaseModel):
    text: str = Field(..., example="Our team released the new feature today. Users love it, but two bug reports were filed regarding login latency.")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    try:
        config = types.GenerateContentConfig(
            system_instruction="You are an expert AI Analyst. Analyze the text strictly and extract clear insights.",
            response_mime_type="application/json",
            response_schema=AnalysisResponse,
            temperature=0.2,
        )

        response = client.models.generate_content(
            model="gemini-2.5-pro",  # <--- YAHAN CHANGE KINYA HAI
            contents=request.text,
            config=config,
        )

        result = AnalysisResponse.model_validate_json(response.text)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))