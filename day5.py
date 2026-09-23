import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    # Key set na hone par error avoid karne ke liye check
    print("Warning: GEMINI_API_KEY environment variable not set!")

genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="AI Integration API - Day 5 (Gemini)")

# Request schema using Pydantic
class PromptRequest(BaseModel):
    prompt: str

@app.get("/")
def home():
    return {"message": "Gemini AI API Server Active"}

@app.post("/generate")
async def generate_text(request: PromptRequest):
    try:
        # Initialize Gemini Model
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Generate content asynchronously
        response = model.generate_content(request.prompt)
        
        return {
            "status": "success",
            "prompt": request.prompt,
            "response": response.text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))