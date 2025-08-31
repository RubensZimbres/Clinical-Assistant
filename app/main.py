"""
Main FastAPI application file.

This script defines the API endpoints for the Clinical Assistant, including:
- A root endpoint to serve the HTML frontend.
- A /chat endpoint to handle user queries and interact with the LangChain agent.
"""
# NOTE: dotenv is for local development. In Cloud Run, you must set these
# variables during deployment using the --set-env-vars flag.
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from starlette.responses import FileResponse
from pydantic import BaseModel
from app.agents.frontend_agent import run_validated_query
from pathlib import Path # <-- FIX: Import pathlib for robust path handling

app = FastAPI(
    title="Clinical Assistant API",
    description="An API for interacting with the Clinical AI Assistant.",
    version="1.0.0",
)

# <-- FIX: Define the project's root directory inside the container
APP_DIR = Path(__file__).resolve().parent

class QueryRequest(BaseModel):
    """Defines the structure of the request body for the /chat endpoint."""
    question: str
    include_reasoning: bool = False

# This endpoint serves your HTML/JS/CSS frontend
@app.get("/")
async def get_index():
    """
    Serves the main index.html file from the 'templates' directory.
    """
    template_path = APP_DIR / "templates" / "index.html"
    return FileResponse(template_path)

@app.post("/chat")
async def chat(request: QueryRequest):
    """
    Receives a question from the user, processes it through the agent,
    and returns the agent's structured response.
    """
    # Use the validated query runner from the frontend agent for robustness
    response = run_validated_query(
        question=request.question,
        include_reasoning=request.include_reasoning
    )
    return response