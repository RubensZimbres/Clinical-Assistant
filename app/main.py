"""
Main FastAPI application file.

This script defines the API endpoints for the Clinical Assistant, including:
- A root endpoint to serve the HTML frontend.
- A /chat endpoint to handle user queries and interact with the LangChain agent.
"""
# FIX: Load environment variables at the very beginning of the application startup
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from starlette.responses import FileResponse
from pydantic import BaseModel
from app.agents.frontend_agent import run_validated_query

app = FastAPI(
    title="Clinical Assistant API",
    description="An API for interacting with the Clinical AI Assistant.",
    version="1.0.0",
)

class QueryRequest(BaseModel):
    """Defines the structure of the request body for the /chat endpoint."""
    question: str
    include_reasoning: bool = False

# This endpoint serves your HTML/JS/CSS frontend
@app.get("/")
async def get_index():
    """
    Serves the main index.html file from the 'templates' directory.
    The path is relative to the project's root directory where Uvicorn is run.
    """
    return FileResponse("templates/index.html")

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

