from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, List, Any, Dict
from pathlib import Path
import os

from src.document_ingestion.data_ingestion import (
    DocumentComparator,
    DocHandler,
    ChatIngestor,
    FaissManager,
)
from utils.document_ops import FastAPIFileAdapter
from src.doc_analyzer.data_analysis import DocumentAnalyzer
from src.doc_compare.document_comparer import DocumentComparerLLM
from src.document_chat.retrieval import ConversationalRAG

app = FastAPI(title="Document Portal API", version="0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Paths / constants (DEFINE THESE) ---
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
UPLOAD_BASE = str(PROJECT_ROOT / "data" / "document_chat" / "uploads")  # <-- define
FAISS_BASE = str(PROJECT_ROOT / "faiss_index")                           # <-- define

app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")

# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-portal"}

# Helper: wrap DocHandler.read_pdf with proper error binding
def _read_pdf_via_handler(handler: DocHandler, path: str) -> str:
    try:
        return handler.read_pdf(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling PDF: {str(e)}")

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
    try:
        dh = DocHandler()
        saved_path = dh.save_p
    except Exception as e:
        raise HTTPException (status_code=500, detail=f"Query failed: {e}")
    
    
