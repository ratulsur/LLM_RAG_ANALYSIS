from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from typing import Any, Dict
from pathlib import Path  
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from src.document_ingestion.data_ingestion import (
    DocumentComparator,
    DocHandler,
    ChatIngestor,
    FaissManager
)
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

# --- Paths ---
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  

app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "static"), name="static")
templates = Jinja2Templates(directory=PROJECT_ROOT / "templates")


# --- Routes ---
@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "document-portal"}


def _read_pdf_via_handler(handler: DocHandler, path:str) ->str:
     try:
          pass
     except:
          raise HTTPException(status_code=500, detail = f"Error handling PDF : {str{e}}")
@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Any:
        try:
            dh = DocHandler()
            saved_path = dh.save_pdf(FastAPIFileAdapter(file))
            text = _read_pdf_via_handler(dh, saved_path)
            analyzer = DocumentAnalyzer()
            analyzer.analyze_document(text)
            return JSONResponse(content=result)
        

        except Exception as e:  
            raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/compare")
async def compare_documents(reference: UploadFile = File(...)) -> Any:
    try:
        dc = DocumentComparator()
        ref_path, actual_path = dc.save_uploaded_files(FastAPIFileAdapter(reference), FastAPIFileAdapter(actual))
        _=ref_path, actual_path
        combined_text = dc.combine_documents()
        comp = DocumentComparerLLM()
        df = comp.compare_documents(combined_text)
        return {"rows":df.to_dict(orient="records"), "session_id": dc.session_id}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"comparison failed: {e}")

@app.post("/chat/index")
async def chat_build_index() -> Any:
    try:
        
        return {"ok": True, "detail": "index stub"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"indexing failed: {e}")

@app.post("/chat/query")
async def chat_query() -> Any:
    try:
        
        return {"ok": True, "detail": "query stub"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
