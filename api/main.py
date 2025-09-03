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
UPLOAD_BASE = str(PROJECT_ROOT / "data" / "document_chat" / "uploads")  
FAISS_BASE = str(PROJECT_ROOT / "faiss_index")                           

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
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        text = _read_pdf_via_handler(dh, saved_path)

        analyzer = DocumentAnalyzer()
        result = analyzer.analyze_document(text)  

        
        return JSONResponse(content=result)
    except Exception as e:  
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

@app.post("/compare")
async def compare_documents(
    reference: UploadFile = File(...),
    actual: UploadFile = File(...),        
) -> Any:
    try:
        dc = DocumentComparator()
        ref_path, actual_path = dc.save_uploaded_files(
            FastAPIFileAdapter(reference), FastAPIFileAdapter(actual)
        )
        _ = ref_path, actual_path  

        combined_text = dc.combine_documents()
        comp = DocumentComparerLLM()
        df = comp.compare_documents(combined_text)

        return {"rows": df.to_dict(orient="records"), "session_id": dc.session_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"comparison failed: {e}")

@app.post("/chat/index")
async def chat_build_index(
    files: List[UploadFile] = File(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(200),
    k: int = Form(5),
) -> Any:
    try:
        wrapped = [FastAPIFileAdapter(f) for f in files]
        ci = ChatIngestor(
            temp_base=UPLOAD_BASE,         
            faiss_base=FAISS_BASE,         
            use_session_dirs=use_session_dirs,
            session_id=session_id or None,
        )
        # keep your original method name if that's what you implemented
        ci.built_retriver(wrapped, chunk_size=chunk_size, chunk_overlap=chunk_overlap, k=k)
        return {"ok": True, "detail": "index built"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"indexing failed: {e}")

@app.post("/chat/query")
async def chat_query(
    question: str = Form(...),
    session_id: Optional[str] = Form(None),
    use_session_dirs: bool = Form(True),
    k: int = Form(5),
) -> Any:
    try:
        if use_session_dirs and not session_id:
            raise HTTPException(status_code=400, detail="session_id is required when use_session_dir = True")

        index_dir = os.path.join(FAISS_BASE, session_id) if use_session_dirs else FAISS_BASE
        if not os.path.isdir(index_dir):
            raise HTTPException(status_code=404, detail=f"FAISS index not found at: {index_dir}")

        rag = ConversationalRAG(session_id=session_id)
        rag.load_retriever_from_faiss(index_dir)
        response = rag.invoke(question, chat_history=[])

        return {"answer": response, "session_id": session_id, "k": k, "engine": "LCEL-RAG"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
