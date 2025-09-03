from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional, List, Any, Dict
from pathlib import Path
import os
import json, traceback
from logger import GLOBAL_LOGGER as log
from typing import Any, Dict
from fastapi import HTTPException, UploadFile, File
from utils.document_ops import FastAPIFileAdapter
from src.document_ingestion.data_ingestion import DocHandler
from src.doc_analyzer.data_analysis import DocumentAnalyzer

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

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import traceback
from logger import GLOBAL_LOGGER as log
# ... your other imports ...

def _read_pdf_via_handler(handler: DocHandler, path: str) -> str:
    try:
        return handler.read_pdf(path)
    except Exception as e:
        log.error("read_pdf failed", error=str(e), traceback=traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error handling PDF: {e}")

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)) -> Dict[str, Any]:
    # 1) Save + read PDF
    try:
        dh = DocHandler()
        saved_path = dh.save_pdf(FastAPIFileAdapter(file))
        log.info(f"[analyze] saved to {saved_path}")
        text = dh.read_pdf(saved_path)
        log.info(f"[analyze] read {len(text)} chars")
    except Exception as e:
        log.error(f"[analyze] save/read failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"PDF processing failed: {e}")

    # 2) Run analysis
    try:
        analyzer = DocumentAnalyzer(config={})
        result = analyzer.analyze_document(text)
        log.info("[analyze] analyzer finished")
    except Exception as e:
        log.error(f"[analyze] analyzer failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analyzer failed: {e}")

    # 3) Make the result JSON-safe
    try:
        if hasattr(result, "model_dump"):            # pydantic v2
            result = result.model_dump()
        elif hasattr(result, "dict"):                # pydantic v1
            result = result.dict()
        elif hasattr(result, "to_dict"):             # pandas/other
            result = result.to_dict()
        elif hasattr(result, "to_json"):             # returns JSON string
            result = json.loads(result.to_json())
        elif type(result).__name__ in ("DataFrame", "Series"):
            try:
                import pandas as pd
                if isinstance(result, pd.DataFrame):
                    result = result.to_dict(orient="records")
                else:
                    result = str(result)
            except Exception:
                result = str(result)
    except Exception as e:
        log.error(f"[analyze] serialization failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Serialization failed: {e}")

    # 4) Return
    return {
        "filename": file.filename,
        "chars": len(text),
        "analysis": result,
    }

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
