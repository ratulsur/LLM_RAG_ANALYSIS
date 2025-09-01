from __future__ import annotations
import os, sys, json, uuid, hashlib, shutil
from pathlib import Path
from typing import Iterable, List, Optional, Dict, Any
import fitz  

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from utils.model_loader import ModelLoader
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# ---------------- Helper functions ----------------
def generate_session_id(prefix: str = "session") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def save_uploaded_files(uploaded_files: Iterable, out_dir: Path) -> List[Path]:
    out_paths = []
    for f in uploaded_files:
        fname = getattr(f, "name", f"upload_{uuid.uuid4().hex[:8]}")
        dest = out_dir / fname
        with open(dest, "wb") as fh:
            if hasattr(f, "read"):
                fh.write(f.read())
            else:
                fh.write(f.getbuffer())
        out_paths.append(dest)
    return out_paths


def load_documents(paths: List[Path]) -> List[Document]:
    docs = []
    for p in paths:
        if p.suffix.lower() == ".pdf":
            with fitz.open(str(p)) as pdf:
                for i, page in enumerate(pdf):
                    text = page.get_text()
                    if text.strip():
                        docs.append(Document(page_content=text, metadata={"source": str(p), "page": i + 1}))
        elif p.suffix.lower() in {".txt", ".md"}:
            text = p.read_text(encoding="utf-8")
            docs.append(Document(page_content=text, metadata={"source": str(p)}))
        else:
            log.warning(f"Unsupported file skipped: {p}")
    return docs


# ---------------- FAISS Manager ----------------
class FaissManager:
    def __init__(self, index_dir: Path, model_loader: Optional[ModelLoader] = None):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.index_dir / "ingested_meta.json"
        self._meta: Dict[str, Any] = {"rows": {}}

        if self.meta_path.exists():
            try:
                self._meta = json.loads(self.meta_path.read_text(encoding="utf-8")) or {"rows": {}}
            except Exception:
                self._meta = {"rows": {}}

        self.model_loader = model_loader or ModelLoader()
        self.emb = self.model_loader.load_embeddings()
        self.vs: Optional[FAISS] = None

    def _exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists()

    def _fingerprint(self, text: str, md: Dict[str, Any]) -> str:
        src = md.get("source")
        rid = md.get("row_id")
        if src is not None:
            return f"{src}::{'' if rid is None else rid}"
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _save_meta(self):
        self.meta_path.write_text(json.dumps(self._meta, indent=2), encoding="utf-8")

    def add_documents(self, docs: List[Document]):
        if self.vs is None:
            raise RuntimeError("Call load_or_create() first")

        new_docs = []
        for d in docs:
            key = self._fingerprint(d.page_content, d.metadata or {})
            if key not in self._meta["rows"]:
                self._meta["rows"][key] = True
                new_docs.append(d)

        if new_docs:
            self.vs.add_documents(new_docs)
            self.vs.save_local(str(self.index_dir))
            self._save_meta()
        return len(new_docs)

    def load_or_create(self, texts: Optional[List[str]] = None, metadatas: Optional[List[dict]] = None):
        if self._exists():
            self.vs = FAISS.load_local(
                str(self.index_dir),
                embeddings=self.emb,
                allow_dangerous_deserialization=True,
            )
            return self.vs
        if not texts:
            raise DocumentPortalException("No FAISS index and no texts provided", sys)
        self.vs = FAISS.from_texts(texts=texts, embedding=self.emb, metadatas=metadatas or [])
        self.vs.save_local(str(self.index_dir))
        return self.vs


# ---------------- Chat Ingestor ----------------
class ChatIngestor:
    def __init__(self, temp_base="data", faiss_base="faiss_index", use_session_dirs=True, session_id=None):
        self.model_loader = ModelLoader()
        self.use_session = use_session_dirs
        self.session_id = session_id or generate_session_id()

        self.temp_base = Path(temp_base); self.temp_base.mkdir(parents=True, exist_ok=True)
        self.faiss_base = Path(faiss_base); self.faiss_base.mkdir(parents=True, exist_ok=True)

        self.temp_dir = self._resolve_dir(self.temp_base)
        self.faiss_dir = self._resolve_dir(self.faiss_base)

        log.info(f"ChatIngestor initialized | session_id={self.session_id} | temp_dir={self.temp_dir} | faiss_dir={self.faiss_dir}")

    def _resolve_dir(self, base: Path):
        if self.use_session:
            d = base / self.session_id
            d.mkdir(parents=True, exist_ok=True)
            return d
        return base

    def _split(self, docs: List[Document], chunk_size=1000, chunk_overlap=200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(docs)
        log.info(f"Documents split | chunks={len(chunks)} | chunk_size={chunk_size} | overlap={chunk_overlap}")
        return chunks

    def build_retriever(self, uploaded_files: Iterable, chunk_size=1000, chunk_overlap=200, k=5):
        try:
            paths = save_uploaded_files(uploaded_files, self.temp_dir)
            docs = load_documents(paths)
            if not docs:
                raise ValueError("No valid documents loaded")

            chunks = self._split(docs, chunk_size, chunk_overlap)

            fm = FaissManager(self.faiss_dir, self.model_loader)
            texts = [c.page_content for c in chunks]
            metas = [c.metadata for c in chunks]

            vs = fm.load_or_create(texts=texts, metadatas=metas)
            added = fm.add_documents(chunks)

            log.info(f"FAISS index updated | added={added} | index={self.faiss_dir}")
            return vs.as_retriever(search_type="similarity", search_kwargs={"k": k})

        except Exception as e:
            log.error(f"Failed to build retriever | error={e}")
            raise DocumentPortalException("Failed to build retriever", e) from e
