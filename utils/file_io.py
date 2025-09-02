from __future__ import annotations
import re
import uuid
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Iterable, List
from logger import GLOBAL_LOGGER as log
from exception.custom_exception import DocumentPortalException

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def generate_session_id(prefix: str = "session") -> str:
    """Generate a timestamped unique session id (IST timezone)."""
    ist = ZoneInfo("Asia/Kolkata")
    return f"{prefix}_{datetime.now(ist).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def save_uploaded_files(uploaded_files: Iterable, target_dir: Path) -> List[Path]:
    """
    Save uploaded files (FastAPI UploadFile, Streamlit, or file-like) and return local paths.
    """
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []

        for uf in uploaded_files:
            # Try both FastAPI (filename) and others (name)
            original_name = getattr(uf, "filename", None) or getattr(uf, "name", None) or "file"
            ext = Path(original_name).suffix.lower()

            if ext not in SUPPORTED_EXTENSIONS:
                log.warning("Unsupported file skipped", filename=original_name)
                continue

            # Safe + unique file name
            safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', Path(original_name).stem).lower()
            fname = f"{safe_name}_{uuid.uuid4().hex[:6]}{ext}"
            out_path = target_dir / fname

            with open(out_path, "wb") as f:
                if hasattr(uf, "read"):
                    f.write(uf.read())
                else:
                    f.write(uf.getbuffer())  # fallback for BytesIO-like

            saved.append(out_path)
            log.info("File saved for ingestion", uploaded=original_name, saved_as=str(out_path))

        return saved

    except Exception as e:
        log.error("Failed to save uploaded files", error=str(e), dir=str(target_dir))
        raise DocumentPortalException("Failed to save uploaded files", e) from e
