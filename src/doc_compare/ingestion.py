import os
import sys
import shutil
from pathlib import Path
from datetime import datetime
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException


class DocumentIngestion:
    def __init__(self):
        self.log = CustomLogger.get_logger(__name__)

        # ✅ create a unique session folder each run
        session_name = f"session_{datetime.now():%Y%m%d_%H%M%S}"
        self.base_dir = Path(
            "/Users/ratulsur/Desktop/all_data/document_portal/data/document_compare"
        ) / session_name

        os.makedirs(self.base_dir, exist_ok=True)
        self.log.info("New session directory created: %s", self.base_dir)

    def delete_existing_file(self):
        try:
            for f in self.base_dir.glob("*.pdf"):
                f.unlink()
            self.log.info("All existing PDF files deleted | directory=%s", self.base_dir)
        except Exception as e:
            self.log.error("Error deleting PDFs in %s: %s", self.base_dir, e)
            raise DocumentPortalException("Error deleting existing PDFs", sys) from e

    def save_uploaded_files(self, ref_upload, act_upload):
        try:
            self.delete_existing_file()

            ref_file = self.base_dir / ref_upload.name
            act_file = self.base_dir / act_upload.name

            with open(ref_file, "wb") as f:
                f.write(ref_upload.getbuffer())
            with open(act_file, "wb") as f:
                f.write(act_upload.getbuffer())

            self.log.info(
                "Files saved successfully | ref_file=%s | act_file=%s",
                ref_file,
                act_file,
            )
            return ref_file, act_file

        except Exception as e:
            self.log.error("Error saving files: %s", e)
            raise DocumentPortalException("Error saving uploaded files", sys) from e

    def combine_documents(self):
        try:
            combined_text = ""
            for pdf_file in self.base_dir.glob("*.pdf"):
                with open(pdf_file, "rb") as f:
                    text = f.read().decode("latin-1", errors="ignore")
                    combined_text += f"\nDocument:{pdf_file.name}\n\n{text}\n"
            self.log.info(
                "Documents combined successfully | length=%s", len(combined_text)
            )
            return combined_text

        except Exception as e:
            self.log.error("Error combining documents: %s", e)
            raise DocumentPortalException("Error combining documents", sys) from e

    def clean_old_sessions(self, keep_latest=3):
        try:
            sessions = sorted(
                [p for p in self.base_dir.parent.glob("session*") if p.is_dir()],
                key=os.path.getmtime,
                reverse=True,
            )
            for old in sessions[keep_latest:]:
                shutil.rmtree(old, ignore_errors=True)
            self.log.info(
                "Old sessions cleaned up | kept latest=%s | total_removed=%s",
                keep_latest,
                max(0, len(sessions) - keep_latest),
            )
        except Exception as e:
            self.log.error("Error cleaning old sessions: %s", e)
            raise DocumentPortalException("Error cleaning old sessions", sys) from e
