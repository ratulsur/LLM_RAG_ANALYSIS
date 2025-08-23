import fitz  # PyMuPDF 
import os
import uuid
from datetime import datetime
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from pathlib import Path
from io import BytesIO


class DocumentHandler:
    """
    Helps to upload PDF and perform reading operations.
    Logs all actions and supports session-based operations.
    
    """
    def __init__(self, data_dir=None, session_id=None):
        try:
            self.log = CustomLogger.get_logger(__name__)

            self.data_dir = data_dir or os.getenv(
                "DATA_STORAGE_PATH",
                os.path.join(os.getcwd(), "data", "document_analysis")
            )
            self.session_id = session_id or f"session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
            self.session_id_path = os.path.join(self.data_dir, self.session_id)
            os.makedirs(self.session_id_path, exist_ok=True)

            self.log.info(f"PDFHandler initialized | session_id={self.session_id} | session_path={self.session_id_path}")

        except Exception as e:
            self.log.error(f"Error initializing DocumentHandler: {e}", exc_info=True)
            raise DocumentPortalException("Error initializing DocumentHandler", e) from e

    def save_pdf(self, uploaded_file):
        """
        Saves an uploaded PDF file to the session directory.
        """
        try:
            filename = os.path.basename(uploaded_file.name)
            if not filename.lower().endswith(".pdf"):
                raise DocumentPortalException("Invalid file type", ValueError("Only PDF files are allowed"))
            
            save_path = os.path.join(self.session_id_path, filename)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            self.log.info(f"PDF saved | file={filename} | path={save_path} | session_id={self.session_id}")
            return save_path

        except Exception as e:
            self.log.error(f"Error saving PDF: {e}", exc_info=True)
            raise DocumentPortalException("Error saving PDF", e) from e

    def read_pdf(self, pdf_path):
        """
        Reads the content of a PDF file using PyMuPDF and returns the text.
        """
        try:
            if not os.path.exists(pdf_path):
                raise DocumentPortalException("PDF file not found", FileNotFoundError(pdf_path))

            text = ""
            with fitz.Document(pdf_path) as doc:
                for page in doc:
                    text += page.get_text()

            self.log.info(f"PDF read | path={pdf_path} | session_id={self.session_id}")
            return text

        except Exception as e:
            self.log.error(f"Error reading PDF: {e}", exc_info=True)
            raise DocumentPortalException("Error reading PDF", e) from e

if __name__ == "__main__":
    try:
        handler = DocumentHandler()
        print(f"Session ID: {handler.session_id}")
        print(f"Session Path: {handler.session_id_path}")
    except DocumentPortalException as e:
        print("Exception occurred:")
        print(str(e))



