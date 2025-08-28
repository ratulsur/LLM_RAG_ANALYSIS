import os
import sys
from pathlib import Path
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
import fitz  

class DocumentIngestion:
    def __init__(self, base_dir: str = "/Users/ratulsur/Desktop/all_data/document_portal/data/document_compare"):
        self.log = CustomLogger.get_logger(__name__)
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def delete_existing_file(self):
        try:
            for file in self.base_dir.glob("*.pdf"):
                file.unlink()
            self.log.info("All existing PDF files deleted | directory=%s", self.base_dir)

        except Exception as e:
            self.log.error("Error deleting files in %s | error=%s", self.base_dir, e)
            raise DocumentPortalException("Error deleting existing PDFs", sys) from e

    def save_uploaded_files(self, reference_file, actual_file):
        try:
            self.delete_existing_file()

            ref_path = self.base_dir / reference_file.name
            act_path = self.base_dir / actual_file.name

            if not reference_file.name.endswith(".pdf") or not actual_file.name.endswith(".pdf"):
                raise ValueError("Only PDF files are allowed")

            with open(ref_path, "wb") as f:
                f.write(reference_file.getbuffer())

            with open(act_path, "wb") as f:
                f.write(actual_file.getbuffer())

            self.log.info("Files saved successfully | reference=%s | actual=%s", ref_path, act_path)
            return ref_path, act_path

        except Exception as e:
            self.log.error("Error saving files: %s", e)
            raise DocumentPortalException("Error saving uploaded files", sys) from e

    def read_pdf(self, pdf_path: Path) -> str:
        try:
            with fitz.open(pdf_path) as doc:
                if doc.is_encrypted:
                    raise ValueError(f"PDF is encrypted: {pdf_path.name}")
                all_text = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    text = page.get_text()
                    if text.strip():
                        all_text.append(f"\n--- Page {page_num + 1} ---\n{text}")
                self.log.info("PDF read successfully | file=%s | pages=%s", pdf_path, len(all_text))
                return "\n".join(all_text)
        except Exception as e:
            self.log.error("Error reading PDF %s: %s", pdf_path, e)
            raise DocumentPortalException("Error reading PDF document", sys) from e
        
    def combine_documents(self) -> str:
        try:
            content_dict = {}
            doc_parts = []

            for filename in sorted(self.base_dir.iterdir()):
                if filename.is_file() and filename.suffix == ".pdf":
                    content_dict[filename.name] = self.read_pdf(filename)

            for filename, content in content_dict.items():
                doc_parts.append(f"Document:{filename}\n{content}")

            combined_text = "\n\n".join(doc_parts)
            self.log.info("Documents combined | count=%s", len(doc_parts))
            return combined_text
        
        except Exception as e:
            self.log.error("Error combining documents: %s", e)
            raise DocumentPortalException("Error occurred while combining", sys) from e


