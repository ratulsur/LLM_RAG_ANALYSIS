"""
Testing code for document comparison using LLMs
"""

import io
from pathlib import Path
from src.doc_compare.data_ingestion import DocumentIngestion
from src.doc_compare.document_comparer import DocumentComparerLLM


# ---- Setup: Load local PDF files as if they were "uploaded" ---- #
def load_fake_uploaded_file(file_path: Path):
    return io.BytesIO(file_path.read_bytes())  # simulate .getbuffer()


# ---- Step 1: Save and combine PDFs ---- #
def test_compare_documents():
    ref_path = Path("/Users/ratulsur/Desktop/all_data/document_portal/data/document_compare/session/doc1.pdf")
    act_path = Path("/Users/ratulsur/Desktop/all_data/document_portal/data/document_compare/session/doc2.pdf")

    # Wrap them like Streamlit UploadedFile-style
    class FakeUpload:
        def __init__(self, file_path: Path):
            self.name = file_path.name
            self._buffer = file_path.read_bytes()

        def getbuffer(self):
            return self._buffer

    # Instantiate ingestion
    ingestion = DocumentIngestion()
    ref_upload = FakeUpload(ref_path)
    act_upload = FakeUpload(act_path)

    # Save files and combine
    ref_file, act_file = ingestion.save_uploaded_files(ref_upload, act_upload)
    combined_text = ingestion.combine_documents()

    # Optional cleanup of old sessions
    if hasattr(ingestion, "clean_old_sessions"):
        ingestion.clean_old_sessions(keep_latest=3)

    print("\n Combined Text Preview (First 1000 chars):\n")
    print(combined_text[:1000])

    from utils.config_loader import load_config
    config = load_config()
    llm_comparator = DocumentComparerLLM(config=config)
    df = llm_comparator.compare_documents(combined_text)

    print("\n Comparison DataFrame:\n")
    print(df)


if __name__ == "__main__":
    test_compare_documents()


