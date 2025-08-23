import io
from pathlib import Path
from src.doc_compare.ingestion import DocumentIngestion
from src.doc_compare.document_comparer import DocumentComparerLLM
from utils.config_loader import load_config
from logger.custom_logger import CustomLogger


# --------- Configure logging once ----------
CustomLogger.configure_logger()
log = CustomLogger.get_logger(__name__)


# ---- Setup: Load local PDF files as if they were "uploaded" ---- #
def load_fake_uploaded_file(file_path: Path):
    return io.BytesIO(file_path.read_bytes())  # simulate .getbuffer()


def test_compare_documents():
    ref_path = Path("/Users/ratulsur/Desktop/all_data/document_portal/data/document_compare/doc1.pdf")
    act_path = Path("/Users/ratulsur/Desktop/all_data/document_portal/data/document_compare/doc2.pdf")

    class FakeUpload:
        def __init__(self, file_path: Path):
            self.name = file_path.name
            self._buffer = file_path.read_bytes()

        def getbuffer(self):
            return self._buffer

    # ---- Step 1: Ingestion ---- #
    ingestion = DocumentIngestion()
    log.info("Session directory created: %s", ingestion.base_dir)
    print(f"Session directory created: {ingestion.base_dir}")

    ref_upload = FakeUpload(ref_path)
    act_upload = FakeUpload(act_path)

    log.info("Saving uploaded files...")
    ref_file, act_file = ingestion.save_uploaded_files(ref_upload, act_upload)

    log.info("Combining documents...")
    combined_text = ingestion.combine_documents()

    if hasattr(ingestion, "clean_old_sessions"):
        ingestion.clean_old_sessions(keep_latest=3)
        log.info("Old sessions cleanup done (kept latest=3)")

    print("\nCombined Text Preview (First 1000 chars):\n")
    print(combined_text[:1000])
    log.info("Combined text length=%s", len(combined_text))

    # ---- Step 2: Run LLM comparison ---- #
    config = load_config()
    log.info("Loaded config: %s", config)
    llm_comparator = DocumentComparerLLM(config=config)

    log.info("Running document comparison...")
    df = llm_comparator.compare_documents(combined_text)

    print("\nComparison DataFrame:\n")
    print(df)
    log.info("Comparison complete | rows=%s | cols=%s", len(df), list(df.columns))


if __name__ == "__main__":
    log.info("Starting document comparison test...")
    test_compare_documents()
    log.info("Finished document comparison test.")
