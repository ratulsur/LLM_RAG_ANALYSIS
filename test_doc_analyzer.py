import os
import sys
from pathlib import Path
from src.doc_analyzer.data_ingestion import DocumentHandler
from src.doc_analyzer.data_analysis import DocumentAnalyzer
from utils.config_loader import load_config

pdf_path = "/Users/ratulsur/Desktop/all_data/bootcamp_projects/data/document_compare"

class DummyFile:
    def __init__(self, file_path):
        self.name = Path(file_path).name
        self._file_path = file_path

    def getbuffer(self):
        return open(self._file_path, "rb").read()

def main():
    try:
        #  Load configuration for the analyzer
        config = load_config()

        print("Starting PDF ingestion")
        dummy_pdf = DummyFile(pdf_path)

        # Initialize handler
        handler = DocumentHandler(session_id="test_session")

        # Save PDF
        saved_path = handler.save_pdf(dummy_pdf)
        print(f"PDF saved at: {saved_path}")

        # Read content
        text_content = handler.read_pdf(saved_path)
        print(f"Extracted text length: {len(text_content)} chars\n")

        #  Prevent LLM payload from being too large
        text_content = text_content[:10000]

        # Analyze document
        print("Starting metadata analysis...")
        analyzer = DocumentAnalyzer(config)
        analysis_result = analyzer.analyze_document(text_content)

        print("\n==== METADATA ANALYSIS RESULT ====")
        for key, value in analysis_result.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


