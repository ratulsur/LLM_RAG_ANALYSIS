import sys
from pathlib import Path
from src.multidoc_chat.data_ingestion import DocumentIngestor
from src.singledoc_chat.retrieval import ConversationalRAG
from logger.custom_logger import CustomLogger
from utils.config_loader import load_config
from utils.model_loader import ModelLoader


def test_document_ingestion_and_rag():
    try:
        # Test files (PDF, DOCX, TXT, etc.)
        test_files = [
            "/Users/ratulsur/Desktop/all_data/document_portal/data/multidoc_chat/09e1d49b-3f68-4976-8d56-34d69614d8c3.pdf",
            "/Users/ratulsur/Desktop/all_data/document_portal/data/multidoc_chat/LSTM.txt",
            "/Users/ratulsur/Desktop/all_data/document_portal/data/multidoc_chat/Natural Language Processing Meets Quantum Physics_ A Survey and Categorization.pdf",
            "/Users/ratulsur/Desktop/all_data/document_portal/data/multidoc_chat/nlp.docx",
            "/Users/ratulsur/Desktop/all_data/document_portal/data/multidoc_chat/quantum_nlp.pdf",
        ]

        uploaded_files = []
        for file_path in test_files:
            if Path(file_path).exists():
                uploaded_files.append(open(file_path, "rb"))
            else:
                print(f"File does not exist: {file_path}")

        if not uploaded_files:
            print("No valid files to upload")
            sys.exit(1)

        # Ingest and create retriever
        ingestor = DocumentIngestor()
        retriever = ingestor.ingest_files(uploaded_files)

        # Close file handles
        for f in uploaded_files:
            f.close()

        # Initialize RAG
        session_id = "test_multi_doc_chat"
        rag = ConversationalRAG(session_id=session_id, retriever=retriever)

        # Test query
        question = "What is the application of quantum computing in NLP?"
        answer = rag.invoke(question)

        print("\nQuestion:", question)
        print("Answer:", answer)

    except Exception as e:
        print(f"Testing failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    CustomLogger.configure_logger()
    test_document_ingestion_and_rag()
