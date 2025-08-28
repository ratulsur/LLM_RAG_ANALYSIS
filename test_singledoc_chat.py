import sys
from pathlib import Path
from langchain_community.vectorstores import FAISS
from src.singledoc_chat.data_ingestion import SingleDocIngestor
from src.singledoc_chat.retrieval import ConversationalRAG
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from utils.config_loader import load_config

FAISS_INDEX_PATH = Path("faiss_index")

def test_conversational_rag_on_pdf(pdf_path: str, question: str):
    try:
        # load config for ModelLoader
        config = load_config()
        model_loader = ModelLoader(config)

        # --- tiny helper to mimic an UploadedFile (name + getbuffer) ---
        class FakeUpload:
            def __init__(self, path: Path):
                self.name = path.name
                self._bytes = path.read_bytes()
            def getbuffer(self):
                return self._bytes

        # Build/Load retriever
        if (FAISS_INDEX_PATH / "index.faiss").exists():
            print("Loading FAISS...")
            embeddings = model_loader.load_embeddings()
            vectorstore = FAISS.load_local(
                folder_path=str(FAISS_INDEX_PATH),
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            )
        else:
            print("FAISS index not found. Ingesting PDF and creating index...")
            ingestor = SingleDocIngestor()
            retriever = ingestor.ingest_files([FakeUpload(Path(pdf_path))])

        # Run RAG regardless of branch above
        print("Running ConversationalRAG...")
        session_id = "test_conversational_rag"
        rag = ConversationalRAG(retriever=retriever, session_id=session_id)

        result = rag.invoke(question)
        # handle both string or dict responses
        answer = (
            result.get("answer")
            if isinstance(result, dict) and "answer" in result
            else (result.get("result") if isinstance(result, dict) else result)
        )

        print(f"\nQuestion: {question}\nAnswer: {answer}")

    except Exception as e:
        print(f"Test failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # optional: enable logging
    CustomLogger.configure_logger()

    pdf_path = r"/Users/ratulsur/Desktop/all_data/document_portal/data/singledoc_chat/MoR_main.pdf"
    question = "What is the main topic of the document?"

    if not Path(pdf_path).exists():
        print(f"PDF does not exist at: {pdf_path}")
        sys.exit(1)

    # actually run the test
    test_conversational_rag_on_pdf(pdf_path, question)

