import uuid
from pathlib import Path
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # ← updated import
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader
from utils.config_loader import load_config
from logger.custom_logger import CustomLogger
from datetime import datetime, timezone

class SingleDocIngestor:
    def __init__(
        self,
        data_dir: str = "/Users/ratulsur/Desktop/all_data/document_portal/data/singledoc_chat",
        faiss_directory: str = "faiss_index",
    ):
        try:
            self.log = CustomLogger.get_logger(__name__)
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)

            self.faiss_dir = Path(faiss_directory)
            self.faiss_dir.mkdir(parents=True, exist_ok=True)

            # Ensure ModelLoader has config
            self.model_loader = ModelLoader(load_config())

            self.log.info("SingleDocIngestor initialized | data_dir=%s | faiss_dir=%s",
                          self.data_dir, self.faiss_dir)
        except Exception as e:
            self.log.error("Initialization error in ingestion: %s", e)
            raise DocumentPortalException("initialization error in ingestion", sys) from e

    def ingest_files(self, uploaded_files):
        """Accepts a list of 'uploaded' files (objects with .getbuffer() or .read())"""
        try:
            documents = []

            for uploaded_file in uploaded_files:
                unique_filename = (
                    f"session_{datetime.now(timezone.utc).strftime('%Y%m%d')}_"
                    f"{uuid.uuid4().hex[:8]}.pdf"
                )
                temp_path = self.data_dir / unique_filename

                # get bytes from either getbuffer() or read()
                if hasattr(uploaded_file, "getbuffer"):
                    data = uploaded_file.getbuffer()
                elif hasattr(uploaded_file, "read"):
                    data = uploaded_file.read()
                else:
                    raise ValueError("uploaded_file must provide getbuffer() or read()")

                # write bytes
                with open(temp_path, "wb") as f_out:
                    f_out.write(data)

                # load PDF into LangChain docs
                loader = PyPDFLoader(str(temp_path))
                docs = loader.load()
                documents.extend(docs)

            self.log.info("PDFs loaded | total_docs=%s | files_ingested=%s",
                          len(documents), len(uploaded_files))

            return self.create_retriever(documents)

        except Exception as e:
            self.log.error("Error ingesting files: %s", e)
            raise DocumentPortalException("Please check your code and retry", sys) from e

    def create_retriever(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_documents(documents)
            self.log.info("Chunking completed | chunks=%s", len(chunks))

            embeddings = self.model_loader.load_embeddings()
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

            vectorstore.save_local(str(self.faiss_dir))
            self.log.info("FAISS index created and saved | path=%s", self.faiss_dir)

            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 5}
            )
            self.log.info("Retriever created")
            return retriever

        except Exception as e:
            self.log.error("Error creating retriever: %s", e)
            raise DocumentPortalException("error spotted in your code, please check", sys) from e
