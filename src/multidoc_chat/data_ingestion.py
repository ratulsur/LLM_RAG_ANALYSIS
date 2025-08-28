import uuid
from pathlib import Path
import sys
from datetime import datetime, timezone

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger


class DocumentIngestor:
    SUPPORTED_FILE_TYPES = {'.pdf', '.docx', '.txt', '.md'}

    def __init__(
        self,
        temp_dir: str = '/Users/ratulsur/Desktop/all_data/document_portal/data/multidoc_chat',
        faiss_dir: str = 'faiss_index',
        session_id: str | None = None
    ):
        try:
            self.log = CustomLogger.get_logger(__name__)

            self.temp_dir = Path(temp_dir)
            self.faiss_directory = Path(faiss_dir)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            self.faiss_directory.mkdir(parents=True, exist_ok=True)

            # %M minutes, %S seconds (was %H%m%s which is wrong)
            self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            # fix typos: sesion_temp_dir -> session_temp_dir, sessio_faiss_dir -> session_faiss_dir
            self.session_temp_dir = self.temp_dir / self.session_id
            self.session_faiss_dir = self.faiss_directory / self.session_id
            self.session_temp_dir.mkdir(parents=True, exist_ok=True)
            self.session_faiss_dir.mkdir(parents=True, exist_ok=True)

            self.model_loader = ModelLoader()

            #  no keyword args in logger call; use %s formatting
            self.log.info(
                "DocumentIngestor initialized | temp_base=%s | faiss_base=%s | session_id=%s | temp_path=%s | faiss_path=%s",
                str(self.temp_dir),
                str(self.faiss_directory),
                self.session_id,
                str(self.session_temp_dir),
                str(self.session_faiss_dir),
            )

        except Exception as e:
            self.log.error("Error initializing DocumentIngestor: %s", e)
            raise DocumentPortalException("Initialization error", sys) from e

    def ingest_files(self, uploaded_files):
        """uploaded_files is a list of open file objects (rb)."""
        try:
            documents = []

            for uploaded_file in uploaded_files:
                # Determine extension from the file handle path/name
                file_name = getattr(uploaded_file, "name", "upload")
                ext = Path(file_name).suffix.lower()

                if ext not in self.SUPPORTED_FILE_TYPES:
                    self.log.warning("Unsupported file type skipped: %s", ext)
                    continue

                unique_filename = f"{uuid.uuid4().hex[:8]}{ext}"
                temp_path = self.session_temp_dir / unique_filename

                # Save to temp
                uploaded_file.seek(0)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.read())
                self.log.info("File saved for ingestion: %s", temp_path)

                # Load via appropriate loader
                if ext == ".pdf":
                    loader = PyPDFLoader(str(temp_path))
                elif ext == ".docx":
                    loader = Docx2txtLoader(str(temp_path))
                elif ext in (".txt", ".md"):
                    loader = TextLoader(str(temp_path))
                else:
                    # already guarded, but keep a fallback
                    self.log.warning("No loader for extension: %s", ext)
                    continue

                docs = loader.load()
                documents.extend(docs)

            if not documents:
                raise DocumentPortalException("No documents were ingested", sys)

            self.log.info("All docs loaded | total_docs=%s | files_ingested=%s", len(documents), len(uploaded_files))
            return self._create_retriever(documents)

        except Exception as e:
            self.log.error("Error ingesting files: %s", e)
            raise DocumentPortalException("Ingestion error", sys) from e

    def _create_retriever(self, documents):
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            #  split_documents (you had split_text earlier)
            chunks = splitter.split_documents(documents)
            self.log.info("Chunking completed | chunks=%s", len(chunks))

            embeddings = self.model_loader.load_embeddings()
            vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
            vectorstore.save_local(str(self.session_faiss_dir))
            self.log.info("FAISS index saved | path=%s", str(self.session_faiss_dir))

            retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k': 5})
            self.log.info("Retriever created")
            return retriever

        except Exception as e:
            self.log.error("Error creating retriever: %s", e)
            raise DocumentPortalException("Retriever creation error", sys) from e
