import uuid
from pathlib import Path
import sys
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  
from exception.custom_exception import DocumentPortalException
from utils.model_loader import ModelLoader
from utils.config_loader import load_config
from logger.custom_logger import CustomLogger
from datetime import datetime, timezone

class DocumentIngestor:
    SUPPORTED_FILE_TYPES = {'.pdf', '.docx', '.txt', '.md'}
    def __init__(self, temp_dir:str = '/Users/ratulsur/Desktop/all_data/document_portal/data/multidoc_chat', faiss_dir:str = 'faiss_index', session_id:str|None = None):

        try:
           self.temp_dir = Path(temp_dir)
           self.faiss_directory = Path(faiss_dir)
           self.temp_dir.mkdir(parents=True, exist_ok=True)
           self.faiss_directory.mkdir(parents=True, exist_ok=True)

           self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%m%s')}_{uuid.uuid4().hex[:8]}"
           self.sesion_temp_dir = self.temp_dir/self.session_id
           self.session_faiss_dir = self.faiss_directory/self.session_id
           self.sesion_temp_dir.mkdir(parents=True, exist_ok=True)
           self.session_faiss_dir.mkdir(parents=True, exist_ok=True)

           self.model_loader = ModelLoader()
           self.log.info(
               "DocumentIngestor initialised",
               temp_base = str(self.temp_dir),
               faiss_base = str(self.faiss_directory),
               session_id = self.session_id,
               temp_path = str(self.sesion_temp_dir),
               faiss_path = str(self.sessio_faiss_dir),
           )
                                      

        except Exception as e:
            self.log.error("error in ingesting!", error = str(e))
            raise DocumentPortalException("initialization error", sys)
        

    def ingest_file(self, uploaded_files):
        try:
            documents = []
            for uploaded_file in uploaded_files:
                ext = Path(uploaded_file).suffix.lower()
                if ext not in self.SUPPORTED_FILE_TYPES:
                    self.log.warning("unsupported file type skipped") 
                unique_filename = f"{uuid.uuid4().hex[:8]}{ext}"
                temp_path = self.sesion_temp_dir/unique_filename

                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.read())
                self.log.info("File saved for ingestion")

                if ext =='.pdf':
                    loader = PyPDFLoader(str(temp_path))
                elif ext == '.docx':
                    loader = Docx2txtLoader(str(temp_path))

                elif ext =='.txt':
                    loader = TextLoader(str(temp_path))

                else:
                    self.log.warning("Not supported!")

                docs = loader.load()
                documents.extend(docs)

            if not documents:
                raise DocumentPortalException("Ingestion Error!", sys)
                self.log.info("All docs loaded")
                return self._create_retriever(documents)
            

                
        except Exception as e:
            self.log.error("error in ingesting!", error = str(e))
            raise DocumentPortalException("initialization error", sys)

    def _create_retriever(self, documents):
        try:
           splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
           chunks = splitter.split_text(documents)
           self.log.info("Chunking completed")
           embeddings = self.model_loader.load_embeddings()
           vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
           vectorstore.save_local(str(self.session_faiss_dir))
           self.log.info("FAISS index saved")
           



        
        except Exception as e:
            self.log.error("error in ingesting!", error = str(e))
            raise DocumentPortalException("initialization error", sys)

    
