import sys
import os
from operator import itemgetter
from typing import List, Optional, Dict, Any
from exception.custom_exception import 
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from logger.custom_logger import CustomLogger
from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger import GLOBAL_LOGGER as log
from prompt.prompt_library import PROMPT_REGISTRY
from models.models import PromptType



class ConversationalRAG:
    def __init__(self, session_id:str, retriever=None):
        try:
            self.log = CustomLogger.get_logger(__name__)
            self.session_id = session_id
            self.retriever = retriever
            self.llm = self._load_llm()
            self.retriever = retriever

            if retriever is None:
                raise ValueError("Retriever is none!")
            self._build_lcel_chain()
            self.log.info("conversationalRAG initialized", session_id=session_id)


            # resolving prompts safely
            self.contextualize_prompt = self._resolve_prompt(PromptType.CONTEXTUALIZE_QUESTION)
            self.qa_prompt = self._resolve_prompt(PromptType.CONTEXT_QA)
        except Exception as e:
            self.log.error("some error in initializing RAG", error=str(e))
            raise DocumentPortalException ("initialization error in conversational RAG", sys)
        
    

    def load_retriever_from_faiss(self, index_path:str):
        """
        loads a retriever from faiss.
        this will be later used for retrieval
        """
        try:
            embeddings = ModelLoader().load_embeddings()
            if not os.path.isdir(index_path):
                raise DocumentPortalException(f"FAISS directory not found:{index_path}")
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True,
            )
            self.retriever = vectorstore.as_retriever(search_type = 'similarity', search_kwargs = {'k':5})
            self.log.ino("FIASS loaded successfully")
            self._build_lcel_chain()
            return self.retriever

        except Exception as e:
            self.log.error("Some error in loading the retrieval", error=str(e))
            raise DocumentPortalException("Retry creating the retriever", sys)
    
    def invoke(self, user_input:str, chat_history:Optional[List[BaseMessage]]=None)->str:
        try:
            chat_history = chat_history or []
            payload = {"input":user_input, "chat_history":chat_history}
            answer = self.chain.invoke(payload)
            if not answer:
                self.log.warning("no answer generated")
                return "no answer generated"
            self.log.info("chain invoked successfully")
        except Exception as e:
            self.log.error("Some error in invoking", error=str(e))
            raise DocumentPortalException("Retry invoking", sys)


    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            if not llm:
                raise ValueError("Failed to load LLM")
            self.log.info("LLM loaded successfully")
            return llm


       
        except Exception as e:
            self.log.error("Some error in loading llm", error=str(e))
            raise DocumentPortalException("Retry loading the conversationalRAG", sys)


    @staticmethod
    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    def _build_lcel_chain(self):
        try:
            retrieve_docs = self.retriever | self._format_docs
            self.chain = (
                {
                    "context":retrieve_docs,
                    "input": itemgetter("input"),
                    "chat_history":itemgetter("chat_history")
                }
                |self.qa_prompt
                |self.llm
                |StrOutputParser
            )
            self.log.info("LCEL graph created successfully")
        except Exception as e:
            self.log.error("Some error in building the LCEL Chain", error=str(e))
            raise DocumentPortalException("Retry building the LCEL Chain", sys)



    

    

        