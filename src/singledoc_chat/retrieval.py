import sys
import os
from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils.model_loader import ModelLoader
from exception.custom_exception import DocumentPortalException
from logger.custom_logger import CustomLogger
from prompt.prompt_library import PROMPT_REGISTRY
from models.models import PromptType


class ConversationalRAG:
    # simple in-memory store for chat histories per session
    _HISTORY_STORE: dict[str, ChatMessageHistory] = {}

    def __init__(self, session_id: str, retriever) -> None:
        try:
            self.log = CustomLogger.get_logger(__name__)
            self.session_id = session_id
            self.retriever = retriever
            self.llm = self._load_llm()

            # resolving prompts safely
            self.contextualize_prompt = self._resolve_prompt(PromptType.CONTEXTUALIZE_QUESTION)
            self.qa_prompt = self._resolve_prompt(PromptType.CONTEXT_QA)

            self.history_aware_retriever = create_history_aware_retriever(
                self.llm, self.retriever, self.contextualize_prompt
            )
            self.log.info("Conversational RAG initialized | session_id=%s", session_id)

            self.qa_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)
            self.rag_chain = create_retrieval_chain(self.history_aware_retriever, self.qa_chain)
            self.log.info("RAG chains created")

            # Runnable with memory
            self.chain = RunnableWithMessageHistory(
                self.rag_chain,
                self._get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            self.log.info("RunnableWithMessageHistory set up | session_id=%s", session_id)

        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.log.error("Error initializing RAG: %s", e)
            raise DocumentPortalException("Some error in initializing the RAG", (exc_type, exc_obj, exc_tb)) from e

    def _resolve_prompt(self, prompt_type: PromptType):
        """Helper to resolve prompts from PROMPT_REGISTRY with fallback keys."""
        tried_keys = [
            prompt_type,
            prompt_type.name,
            prompt_type.value,
            str(prompt_type).lower(),
        ]
        for key in tried_keys:
            if key in PROMPT_REGISTRY:
                return PROMPT_REGISTRY[key]

        self.log.error("Required prompt missing. Tried keys=%s | available_keys=%s",
                       tried_keys, list(PROMPT_REGISTRY.keys()))
        exc_type, exc_obj, exc_tb = sys.exc_info()
        raise DocumentPortalException("Required prompt missing from PROMPT_REGISTRY",
                                      (exc_type, exc_obj, exc_tb))

    def _load_llm(self):
        try:
            llm = ModelLoader().load_llm()
            self.log.info("LLM loaded")
            return llm
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.log.error("Failed to load LLMs: %s", e)
            raise DocumentPortalException("some error in loading LLMs", (exc_type, exc_obj, exc_tb)) from e

    def _get_session_history(self, config) -> ChatMessageHistory:
        """Return a ChatMessageHistory for the given config (required signature)."""
        try:
            cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
            sid = cfg.get("session_id", self.session_id)
            if sid not in self._HISTORY_STORE:
                self._HISTORY_STORE[sid] = ChatMessageHistory()
            return self._HISTORY_STORE[sid]
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.log.error("Failed to get session history: %s", e)
            raise DocumentPortalException("some error in loading the session history",
                                          (exc_type, exc_obj, exc_tb)) from e

    def load_retriever_from_faiss(self, index_path: str):
        try:
            embeddings = ModelLoader().load_embeddings()
            if not os.path.isdir(index_path):
                raise FileNotFoundError(f"FAISS not found at: {index_path}")

            vectorstore = FAISS.load_local(
                folder_path=index_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True,
            )
            self.log.info("Loaded retriever from FAISS | path=%s", index_path)
            return vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 5})
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.log.error("Failed to load retriever: %s", e)
            raise DocumentPortalException("some error in loading retriever from FAISS",
                                          (exc_type, exc_obj, exc_tb)) from e

    def invoke(self, user_input: str) -> str:
        try:
            response = self.chain.invoke(
                {"input": user_input},
                config={"configurable": {"session_id": self.session_id}},
            )
            # handle both string and dict responses
            if isinstance(response, dict):
                answer = response.get("answer") or response.get("result") or ""
            else:
                answer = str(response) if response is not None else ""

            if not answer:
                self.log.warning("No answer received from chain")

            self.log.info("Chain invoked successfully")
            return answer
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            self.log.error("Failed to invoke the chain: %s", e)
            raise DocumentPortalException("some error in invoking the RAG",
                                          (exc_type, exc_obj, exc_tb)) from e


if __name__ == "__main__":
    CustomLogger.configure_logger()
    log = CustomLogger.get_logger(__name__)
    log.info("Starting logging test")


