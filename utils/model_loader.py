from typing import Any, Dict, Optional
import os
from logger.custom_logger import CustomLogger

class ModelLoader:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        from dotenv import load_dotenv, find_dotenv
        load_dotenv(find_dotenv(), override=True)
        self.log = CustomLogger.get_logger(__name__)
        from utils.config_loader import load_config
        self.config = config or load_config()
        self.log.info("Environment variables validated")
        self.log.info("Config loaded successfully")

    def _require_env(self, key: str):
        if not os.getenv(key):
            raise RuntimeError(f"Missing environment variable: {key}")

    def load_embeddings(self):
        self.log.info("Loading embedding model (customize as needed)")
        emb_cfg = self.config.get("embedding_model", {})
        provider = (emb_cfg.get("provider") or "").lower()
        model_name = emb_cfg.get("model_name")
        if not model_name:
            raise ValueError("Embedding model_name missing in config['embedding_model'].")

        if provider in ("huggingface", "hf", "local"):
            from langchain_community.embeddings import HuggingFaceEmbeddings
            return HuggingFaceEmbeddings(model_name=model_name)

        if provider == "google":
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
            except Exception as e:
                raise ImportError("Install langchain-google-genai for Google embeddings") from e
            return GoogleGenerativeAIEmbeddings(model=model_name)

        raise ValueError(f"Unknown embeddings provider: {provider}")

    def load_llm(self):
        self.log.info("Loading LLM (customize as needed)")
        llm_cfg = self.config.get("llm", {})
        provider = (llm_cfg.get("provider") or "groq").lower()
        model_name = llm_cfg.get("model_name")
        temperature = llm_cfg.get("temperature", 0)
        max_tokens = llm_cfg.get("max_output_tokens", 2048)
        if not model_name:
            raise ValueError("LLM model_name missing in config['llm'].")

        if provider == "groq":
            self._require_env("GROQ_API_KEY")
            from langchain_groq import ChatGroq
            return ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens)

        if provider == "openai":
            self._require_env("OPENAI_API_KEY")
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=model_name, temperature=temperature, max_tokens=max_tokens)

        if provider == "google":
            self._require_env("GOOGLE_API_KEY")
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(model=model_name, temperature=temperature, max_output_tokens=max_tokens)

        raise ValueError(f"Unknown LLM provider: {provider}")
