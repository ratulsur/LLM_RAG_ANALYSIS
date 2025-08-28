# utils/model_loader.py
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv, find_dotenv  # <-- add this
from logger.custom_logger import CustomLogger
from utils.config_loader import load_config

# Embeddings providers
from langchain_community.embeddings import HuggingFaceEmbeddings
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
except Exception:
    GoogleGenerativeAIEmbeddings = None

# LLM provider (Groq)
from langchain_groq import ChatGroq


class ModelLoader:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        # Always load .env first
        load_dotenv(find_dotenv(), override=True)

        self.log = CustomLogger.get_logger(__name__)
        self.config = config or load_config()

        self.log.info("Environment variables validated")
        self.log.info("Config loaded successfully")

    # ---------------- Embeddings ----------------
    def load_embeddings(self):
        self.log.info("Loading embedding model (customize as needed)")
        self.log.info("loading embedding models")

        emb_cfg = self.config.get("embedding_model", {})
        provider = (emb_cfg.get("provider") or "").lower()
        model_name = emb_cfg.get("model_name")

        if not model_name:
            raise ValueError("Embedding model_name missing in config['embedding_model'].")

        if provider in ("huggingface", "hf", "local"):
            return HuggingFaceEmbeddings(model_name=model_name)

        if provider == "google":
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError(
                    "GoogleGenerativeAIEmbeddings not available. Install langchain-google-genai."
                )
            return GoogleGenerativeAIEmbeddings(model=model_name)

        raise ValueError(f"Unknown embeddings provider: {provider}")

    # ---------------- LLM ----------------
    def load_llm(self):
        self.log.info("Loading LLM (customize as needed)")
        llm_cfg = self.config.get("llm", {})
        model_name = llm_cfg.get("model_name")
        if not model_name:
            raise ValueError("LLM model_name missing in config['llm'].")

        temperature = llm_cfg.get("temperature", 0)
        max_tokens = llm_cfg.get("max_output_tokens", 2048)

        # ChatGroq will now always find GROQ_API_KEY from os.environ
        return ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens)
