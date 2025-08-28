# utils/model_loader.py
import os
from typing import Any, Dict, Optional

from logger.custom_logger import CustomLogger
from utils.config_loader import load_config

# Embeddings providers
from langchain_community.embeddings import HuggingFaceEmbeddings
# Google branch kept available (in case you switch back via config)
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings  # optional
except Exception:  # pragma: no cover
    GoogleGenerativeAIEmbeddings = None  # type: ignore

# LLM provider (Groq)
from langchain_groq import ChatGroq


class ModelLoader:
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Accepts an optional config. If not provided, loads via load_config().
        This keeps backward compatibility with places that call ModelLoader()
        and those that call ModelLoader(config).
        """
        self.log = CustomLogger.get_logger(__name__)
        self.config = config or load_config()

        # Keep your existing log lines for continuity
        self.log.info("Environment variables validated")
        self.log.info("Config loaded successfully")

    # ---------------- Embeddings ----------------
    def load_embeddings(self):
        """
        Returns a LangChain-compatible embeddings object based on config.
        Supports:
          - HuggingFace (local): sentence-transformers/*
          - Google (Gemini): models/text-embedding-004  (if you switch back)
        """
        self.log.info("Loading embedding model (customize as needed)")
        self.log.info("loading embedding models")

        emb_cfg = self.config.get("embedding_model", {})
        provider = (emb_cfg.get("provider") or "").lower()
        model_name = emb_cfg.get("model_name")

        if not model_name:
            raise ValueError("Embedding model_name missing in config['embedding_model'].")

        if provider in ("huggingface", "hf", "local"):
            # Local CPU embeddings (no quotas). Example: sentence-transformers/all-MiniLM-L6-v2
            return HuggingFaceEmbeddings(model_name=model_name)

        if provider == "google":
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError(
                    "GoogleGenerativeAIEmbeddings not available. Install langchain-google-genai."
                )
            # If you switch back to Google, make sure your API key & quotas are set
            return GoogleGenerativeAIEmbeddings(model=model_name)

        raise ValueError(f"Unknown embeddings provider: {provider}")

    # ---------------- LLM ----------------
    def load_llm(self):
        """
        Returns your chat LLM (Groq) based on config.
        """
        self.log.info("Loading LLM (customize as needed)")
        llm_cfg = self.config.get("llm", {})
        model_name = llm_cfg.get("model_name")
        if not model_name:
            raise ValueError("LLM model_name missing in config['llm'].")

        temperature = llm_cfg.get("temperature", 0)
        max_tokens = llm_cfg.get("max_output_tokens", 2048)

        # Expects GROQ_API_KEY in environment, same as before
        return ChatGroq(model=model_name, temperature=temperature, max_tokens=max_tokens)
