import os
import sys
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config_loader import load_config
from langchain_groq import ChatGroq
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException


log = CustomLogger().get_logger(__name__)


class ModelLoader:
    def __init__(self, config):
        load_dotenv()
        self.config = config
        self._validate_env()
        log.info("Config loaded successfully")
    
    def _validate_env(self):
        """
        Validate necessary environment variables for Groq/OpenAI.
        """
        required_vars = ["GROQ_API_KEY", "OPENAI_API_KEY"]
        self.api_keys = {key: os.getenv(key) for key in required_vars}
        missing = [k for k, v in self.api_keys.items() if not v]
        if missing:
            log.error("Missing env variables", missing_vars=missing)
            raise DocumentPortalException("Missing env variables", sys)
        log.info("Environment variables validated")

    def load_embeddings(self):
        """
        Only keep what you need for Groq/OpenAI embeddings.
        """
        log.info("Loading embedding model (customize as needed)")
        # Example: if using OpenAI embeddings:
        # from langchain_openai import OpenAIEmbeddings
        # return OpenAIEmbeddings(model="text-embedding-3-small", api_key=self.api_keys["OPENAI_API_KEY"])
        return None

    def load_llm(self):
        """
        Load and return the LLM model.
        Supports: Groq, OpenAI.
        """
        llm_config = self.config["llm"]
        provider = llm_config["provider"].strip().lower()

        if provider == "groq":
            log.info(f"Loading Groq LLM: {llm_config['model_name']}")


            return ChatGroq(
                model=llm_config["model_name"],
                temperature=llm_config.get("temperature", 0.2),
                api_key=self.api_keys["GROQ_API_KEY"],
            )

        elif provider == "openai":
            from langchain_openai import ChatOpenAI
            log.info("Loading OpenAI LLM", model=llm_config["model_name"])
            return ChatOpenAI(
                model=llm_config["model_name"],
                temperature=llm_config.get("temperature", 0.2),
                max_tokens=llm_config.get("max_output_tokens", 2048),
                api_key=self.api_keys["OPENAI_API_KEY"],
            )

        else:
            log.error("Unsupported LLM provider", provider=provider)
            raise ValueError(f"Unsupported LLM provider: {provider}")


if __name__ == "__main__":
    # Example usage
    config = {
        "llm": {
            "provider": "Groq",
            "model_name": "deepseek-r1-distill-llama-70b",
            "temperature": 0,
            "max_output_tokens": 2048
        }
    }
    loader = ModelLoader(config)
    llm = loader.load_llm()

    result = llm.invoke("Hi there! Hhow are you?")
    print(f"Embedding result: {result}")

    