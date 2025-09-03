import os
import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from models.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY

# keep your existing logger boot
CustomLogger.configure_logger()
log = CustomLogger.get_logger(__name__)
log.info("Logger started for data_analysis")


class DocumentAnalyzer:
    """
    Analyzes document using a pre-trained model.
    Logs all operations automatically into the logger.
    """

    def __init__(self, config):
        # keep module-level boot, but also have an instance logger
        self.log = CustomLogger.get_logger(__name__)
        try:
            self.loader = ModelLoader(config)
            self.llm = self.loader.load_llm()
            self.parser = JsonOutputParser(pydantic_object=MetaData)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            self.prompt = PROMPT_REGISTRY["document_analysis"]

            self.log.info("DocumentAnalyzer initialized successfully")
        except Exception as e:
            # Avoid kwargs/extra; include details in the string and log traceback
            self.log.exception(f"Error initializing the document analyzer: {e}")
            # Propagate the real exception as the cause
            raise DocumentPortalException("error in DocumentAnalyzer", e) from e

    def analyze_document(self, document_text: str) -> dict:
        """
        Analyze a document and extract metadata and create summary.
        """
        try:
            chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("Meta-data analysis chain initialized")

            response = chain.invoke({
                "format_instructions": self.parser.get_format_instructions(),
                "document_text": document_text
            })

            # No 'extra' kwarg; just stringify what you want to see
            try:
                keys = list(response.keys()) if hasattr(response, "keys") else "n/a"
            except Exception:
                keys = "n/a"
            self.log.info(f"Metadata analysis performed successfully! | keys={keys}")

            return response

        except Exception as e:
            # Capture full traceback
            self.log.exception(f"Metadata Analysis failed: {e}")
            # Preserve original exception as context
            raise DocumentPortalException("Metadata extraction failed", e) from e
