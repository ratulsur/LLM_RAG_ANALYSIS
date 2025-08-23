import sys
from dotenv import load_dotenv
import pandas as pd
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from models.models import *
from prompt.prompt_library import PROMPT_REGISTRY
from utils.model_loader import ModelLoader
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser

class DocumentComparerLLM:
    def __init__(self, config):
        load_dotenv()
        self.log = CustomLogger.get_logger(__name__)

        try:
            self.loader = ModelLoader(config)
            self.llm = self.loader.load_llm()
            self.parser = JsonOutputParser(pydantic_object=SummaryResponse)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            self.prompt = PROMPT_REGISTRY["document_comparison"]
            self.chain = self.prompt | self.llm | self.fixing_parser
            self.log.info("DocumentComparer initialized")
        except Exception as e:
            self.log.error(f"Initialization failed: {e}")
            raise DocumentPortalException("Initialization failed", sys) from e

    def compare_documents(self, combined_docs: str) -> pd.DataFrame:
        try:
            inputs = {
                "combined_documents": combined_docs,  
                "format_instructions": self.parser.get_format_instructions()
            }
            self.log.info("Starting document comparison", inputs=inputs)
            response = self.chain.invoke(inputs)
            self.log.info("Document comparison completed", response=response)
            return self._format_response(response)
        except Exception as e:
            self.log.error(f"Error in compare_documents: {e}")
            raise DocumentPortalException("An error occurred while comparing documents.", sys) from e

    def _format_response(self, response_parsed: list[dict]) -> pd.DataFrame:

        """
        formats the resoponse from the LLM into a structured manner.
        
        """
        try:
            df = pd.DataFrame(response_parsed)
            self.log.info("Response formatted to dataframe")
            return df
        except Exception as e:
            self.log.error(f"Error in formatting data into dataframe: {e}")
            raise DocumentPortalException("Error formatting comparison result", sys) from e

