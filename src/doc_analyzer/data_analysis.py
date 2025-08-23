import os
import sys
from utils.model_loader import ModelLoader
from logger.custom_logger import CustomLogger
from exception.custom_exception import DocumentPortalException
from models.models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import OutputFixingParser
from prompt.prompt_library import PROMPT_REGISTRY


CustomLogger.configure_logger()  

log = CustomLogger.get_logger(__name__)
log.info("Logger started for data_analysis")


class DocumentAnalyzer:
    """
    Analyzes document using a pre-trained model.
    Logs all operations automatically into the logger.
    """

    def __init__(self, config):
        self.log = CustomLogger().get_logger(__name__)
        try:
            self.loader = ModelLoader(config)
            self.llm = self.loader.load_llm()  
            self.parser = JsonOutputParser(pydantic_object=MetaData)
            self.fixing_parser = OutputFixingParser.from_llm(parser=self.parser, llm=self.llm)
            self.prompt = PROMPT_REGISTRY["document_analysis"]

            self.log.info("DocumentAnalyzer initialized successfully")

        except Exception as e:
            self.log.error(f"Error initializing the document analyzer: {e}")
            raise DocumentPortalException("error in DocumentAnalyzer", sys)

    
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

            self.log.info("Metadata analysis performed successfully!", extra={"keys": list(response.keys())})
            return response

        except Exception as e:
            self.log.error("Metadata Analysis failed", extra={"error": str(e)})
            raise DocumentPortalException("Metadata extraction failed", str(e))


        


        
    