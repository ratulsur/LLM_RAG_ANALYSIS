from pydantic import BaseModel, RootModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class MetaData(BaseModel):
    Summary: List[str] = Field(default_factory=list, description="Sumamry of the Doc")
    Title: str
    Author: str
    DateCreated: str
    LastModifiedDate: str
    Publisher: str
    Language: str
    PageCount: Union[int,str]
    SentimentTone: str

class ChangeFormat(BaseModel):
    page: str
    changes: str

class SummaryResponse(RootModel[list[ChangeFormat]]):
    pass

class PromptType(str, Enum):
    DOCUMENT_ANALYSIS= "document_analysis"
    DOCUMENT_COMPARISON = "document_comparison"
    CONTEXTUALIZE_QUESTION = "contextualize_question"
    CONTEXT_QA = "context_qa"







