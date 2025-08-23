from pydantic import BaseModel, RootModel, Field
from typing import Optional, List, Dict, Any, Union

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




