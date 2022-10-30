from pydantic import BaseModel
from typing import Optional

class SentimentRequest(BaseModel):
    text: Optional[str]
    request_id: Optional[str]
    class Config:
        arbitrary_types_allowed = True
        schema_extra = {
            "example":{
                "text": "the real crap is that annoying thing", 
                "request_id": "1234-04x4-894"       
            }
        }
class SentimentResponse(BaseModel):
    status_code: Optional[int]
    message: Optional[str]
    sentiment: Optional[list[str]]
