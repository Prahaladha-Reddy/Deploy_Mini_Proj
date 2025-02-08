from pydantic import BaseModel
from typing import List,Dict


class QuestionRequest(BaseModel):
    question: str

class DocumentResponse(BaseModel):
    content: str
    metadata: dict

class AnswerResponse(BaseModel):
    question: str
    answer: str
    Encoded_Images: List[str]


