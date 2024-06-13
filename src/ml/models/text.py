from pydantic import BaseModel
from typing import List


class TagsRequest(BaseModel):
    tags: List[str]


class TagsTextRequest(BaseModel):
    tags: List[str]
    text: str
