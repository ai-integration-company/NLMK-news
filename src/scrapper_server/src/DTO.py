from pydantic import BaseModel, EmailStr, ConfigDict, Field
from typing import Optional, List
from datetime import date


class NewsArticleDto(BaseModel):
    source_name: Optional[str] = None
    url: Optional[str] = None
    date_published: Optional[date] = None


class RSSSource(BaseModel):
    source_name:                Optional[str] = None
    rss_feeds:            List[str]     = []
