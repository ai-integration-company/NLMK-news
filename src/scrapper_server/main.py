import logging
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
from typing import List

from src.parse_manager import NewsCrawlService
from src.database import get_session, NewsArticle
from src.DTO import NewsArticleDto
from src.example_articles import example_articles
import datetime

app = FastAPI()
router = APIRouter()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

crawl_service = NewsCrawlService()
crawl_service.start_scrappers()


@router.get("/week_news", response_model=List[NewsArticleDto], status_code=status.HTTP_200_OK)
async def get_news_for_week(session: Session = Depends(get_session)):
    
    date_one_week_ago = datetime.now() - datetime.timedelta(weeks=1)
    articles_query = session.query(NewsArticle)

    articles_query = articles_query.filter(
        (NewsArticle.date_published is not None) & (NewsArticle.date_published >= date_one_week_ago)
    )
    articles_query = articles_query.filter(
        or_(
            NewsArticle.date_published is not None,
            and_(
                NewsArticle.date_published is None,
                NewsArticle.date_downloaded >= date_one_week_ago
            )
        )
    )

    articles_result = articles_query.all()
    return articles_result


@router.get("/news", response_model=List[NewsArticleDto], status_code=status.HTTP_200_OK)
async def get_news_for_week():
    res = [NewsArticleDto(**a) for a in example_articles]
    return res