import json
import logging
import random
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.concurrency import asynccontextmanager
from typing import List
from src.DTO import NewsArticleDto, RSSSource
from src.example_articles import example_articles

from src.rss import fetch_rss_source_recent

news_sources = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("getting news sources")
    with open("news_sources.json") as news_sources_file:
        data = json.load(news_sources_file)
        for d in data:
            news = RSSSource(source_name=d["source_name"],
                             rss_feeds=d["rss_feeds"])
            news_sources.append(news)
    yield
    news_sources.clear()

app = FastAPI(lifespan=lifespan)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.get("/news", response_model=List[NewsArticleDto], status_code=status.HTTP_200_OK)
async def fetch_all_news():
    limit = 40
    articles = []
    futures = []
    for s in news_sources:
        futures.append(fetch_rss_source_recent(s.source_name, s.rss_feeds))
    for f in futures:
        a = await f
        random.shuffle(a)
        articles.extend( a[:min(limit,len(a))-1] )
        
    return articles


@app.get("/example_news", response_model=List[NewsArticleDto], status_code=status.HTTP_200_OK)
async def get_news_for_week():
    res = [NewsArticleDto(**a) for a in example_articles]
    return res