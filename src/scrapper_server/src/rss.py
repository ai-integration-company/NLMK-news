import logging
import dateutil.parser
import feedparser
import datetime
from fastapi import Depends
from sqlalchemy.orm import Session
from src.database import NewsArticle, get_session

logger = logging.getLogger(__name__)

def parse_rss_feed(rss_url):
    feed = feedparser.parse(rss_url)
    article_fields = ["link", "date_published", "date_downloaded"]
    articles = []
    for entry in feed.entries:
        pub_date = dateutil.parser.parse(entry.published).strftime('%Y-%m-%d')
        downl_date = datetime.date.today().strftime("%Y-%m-%d")
        article = dict(zip(article_fields, [entry.link, pub_date, downl_date]))
        articles.append(article)
    return articles


def get_news_from_feeds(feeds):
    articles = []
    for feed in feeds:
        it = parse_rss_feed(feed)
        articles.extend(it)
    return articles


def update_rss_source(source_name, rss_feeds):
    session = get_session()
    logger.info(f'fetching news from "{source_name}" feeds')

    articles = get_news_from_feeds(rss_feeds)
    for art in articles:
        week_ago = datetime.now() - datetime.timedelta(weeks=1)
        art_timestamp = datetime.fromtimestamp(art['date_published'])
        if  week_ago <= art_timestamp <= datetime.now():
            art = NewsArticle(source_name,
                              datetime.datetime.strptime(art['date_published'], '%Y-%m-%d').date(), 
                              art['date_downloaded']
                              )
            session.add(art)
    session.commit()

