import logging
import random
import dateutil.parser
import feedparser
from datetime import datetime, date,timedelta 
from src.DTO import NewsArticleDto

logger = logging.getLogger(__name__)

def from_feed(rss_url):
    feed = feedparser.parse(rss_url)
    article_fields = ["url", "date_published"]
    articles = []
    for entry in feed.entries:
        pub_date = dateutil.parser.parse(entry.published).strftime('%Y-%m-%d')
        article = dict(zip(article_fields, [entry.link, pub_date]))
        articles.append(article)
    return articles


def from_feeds(feeds):
    articles = []
    for feed in feeds:
        it = from_feed(feed)
        articles.extend(it)
    return articles

async def fetch_rss_source_recent(source_name, rss_feeds):
    logger.info(f'fetching news from "{source_name}" feeds')
    articles = [ NewsArticleDto(source_name = source_name, **f) for f in from_feeds(rss_feeds)]
    week_ago = datetime.now() - timedelta(weeks=1)
    p = lambda x: datetime.combine(x.date_published, datetime.min.time()) >= week_ago
    articles = list(filter(p, articles))
    return articles