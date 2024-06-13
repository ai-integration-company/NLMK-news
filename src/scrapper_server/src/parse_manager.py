import json
import logging
from src.DTO import NewsSource
from src.rss import update_rss_source
from typing import List, Optional
from schedule_manager import ScheduleManager

logger = logging.getLogger(__name__)

def get_news_sources():
    logger.info("getting news sources")
    with open("news_sources.json") as news_sources_file:
        data = json.load(news_sources_file)
        sources = []
        for d in data:
            sources.append(NewsSource(**d))
        return sources


class NewsCrawlService:
    news_sources: List[NewsSource]
    manager: ScheduleManager
    default_period = 900
    
    def __init__(self):
        self.news_sources= get_news_sources()
        self.manager = ScheduleManager()

    def start_scrappers(self):
        logger.info("Starting scrappers")
        i = 1
        for source in self.news_sources:
            logger.info(f"source = {str(source)}")
            def job_wrapper():
                sn = source.name
                feeds = source.rss_urls
                try:
                    return update_rss_source(sn, feeds)
                except Exception as e:
                    logger.error(f"news_source: {sn}, error: {e}")
            
            task_name = f"rss-task:source={source.name},number={i}"
            i += 1
            logger.info(f"task_name = {task_name}")
            self.manager.register_task(name=task_name, job=job_wrapper).period(self.default_period).start()