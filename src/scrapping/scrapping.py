import dateutil.parser
import feedparser
from bs4 import BeautifulSoup
import requests

news_rss_feeds = {
    'CNews'             : ['https://www.cnews.ru/inc/rss/news.xml'],
    'TAdviser'          : ['http://www.tadviser.ru/xml/tadviser.xml'],
    'Rusbase'           : ['https://rb.ru/feeds/all/'],
    'Habr'              : ["https://habr.com/ru/rss/news/?fl=ru&limit=40"],
    'VC.ru'             : ["https://vc.ru/rss/all"],
    'Коммерсантъ'       : ["https://www.kommersant.ru/RSS/news.xml"],
    'Ведомости'         : ['https://vedomosti.ru/rss/news',
                            'https://vedomosti.ru/rss/issue',
                            'https://vedomosti.ru/rss/articles'],
    'РБК'               : ['https://rssexport.rbc.ru/rbcnews/news/50/full.rss'],
    'Интерфакс'         : ['https://www.interfax.ru/rss.asp'],
    'The Verge'         : ['http://www.theverge.com/rss/index.xml'],
    'TechCrunch'        : ['https://techcrunch.com/feed/'],
    'CIO'               : ['https://www.cio.com/feed/', 
                           'https://www.cio.com/comments/feed/'],
    'Forbes Technology' : ['https://www.forbes.com/innovation/feed/'],
    'VentureBeat'       : ['http://feeds.feedburner.com/venturebeat/SZYF'],
    'Digital Trends'    : ['https://www.digitaltrends.com/feed/'],
    'ITPro Today'       : ['https://www.itprotoday.com/rss.xml'],
    'Computerworld'     : ['https://www.computerworld.com/feed/'],
    'ZDNet'             : ['http://www.zdnet.com/rss.xml'],
    'Bloomberg'         : ["https://feeds.bloomberg.com/markets/news.rss",
                            "https://feeds.bloomberg.com/politics/news.rss",
                            "https://feeds.bloomberg.com/technology/news.rss",
                            "https://feeds.bloomberg.com/wealth/news.rss",
                            "https://feeds.bloomberg.com/economics/news.rss",
                            "https://feeds.bloomberg.com/industries/news.rss",
                            "https://feeds.bloomberg.com/green/news.rss",
                            "https://feeds.bloomberg.com/bview/news.rss"],
    'Business Insider'  : ['https://feeds.businessinsider.com/custom/all'],
    'EdSurge'           : ['https://www.edsurge.com/articles_rss'],
    'Technode'          : ['https://technode.com/feed/'],
    'TechEU'            : ['https://tech.eu/feed/'],
    '36kr'              : ['https://36kr.com/feed']
}

def parse_rss_feed(rss_url):
    feed = feedparser.parse(rss_url)

    article_fields = ["link", "published"]
    articles = []
    for entry in feed.entries:
        pub_date = dateutil.parser.parse(entry.published).strftime('%Y-%m-%d')
        article = dict(zip(article_fields, [entry.link, pub_date]))
        articles.append(article)
        
    return articles

def get_all_news(source_name, url, news_href, news_card_tag, news_card_class, pub_date_tag, pub_date_class):

    response = requests.get(url + news_href, headers = {'User-agent': 'Who(?)'}, verify=False)
    
    if response.status_code == 200:
        page_content = response.content
        soup = BeautifulSoup(page_content, 'html.parser')
        elements = soup.find_all(news_card_tag, class_= news_card_class, href = True)

        article_fields = ["source_name", "link", "published"]
        all_articles = []
        
        for e in elements:
            link = ""
            if news_card_tag == "a":
                link = url + e.get("href")
            else:
                tag_a = e.find_all("a")
                link = url + tag_a[0].get("href")
        
            pub_tag = e.find_all(pub_date_tag, pub_date_class)
            date = pub_tag[0].text.strip() 
            article = dict(zip(article_fields, [source_name, link, date]))
            all_articles.append(article)
            
        return all_articles
    else:
        print(f'Failed to load page with status code {response.status_code}')
        return None


if __name__ == "__main__":
    all_articles = []
    for source, feeds in news_rss_feeds.items():
    articles = []
    for feed in feeds:
        it = parse_rss_feed(feed)
        for i in it:
            i.update({"source_name": source})
        articles.extend(it)
    all_articles.extend(articles)
    
    print(articles) # ахтунг! многабукав!

    ss_news = get_all_news("Северсталь", "https://severstal.com", "/rus/media/", "a", "news-list-card", "p", "text-2 news-list-card__date")
    print(ss_news)