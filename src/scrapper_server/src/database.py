from datetime import date
from config import DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME
from sqlalchemy import Integer, String, Date, engine, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, mapped_column, Mapped, Session

DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

engine = create_engine(DATABASE_URL)
session_maker = sessionmaker(engine, expire_on_commit=False)


def get_session():
    with session_maker() as session:
        yield session


class Base(DeclarativeBase):
    pass


class NewsArticle(Base):
    __tablename__ = 'news_articles'

    id:              Mapped[int]  = mapped_column(Integer, primary_key=True)
    source_name:     Mapped[str]  = mapped_column(String)
    url:             Mapped[str]  = mapped_column(String, unique=True)
    date_published:  Mapped[date] = mapped_column(Date)
    date_downloaded: Mapped[date] = mapped_column(Date)
