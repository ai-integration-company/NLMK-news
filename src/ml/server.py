from models import MYLLMGraphTransformer, MyNeo4jVect, create_sys_promt, TagsRequest, TagsTextRequest
from utlis import retriever
from langchain_openai import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from yandex_chain import YandexLLM


from langchain_openai import OpenAIEmbeddings

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)

import os
from dotenv import load_dotenv

from fastapi import FastAPI

from langchain_community.graphs import Neo4jGraph


load_dotenv()
API_KEY = os.environ.get("API_KEY")
FOLDER_ID = os.environ.get("FOLDER_ID")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD)

embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

llmg = ChatOpenAI(temperature=0, model_name="gpt-4o",
                      api_key=OPENAI_API_KEY)
app = FastAPI()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125",
                      api_key=OPENAI_API_KEY)


def get_text_chunks_langchain(text, date):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=40)
    docs = [
        Document(page_content=x, matadata={"date": date, "news_name": "User", "link": "Provided by user"})
        for x in text_splitter.split_text(text)]
    return docs


@app.get("/ping")
def ping() -> str:
    return "pong"




@app.post("/weekly_news")
def question(tags: TagsRequest):
    
    vector_index = MyNeo4jVect.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
        url=NEO4J_URI, username=NEO4J_USERNAME, password=NEO4J_PASSWORD, tags=tags.tags,
    )

    graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")

    template = """You must summarize the news:

    Title: {title}
    Text: {text}
    Your summury will be in news digest. Dont write anything instead of summary. Dont write any entry words just summary.
    Answer in Russian:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "text": RunnablePassthrough(),
                'title': RunnablePassthrough(),
            }
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"answer": retriever(
        graph=graph, vector_index=vector_index, colbert_reranker="", tags=tags.tags, chain=chain,
        k_sine=5, k_rerank=10)}
