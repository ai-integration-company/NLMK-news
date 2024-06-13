from utlis import get_text_chunks
from models import colbert_reranker, MYLLMGraphTransformer, MyNeo4jVect, create_sys_promt, TagsRequest, TagsTextRequest
from utlis import retriever
from langchain_openai import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import TokenTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

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
# gpt-4-0125-preview occasionally has issues

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)


app = FastAPI()


def get_text_chunks_langchain(text, date):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=40)
    docs = [
        Document(page_content=x, matadata={"date": date, "news_name": "User", "link": "Provided by user"})
        for x in text_splitter.split_text(text)]
    return docs


@app.get("/ping")
def ping() -> str:
    return "pong"


# @app.post("/load_text")
# def load_text(tags: TagsTextRequest):
#     promt = create_sys_promt(tags.tags)
#     llmg = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125",
#                       api_key=OPENAI_API_KEY)
#     llm_transformer = MYLLMGraphTransformer(llm=llmg, prompt=promt)

#     graph_documents = llm_transformer.convert_to_graph_documents(get_text_chunks_langchain(tags.text, "2023-06-13"))

#     graph.add_graph_documents(
#         graph_documents,
#         baseEntityLabel=True,
#         include_source=True,
#     )


@app.post("/weekly_news")
def question(tags: TagsRequest):
    promt = create_sys_promt(tags.tags)
    llmg = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125",
                      api_key=OPENAI_API_KEY)
    # llm_transformer = MYLLMGraphTransformer(llm=llmg, prompt=promt)
    # news = [
    #    {'date_published': '2023-06-01', 'url': 'https://habr.com/ru/articles/780008/', 'source_name': 'habr'},
    #    {'date_published': '2023-06-02',
    #     'url': 'https://www.rbc.ru/sport/09/06/2024/6665b86f9a79472a6485f52a?from=newsfeed', 'source_name': 'rbc'},
    #    {'date_published': '2023-06-03',
    #     'url': 'https://www.reddit.com/r/OpenAI/comments/187fzdb/openai_api_free_alternative_or_does_openai_api/',
    #     'source_name': 'reddit'}]

    # docs = []
    # for i in news:
    #    doc = WebBaseLoader(i['url']).load()
#
#        doc[0].metadata['date'] = i['date_published']
#        doc[0].metadata['news_name'] = i['source_name']
#        docs.append(doc[0])
#
#    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=24)
#    documents = text_splitter.split_documents(docs[0:])
#    graph_documents = llm_transformer.convert_to_graph_documents(documents)
#
#    graph.add_graph_documents(
#        graph_documents,
#        baseEntityLabel=True,
#        include_source=True,
#    )

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

    template = """You must summarize the news depends on next structured data:
        Structured data: {context}

        Text: {text}
        Your summury will be in news digest. Dont write anything instead of summary.
        Answer in Russian:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        RunnableParallel(
            {
                "context": RunnablePassthrough(),
                "text": RunnablePassthrough(),
            },
        )
        | prompt
        | llmg
        | StrOutputParser()
    )

    return {"answer": retriever(
        graph=graph, vector_index=vector_index, colbert_reranker=colbert_reranker, tags=tags.tags, chain=chain,
        k_sine=30, k_rerank=10)}
