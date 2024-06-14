from langchain_core.documents import Document
import json
from typing import List


def generate_full_text_query(input: str) -> str:
    words = [el for el in input.split() if el]
    full_text_query = " AND ".join([f"{word}~2" for word in words])
    return full_text_query.strip()


def structured_retriever(graph, tags: List[str]) -> str:
    result = ""
    for entity in tags:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node, score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' + node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result


def generate_news_digest(news_items: List[Document]) -> str:
    digest = []
    for el in news_items:
        news_name = el.metadata['news_name']
        news_link = el.metadata['source']
        summary = el.metadata['summary']
        date = el.metadata['date']

        digest.append(f"{news_name} ({news_link}). {date}:\n{summary}\n\n")
    return digest


def reranking(sine, k, question, colbert_reranker):
    documents_text = [el.page_content for el in sine]
    scores = colbert_reranker(question, documents_text)
    documents_with_scores = sorted(zip(sine, scores), key=lambda x: x[1], reverse=True)
    sorted_documents = [item[0] for item in documents_with_scores][:k]
    return sorted_documents


def retriever(graph, vector_index, colbert_reranker, tags: List[str], chain, k_sine, k_rerank) -> str:
    question = "News by tags " + ", ".join(tags)
    structured_data = structured_retriever(graph, tags)
    sine = vector_index.similarity_search(question, k=k_sine)
    # sorted_documents = reranking(sine=sine, k=k_rerank, question=question, colbert_reranker=colbert_reranker)
    for el in sine:
        el.metadata['summary'] = chain.invoke({"text": el.page_content, "context": structured_data, "title": el.metadata.get('title', None)})
    return generate_news_digest(sine)
