from __future__ import annotations

import enum
import logging
import os
from hashlib import md5
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import Neo4jVector

from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores.utils import DistanceStrategy

DEFAULT_DISTANCE_STRATEGY = DistanceStrategy.COSINE
DISTANCE_MAPPING = {
    DistanceStrategy.EUCLIDEAN_DISTANCE: "euclidean",
    DistanceStrategy.COSINE: "cosine",
}

COMPARISONS_TO_NATIVE = {
    "$eq": "=",
    "$ne": "<>",
    "$lt": "<",
    "$lte": "<=",
    "$gt": ">",
    "$gte": ">=",
}

SPECIAL_CASED_OPERATORS = {
    "$in",
    "$nin",
    "$between",
}

TEXT_OPERATORS = {
    "$like",
    "$ilike",
}

LOGICAL_OPERATORS = {"$and", "$or"}

SUPPORTED_OPERATORS = (
    set(COMPARISONS_TO_NATIVE)
    .union(TEXT_OPERATORS)
    .union(LOGICAL_OPERATORS)
    .union(SPECIAL_CASED_OPERATORS)
)


class SearchType(str, enum.Enum):
    """Enumerator of the Distance strategies."""

    VECTOR = "vector"
    HYBRID = "hybrid"


DEFAULT_SEARCH_TYPE = SearchType.VECTOR


class IndexType(str, enum.Enum):
    """Enumerator of the index types."""

    NODE = "NODE"
    RELATIONSHIP = "RELATIONSHIP"


DEFAULT_INDEX_TYPE = IndexType.NODE


class MyNeo4jVect(Neo4jVector):

    def __init__(
        self,
        embedding: Embeddings,
        *,
        search_type: SearchType = SearchType.VECTOR,
        username: Optional[str] = None,
        password: Optional[str] = None,
        url: Optional[str] = None,
        keyword_index_name: Optional[str] = "keyword",
        database: Optional[str] = None,
        index_name: str = "vector",
        node_label: str = "Chunk",
        embedding_node_property: str = "embedding",
        text_node_property: str = "text",
        distance_strategy: DistanceStrategy = DEFAULT_DISTANCE_STRATEGY,
        logger: Optional[logging.Logger] = None,
        pre_delete_collection: bool = False,
        retrieval_query: str = "",
        relevance_score_fn: Optional[Callable[[float], float]] = None,
        index_type: IndexType = DEFAULT_INDEX_TYPE,
        graph: Optional[Neo4jGraph] = None,
        tags
    ) -> None:
        try:
            import neo4j
        except ImportError:
            raise ImportError(
                "Could not import neo4j python package. "
                "Please install it with `pip install neo4j`."
            )
        self.tags = tags

        # Allow only cosine and euclidean distance strategies
        if distance_strategy not in [
            DistanceStrategy.EUCLIDEAN_DISTANCE,
            DistanceStrategy.COSINE,
        ]:
            raise ValueError(
                "distance_strategy must be either 'EUCLIDEAN_DISTANCE' or 'COSINE'"
            )

        # Graph object takes precedent over env or input params
        if graph:
            self._driver = graph._driver
            self._database = graph._database
        else:
            # Handle if the credentials are environment variables
            # Support URL for backwards compatibility
            if not url:
                url = os.environ.get("NEO4J_URL")

            url = get_from_dict_or_env({"url": url}, "url", "NEO4J_URI")
            username = get_from_dict_or_env(
                {"username": username}, "username", "NEO4J_USERNAME"
            )
            password = get_from_dict_or_env(
                {"password": password}, "password", "NEO4J_PASSWORD"
            )
            database = get_from_dict_or_env(
                {"database": database}, "database", "NEO4J_DATABASE", "neo4j"
            )

            self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
            self._database = database
            # Verify connection
            try:
                self._driver.verify_connectivity()
            except neo4j.exceptions.ServiceUnavailable:
                raise ValueError(
                    "Could not connect to Neo4j database. "
                    "Please ensure that the url is correct"
                )
            except neo4j.exceptions.AuthError:
                raise ValueError(
                    "Could not connect to Neo4j database. "
                    "Please ensure that the username and password are correct"
                )

        self.schema = ""
        # Verify if the version support vector index
        self._is_enterprise = False
        self.verify_version()

        # Verify that required values are not null
        check_if_not_null(
            [
                "index_name",
                "node_label",
                "embedding_node_property",
                "text_node_property",
            ],
            [index_name, node_label, embedding_node_property, text_node_property],
        )

        self.embedding = embedding
        self._distance_strategy = distance_strategy
        self.index_name = index_name
        self.keyword_index_name = keyword_index_name
        self.node_label = node_label
        self.embedding_node_property = embedding_node_property
        self.text_node_property = text_node_property
        self.logger = logger or logging.getLogger(__name__)
        self.override_relevance_score_fn = relevance_score_fn
        self.retrieval_query = retrieval_query
        self.search_type = search_type
        self._index_type = index_type
        # Calculate embedding dimension
        self.embedding_dimension = len(embedding.embed_query("foo"))

        # Delete existing data if flagged
        if pre_delete_collection:
            from neo4j.exceptions import DatabaseError

            self.query(
                f"MATCH (n:`{self.node_label}`) "
                "CALL { WITH n DETACH DELETE n } "
                "IN TRANSACTIONS OF 10000 ROWS;"
            )
            # Delete index
            try:
                self.query(f"DROP INDEX {self.index_name}")
            except DatabaseError:  # Index didn't exist yet
                pass

    @classmethod
    def from_existing_graph(
        cls: Type[Neo4jVector],
        embedding: Embeddings,
        node_label: str,
        embedding_node_property: str,
        text_node_properties: List[str],
        tags,
        *,
        keyword_index_name: Optional[str] = "keyword",
        index_name: str = "vector",
        search_type: SearchType = DEFAULT_SEARCH_TYPE,
        retrieval_query: str = "",
        **kwargs: Any,
    ) -> Neo4jVector:
        """
        Initialize and return a Neo4jVector instance from an existing graph.

        This method initializes a Neo4jVector instance using the provided
        parameters and the existing graph. It validates the existence of
        the indices and creates new ones if they don't exist.

        Returns:
        Neo4jVector: An instance of Neo4jVector initialized with the provided parameters
                    and existing graph.

        Example:
        >>> neo4j_vector = Neo4jVector.from_existing_graph(
        ...     embedding=my_embedding,
        ...     node_label="Document",
        ...     embedding_node_property="embedding",
        ...     text_node_properties=["title", "content"]
        ... )

        Note:
        Neo4j credentials are required in the form of `url`, `username`, and `password`,
        and optional `database` parameters passed as additional keyword arguments.
        """
        # Validate the list is not empty
        if not text_node_properties:
            raise ValueError(
                "Parameter `text_node_properties` must not be an empty list"
            )
        # Prefer retrieval query from params, otherwise construct it
        if not retrieval_query:
            retrieval_query = (
                f"RETURN reduce(str='', k IN {text_node_properties} |"
                " str + '\\n' + k + ': ' + coalesce(node[k], '')) AS text, "
                "node {.*, `"
                + embedding_node_property
                + "`: Null, id: Null, "
                + ", ".join([f"`{prop}`: Null" for prop in text_node_properties])
                + "} AS metadata, score"
            )
        print(retrieval_query)
        store = cls(
            embedding=embedding,
            index_name=index_name,
            keyword_index_name=keyword_index_name,
            search_type=search_type,
            retrieval_query=retrieval_query,
            node_label=node_label,
            embedding_node_property=embedding_node_property,
            tags=tags,
            **kwargs,
        )

        # Check if the vector index already exists
        embedding_dimension, index_type = store.retrieve_existing_index()

        # Raise error if relationship index type
        if index_type == "RELATIONSHIP":
            raise ValueError(
                "`from_existing_graph` method does not support "
                " existing relationship vector index. "
                "Please use `from_existing_relationship_index` method"
            )

        # If the vector index doesn't exist yet
        if not embedding_dimension:
            store.create_new_index()
        # If the index already exists, check if embedding dimensions match
        elif not store.embedding_dimension == embedding_dimension:
            raise ValueError(
                f"Index with name {store.index_name} already exists."
                "The provided embedding function and vector index "
                "dimensions do not match.\n"
                f"Embedding function dimension: {store.embedding_dimension}\n"
                f"Vector index dimension: {embedding_dimension}"
            )
        # FTS index for Hybrid search
        if search_type == SearchType.HYBRID:
            fts_node_label = store.retrieve_existing_fts_index(text_node_properties)
            # If the FTS index doesn't exist yet
            if not fts_node_label:
                store.create_new_keyword_index(text_node_properties)
            else:  # Validate that FTS and Vector index use the same information
                if not fts_node_label == store.node_label:
                    raise ValueError(
                        "Vector and keyword index don't index the same node label"
                    )

        # Populate embeddings
        while True:
            where_clause = " OR ".join([f"nn:{tag}" for tag in tags])
            fetch_query = (
                "MATCH (nn) "
                f"WHERE {where_clause} "
                "OPTIONAL MATCH (nn)-[r1]-(connected1) "
                "OPTIONAL MATCH (connected1)-[r2]-(connected2) "
                "WITH collect(nn) + collect(connected1) + collect(connected2) AS allNodes "
                "UNWIND allNodes AS node "
                "WITH DISTINCT node, allNodes "
                f"MATCH (n:`{node_label}`) "
                "WHERE n IN allNodes "
                f"AND n.{embedding_node_property} IS null "
                "AND any(k in $props WHERE n[k] IS NOT null) "
                f"RETURN elementId(n) AS id, reduce(str='',"
                "k IN $props | str + '\\n' + k + ':' + coalesce(n[k], '')) AS text "
                "LIMIT 1000"
            )
            print(fetch_query)
            data = store.query(fetch_query, params={"props": text_node_properties})
            text_embeddings = embedding.embed_documents([el["text"] for el in data])

            params = {
                "data": [
                    {"id": el["id"], "embedding": embedding}
                    for el, embedding in zip(data, text_embeddings)
                ]
            }

            print("UNWIND $data AS row "
                  f"MATCH (n:`{node_label}`) "
                  "WHERE elementId(n) = row.id "
                  f"CALL db.create.setVectorProperty(n, "
                  f"'{embedding_node_property}', row.embedding) "
                  "YIELD node RETURN count(*)")

            store.query(
                "UNWIND $data AS row "
                f"MATCH (n:`{node_label}`) "
                "WHERE elementId(n) = row.id "
                f"CALL db.create.setVectorProperty(n, "
                f"'{embedding_node_property}', row.embedding) "
                "YIELD node RETURN count(*)",
                params=params,
            )
            print(data)
            # If embedding calculation should be stopped
            if len(data) < 1000:
                break
        return store

    def similarity_search_with_score_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
        params: Dict[str, Any] = {},
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """
        Perform a similarity search in the Neo4j database using a
        given vector and return the top k similar documents with their scores.

        This method uses a Cypher query to find the top k documents that
        are most similar to a given embedding. The similarity is measured
        using a vector index in the Neo4j database. The results are returned
        as a list of tuples, each containing a Document object and
        its similarity score.

        Args:
            embedding (List[float]): The embedding vector to compare against.
            k (int, optional): The number of top similar documents to retrieve.

        Returns:
            List[Tuple[Document, float]]: A list of tuples, each containing
                                a Document object and its similarity score.
        """
        if filter:
            # Verify that 5.18 or later is used
            if not self.support_metadata_filter:
                raise ValueError(
                    "Metadata filtering is only supported in "
                    "Neo4j version 5.18 or greater"
                )
            # Metadata filtering and hybrid doesn't work
            if self.search_type == SearchType.HYBRID:
                raise ValueError(
                    "Metadata filtering can't be use in combination with "
                    "a hybrid search approach"
                )
            parallel_query = (
                "CYPHER runtime = parallel parallelRuntimeSupport=all "
                if self._is_enterprise
                else ""
            )
            base_index_query = parallel_query + (
                f"MATCH (n:`{self.node_label}`) WHERE "
                f"n.`{self.embedding_node_property}` IS NOT NULL AND "
                f"size(n.`{self.embedding_node_property}`) = "
                f"toInteger({self.embedding_dimension}) AND "
            )
            base_cosine_query = (
                " WITH n as node, vector.similarity.cosine("
                f"n.`{self.embedding_node_property}`, "
                "$embedding) AS score ORDER BY score DESC LIMIT toInteger($k) "
            )
            filter_snippets, filter_params = construct_metadata_filter(filter)
            index_query = base_index_query + filter_snippets + base_cosine_query

        else:
            index_query = self.get_search_index_query(self.search_type, self._index_type)
            filter_params = {}

        if self._index_type == IndexType.RELATIONSHIP:
            default_retrieval = (
                f"RETURN relationship.`{self.text_node_property}` AS text, score, "
                f"relationship {{.*, `{self.text_node_property}`: Null, "
                f"`{self.embedding_node_property}`: Null, id: Null }} AS metadata"
            )
        else:
            default_retrieval = (
                f"RETURN node.`{self.text_node_property}` AS text, score, "
                f"node {{.*, `{self.text_node_property}`: Null, "
                f"`{self.embedding_node_property}`: Null, id: Null }} AS metadata"
            )

        retrieval_query = (
            self.retrieval_query if self.retrieval_query else default_retrieval
        )

        read_query = index_query + retrieval_query
        parameters = {
            "index": self.index_name,
            "k": k,
            "embedding": embedding,
            "keyword_index": self.keyword_index_name,
            "query": remove_lucene_chars(kwargs["query"]),
            **params,
            **filter_params,
        }

        results = self.query(read_query, params=parameters)

        if any(result["text"] is None for result in results):
            if not self.retrieval_query:
                raise ValueError(
                    f"Make sure that none of the `{self.text_node_property}` "
                    f"properties on nodes with label `{self.node_label}` "
                    "are missing or empty"
                )
            else:
                raise ValueError(
                    "Inspect the `retrieval_query` and ensure it doesn't "
                    "return None for the `text` column"
                )

        docs = [
            (
                Document(
                    page_content=dict_to_yaml_str(result["text"])
                    if isinstance(result["text"], dict)
                    else result["text"],
                    metadata={
                        k: v for k, v in result["metadata"].items() if v is not None
                    },
                ),
                result["score"],
            )
            for result in results
        ]
        return docs

    def get_search_index_query(
        self, search_type: SearchType, index_type: IndexType = DEFAULT_INDEX_TYPE
    ) -> str:
        where_clause = " OR ".join([f"nn:{tag}" for tag in self.tags])
        if index_type == IndexType.NODE:
            type_to_query_map = {
                SearchType.VECTOR: (
                    "CALL db.index.vector.queryNodes($index, $k, $embedding) "
                    "YIELD node, score "
                ),
                SearchType.HYBRID: (
                    "MATCH (nn) "
                    f"WHERE {where_clause} "
                    "OPTIONAL MATCH (nn)-[r1]-(connected1) "
                    "OPTIONAL MATCH (connected1)-[r2]-(connected2) "
                    "WITH collect(nn) + collect(connected1) + collect(connected2) AS allNodes "
                    "UNWIND allNodes AS nod "
                    "WITH DISTINCT nod, allNodes "
                    "CALL { "
                    "WITH allNodes "
                    "CALL db.index.vector.queryNodes($index, $k, $embedding) "
                    "YIELD node, score "
                    "WHERE node IN allNodes "
                    "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
                    "UNWIND nodes AS n "
                    # We use 0 as min
                    "RETURN n.node AS node, (n.score / max) AS score UNION "
                    "WITH allNodes "
                    "CALL db.index.fulltext.queryNodes($keyword_index, $query, "
                    "{limit: $k}) YIELD node, score "
                    "WHERE node IN allNodes "
                    "WITH collect({node:node, score:score}) AS nodes, max(score) AS max "
                    "UNWIND nodes AS n "
                    # We use 0 as min
                    "RETURN n.node AS node, (n.score / max) AS score "
                    "} "
                    # dedup
                    "WITH node, max(score) AS score ORDER BY score DESC LIMIT $k "
                ),
            }
            return type_to_query_map[search_type]
        else:
            return (
                "CALL db.index.vector.queryRelationships($index, $k, $embedding) "
                "YIELD relationship, score "
            )


def check_if_not_null(props: List[str], values: List[Any]) -> None:
    """Check if the values are not None or empty string"""
    for prop, value in zip(props, values):
        if not value:
            raise ValueError(f"Parameter `{prop}` must not be None or empty string")
