"""
Knowledge Graph module for documentation storage.
"""

from doc_builder.kg.neo4j import (
    Neo4jClient,
    close_neo4j_client,
    get_neo4j_client,
)
from doc_builder.kg.repo import DocRepository, get_repository

__all__ = [
    "Neo4jClient",
    "get_neo4j_client",
    "close_neo4j_client",
    "DocRepository",
    "get_repository",
]
