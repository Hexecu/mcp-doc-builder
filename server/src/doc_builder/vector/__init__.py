"""
Vector module for embedding and indexing.
"""

from doc_builder.vector.chunker import Chunk, ChunkerConfig, SmartChunker, chunk_document
from doc_builder.vector.embedder import Embedder, get_embedder
from doc_builder.vector.indexer import (
    IndexStats,
    SearchResult,
    VectorIndexer,
    get_indexer,
    search_documents,
)

__all__ = [
    # Embedder
    "Embedder",
    "get_embedder",
    # Chunker
    "SmartChunker",
    "ChunkerConfig",
    "Chunk",
    "chunk_document",
    # Indexer
    "VectorIndexer",
    "IndexStats",
    "SearchResult",
    "get_indexer",
    "search_documents",
]
