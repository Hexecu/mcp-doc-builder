"""
Vector indexer for storing and searching document embeddings.
Uses Neo4j Vector Index for similarity search.
"""

import logging
from dataclasses import dataclass
from typing import Any

from doc_builder.config import Settings, get_settings
from doc_builder.kg import get_repository
from doc_builder.vector.chunker import Chunk, SmartChunker
from doc_builder.vector.embedder import Embedder, get_embedder

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of a vector search."""

    chunk_id: str
    content: str
    score: float
    page_url: str
    page_title: str
    source_id: str
    semantic_type: str
    heading_context: str


@dataclass
class IndexStats:
    """Statistics for indexing operation."""

    chunks_created: int = 0
    chunks_updated: int = 0
    chunks_failed: int = 0
    total_tokens: int = 0
    empty_pages: int = 0
    embedding_failures: int = 0
    
    def __add__(self, other: "IndexStats") -> "IndexStats":
        """Add two IndexStats together."""
        return IndexStats(
            chunks_created=self.chunks_created + other.chunks_created,
            chunks_updated=self.chunks_updated + other.chunks_updated,
            chunks_failed=self.chunks_failed + other.chunks_failed,
            total_tokens=self.total_tokens + other.total_tokens,
            empty_pages=self.empty_pages + other.empty_pages,
            embedding_failures=self.embedding_failures + other.embedding_failures,
        )


class VectorIndexer:
    """
    Handles vectorization and indexing of document content.
    
    Workflow:
    1. Chunk document into semantic pieces
    2. Generate embeddings for each chunk
    3. Store chunks with embeddings in Neo4j
    4. Support similarity search via vector index
    """

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
    ):
        """
        Initialize the indexer.
        
        Args:
            settings: Optional settings override
            embedder: Optional embedder override
        """
        self.settings = settings or get_settings()
        self.embedder = embedder or get_embedder()
        self.chunker = SmartChunker(settings=self.settings)
        self.repo = get_repository()

    async def index_page(
        self,
        page_id: str,
        content: str,
        heading_context: str = "",
    ) -> IndexStats:
        """
        Index a page's content.
        
        Args:
            page_id: The page ID in the graph
            content: Page content to index
            heading_context: Optional heading context
            
        Returns:
            IndexStats with operation results
        """
        stats = IndexStats()

        if not content or not content.strip():
            # Use debug instead of warning for empty pages
            logger.debug(f"Empty content for page {page_id}")
            stats.empty_pages = 1
            return stats

        # Delete existing chunks for this page
        deleted = await self.repo.delete_page_chunks(page_id)
        if deleted > 0:
            logger.debug(f"Deleted {deleted} existing chunks for page {page_id}")

        # Chunk the content
        chunks = self.chunker.chunk(content, heading_context)

        if not chunks:
            logger.warning(f"No chunks generated for page {page_id}")
            return stats

        logger.debug(f"Generated {len(chunks)} chunks for page {page_id}")

        # Generate embeddings in batch
        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedder.embed_batch(texts)

        # Check for embedding failures (zero vectors)
        for i, embedding in enumerate(embeddings):
            if all(v == 0.0 for v in embedding[:10]):  # Check first 10 values
                stats.embedding_failures += 1

        # Store each chunk
        for chunk, embedding in zip(chunks, embeddings):
            # Skip chunks with failed embeddings (zero vectors)
            if all(v == 0.0 for v in embedding[:10]):
                logger.debug(f"Skipping chunk {chunk.index} with zero embedding")
                stats.chunks_failed += 1
                continue
                
            try:
                await self.repo.create_chunk(
                    page_id=page_id,
                    content=chunk.content,
                    embedding=embedding,
                    chunk_index=chunk.index,
                    token_count=chunk.token_count,
                    semantic_type=chunk.semantic_type,
                    heading_context=chunk.heading_context,
                )
                stats.chunks_created += 1
                stats.total_tokens += chunk.token_count
            except Exception as e:
                logger.error(f"Failed to create chunk {chunk.index} for page {page_id}: {e}")
                stats.chunks_failed += 1

        if stats.chunks_created > 0:
            logger.info(
                f"Indexed page {page_id}: {stats.chunks_created} chunks, "
                f"{stats.total_tokens} tokens"
                + (f", {stats.embedding_failures} embedding failures" if stats.embedding_failures > 0 else "")
            )
        elif stats.empty_pages > 0:
            logger.debug(f"Skipped empty page {page_id}")
        else:
            logger.debug(f"No chunks created for page {page_id}")

        return stats

    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float | None = None,
        source_ids: list[str] | None = None,
        search_mode: str = "hybrid",
    ) -> list[SearchResult]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            limit: Maximum results
            min_score: Minimum similarity score
            source_ids: Filter by source IDs
            search_mode: "vector", "fulltext", or "hybrid"
            
        Returns:
            List of SearchResult objects
        """
        min_score = min_score or self.settings.vector_similarity_threshold

        if search_mode == "vector":
            return await self._vector_search(query, limit, min_score, source_ids)
        elif search_mode == "fulltext":
            return await self._fulltext_search(query, limit, source_ids)
        else:
            # Hybrid: combine vector and fulltext
            return await self._hybrid_search(query, limit, min_score, source_ids)

    async def _vector_search(
        self,
        query: str,
        limit: int,
        min_score: float,
        source_ids: list[str] | None,
    ) -> list[SearchResult]:
        """Perform vector similarity search."""
        # Generate query embedding
        query_embedding = await self.embedder.embed(query)

        # Search
        results = await self.repo.vector_search(
            embedding=query_embedding,
            limit=limit,
            min_score=min_score,
            source_ids=source_ids,
        )

        return [
            SearchResult(
                chunk_id=r["chunk"]["id"],
                content=r["chunk"]["content"],
                score=r["score"],
                page_url=r["page_url"],
                page_title=r["page_title"],
                source_id=r["source_id"],
                semantic_type=r["chunk"].get("semantic_type", "general"),
                heading_context=r["chunk"].get("heading_context", ""),
            )
            for r in results
        ]

    async def _fulltext_search(
        self,
        query: str,
        limit: int,
        source_ids: list[str] | None,
    ) -> list[SearchResult]:
        """Perform fulltext search."""
        results = await self.repo.fulltext_search(
            query_text=query,
            limit=limit,
            source_ids=source_ids,
        )

        return [
            SearchResult(
                chunk_id=r["chunk"]["id"],
                content=r["chunk"]["content"],
                score=r["score"],
                page_url=r["page_url"],
                page_title=r["page_title"],
                source_id=r["source_id"],
                semantic_type=r["chunk"].get("semantic_type", "general"),
                heading_context=r["chunk"].get("heading_context", ""),
            )
            for r in results
        ]

    async def _hybrid_search(
        self,
        query: str,
        limit: int,
        min_score: float,
        source_ids: list[str] | None,
    ) -> list[SearchResult]:
        """
        Perform hybrid search combining vector and fulltext.
        
        Uses reciprocal rank fusion to combine results.
        """
        # Get results from both methods
        vector_results = await self._vector_search(
            query, limit * 2, min_score, source_ids
        )
        fulltext_results = await self._fulltext_search(
            query, limit * 2, source_ids
        )

        # Reciprocal Rank Fusion
        k = 60  # RRF constant
        scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        for rank, result in enumerate(vector_results):
            chunk_id = result.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            result_map[chunk_id] = result

        for rank, result in enumerate(fulltext_results):
            chunk_id = result.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank + 1)
            if chunk_id not in result_map:
                result_map[chunk_id] = result

        # Sort by combined score
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

        # Update scores and return top results
        results = []
        for chunk_id in sorted_ids[:limit]:
            result = result_map[chunk_id]
            # Update score to combined score
            result.score = scores[chunk_id]
            results.append(result)

        return results

    async def reindex_source(self, source_id: str) -> IndexStats:
        """
        Reindex all pages for a source.
        
        Args:
            source_id: Source ID to reindex
            
        Returns:
            Combined IndexStats
        """
        total_stats = IndexStats()

        # Get all pages for source
        pages = await self.repo.list_pages(source_id, limit=10000)

        logger.info(f"Reindexing {len(pages)} pages for source {source_id}")

        for page in pages:
            try:
                # We need to fetch full content - for now use content_preview
                # In a full implementation, we'd store/retrieve full content
                content = page.get("content_preview", "")
                
                if content:
                    stats = await self.index_page(
                        page_id=page["id"],
                        content=content,
                        heading_context=page.get("title", ""),
                    )
                    total_stats.chunks_created += stats.chunks_created
                    total_stats.chunks_failed += stats.chunks_failed
                    total_stats.total_tokens += stats.total_tokens

            except Exception as e:
                logger.error(f"Failed to reindex page {page.get('id')}: {e}")
                total_stats.chunks_failed += 1

        return total_stats


# Global indexer instance
_indexer: VectorIndexer | None = None


def get_indexer() -> VectorIndexer:
    """Get the global vector indexer instance."""
    global _indexer
    if _indexer is None:
        _indexer = VectorIndexer()
    return _indexer


async def search_documents(
    query: str,
    limit: int = 10,
    source_ids: list[str] | None = None,
    search_mode: str = "hybrid",
) -> list[SearchResult]:
    """
    Convenience function to search documents.
    
    Args:
        query: Search query
        limit: Maximum results
        source_ids: Optional source filter
        search_mode: Search mode
        
    Returns:
        List of SearchResult objects
    """
    indexer = get_indexer()
    return await indexer.search(
        query=query,
        limit=limit,
        source_ids=source_ids,
        search_mode=search_mode,
    )
