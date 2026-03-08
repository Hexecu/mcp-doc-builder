"""
Knowledge Graph repository for documentation entities.
Handles all CRUD operations for DocSource, DocPage, DocChunk, DocConcept, etc.
"""

import logging
from datetime import datetime, timezone
from typing import Any

from doc_builder.kg.neo4j import get_neo4j_client
from doc_builder.utils import generate_id

logger = logging.getLogger(__name__)


class DocRepository:
    """Repository for documentation graph operations."""

    def __init__(self):
        self.client = get_neo4j_client()

    # ─────────────────────────────────────────────────────────────────────────
    # DocSource Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def create_source(
        self,
        root_url: str,
        name: str,
        domain: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Create a new documentation source."""
        source_id = generate_id("source", root_url)

        query = """
        MERGE (s:DocSource {id: $id})
        ON CREATE SET
            s.root_url = $root_url,
            s.name = $name,
            s.domain = $domain,
            s.description = $description,
            s.created_at = datetime(),
            s.status = 'pending',
            s.total_pages = 0
        ON MATCH SET
            s.name = $name,
            s.description = $description,
            s.updated_at = datetime()
        RETURN s {.*} as source
        """

        result = await self.client.execute_write_return(
            query,
            {
                "id": source_id,
                "root_url": root_url,
                "name": name,
                "domain": domain,
                "description": description,
            },
        )

        return result[0]["source"] if result else {}

    async def get_source(self, source_id: str) -> dict[str, Any] | None:
        """Get a documentation source by ID."""
        query = """
        MATCH (s:DocSource {id: $id})
        RETURN s {.*} as source
        """
        result = await self.client.execute_query(query, {"id": source_id})
        return result[0]["source"] if result else None

    async def get_source_by_url(self, root_url: str) -> dict[str, Any] | None:
        """Get a documentation source by root URL."""
        source_id = generate_id("source", root_url)
        return await self.get_source(source_id)

    async def list_sources(self, status: str | None = None) -> list[dict[str, Any]]:
        """List all documentation sources."""
        if status:
            query = """
            MATCH (s:DocSource {status: $status})
            RETURN s {.*} as source
            ORDER BY s.created_at DESC
            """
            result = await self.client.execute_query(query, {"status": status})
        else:
            query = """
            MATCH (s:DocSource)
            RETURN s {.*} as source
            ORDER BY s.created_at DESC
            """
            result = await self.client.execute_query(query)

        return [r["source"] for r in result]

    async def update_source_status(
        self,
        source_id: str,
        status: str,
        total_pages: int | None = None,
    ) -> None:
        """Update source status and optionally page count."""
        params: dict[str, Any] = {"id": source_id, "status": status}

        if total_pages is not None:
            query = """
            MATCH (s:DocSource {id: $id})
            SET s.status = $status,
                s.total_pages = $total_pages,
                s.last_crawled = datetime()
            """
            params["total_pages"] = total_pages
        else:
            query = """
            MATCH (s:DocSource {id: $id})
            SET s.status = $status
            """

        await self.client.execute_write(query, params)

    async def delete_source(self, source_id: str) -> dict[str, int]:
        """Delete a source and all its related entities."""
        query = """
        MATCH (s:DocSource {id: $id})
        OPTIONAL MATCH (s)-[:CONTAINS]->(p:DocPage)
        OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:DocChunk)
        OPTIONAL MATCH (p)-[:HAS_METATAG]->(m:DocMetatag)
        OPTIONAL MATCH (s)-[:DEFINES]->(concept:DocConcept)
        OPTIONAL MATCH (s)-[:HAS_JOB]->(j:DocCrawlJob)
        DETACH DELETE s, p, c, m, concept, j
        RETURN count(DISTINCT p) as pages_deleted,
               count(DISTINCT c) as chunks_deleted,
               count(DISTINCT concept) as concepts_deleted
        """
        result = await self.client.execute_write_return(query, {"id": source_id})
        return result[0] if result else {"pages_deleted": 0, "chunks_deleted": 0}

    # ─────────────────────────────────────────────────────────────────────────
    # DocPage Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def upsert_page(
        self,
        source_id: str,
        url: str,
        title: str,
        description: str = "",
        content_preview: str = "",
        content_hash: str = "",
        depth: int = 0,
        language: str = "en",
        word_count: int = 0,
    ) -> dict[str, Any]:
        """Create or update a documentation page."""
        page_id = generate_id("page", url)

        query = """
        MATCH (s:DocSource {id: $source_id})
        MERGE (p:DocPage {id: $page_id})
        ON CREATE SET
            p.url = $url,
            p.source_id = $source_id,
            p.title = $title,
            p.description = $description,
            p.content_preview = $content_preview,
            p.content_hash = $content_hash,
            p.depth = $depth,
            p.language = $language,
            p.word_count = $word_count,
            p.crawled_at = datetime()
        ON MATCH SET
            p.title = $title,
            p.description = $description,
            p.content_preview = $content_preview,
            p.content_hash = $content_hash,
            p.word_count = $word_count,
            p.updated_at = datetime()
        MERGE (s)-[:CONTAINS]->(p)
        RETURN p {.*} as page
        """

        result = await self.client.execute_write_return(
            query,
            {
                "source_id": source_id,
                "page_id": page_id,
                "url": url,
                "title": title,
                "description": description,
                "content_preview": content_preview,
                "content_hash": content_hash,
                "depth": depth,
                "language": language,
                "word_count": word_count,
            },
        )

        return result[0]["page"] if result else {}

    async def get_page(self, page_id: str) -> dict[str, Any] | None:
        """Get a page by ID."""
        query = """
        MATCH (p:DocPage {id: $id})
        RETURN p {.*} as page
        """
        result = await self.client.execute_query(query, {"id": page_id})
        return result[0]["page"] if result else None

    async def get_page_by_url(self, url: str) -> dict[str, Any] | None:
        """Get a page by URL."""
        query = """
        MATCH (p:DocPage {url: $url})
        RETURN p {.*} as page
        """
        result = await self.client.execute_query(query, {"url": url})
        return result[0]["page"] if result else None

    async def list_pages(
        self,
        source_id: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List pages for a source."""
        query = """
        MATCH (s:DocSource {id: $source_id})-[:CONTAINS]->(p:DocPage)
        RETURN p {.*} as page
        ORDER BY p.crawled_at DESC
        SKIP $offset
        LIMIT $limit
        """
        result = await self.client.execute_query(
            query,
            {"source_id": source_id, "limit": limit, "offset": offset},
        )
        return [r["page"] for r in result]

    async def create_page_link(
        self,
        from_url: str,
        to_url: str,
        anchor_text: str = "",
    ) -> None:
        """Create a link relationship between pages."""
        query = """
        MATCH (from:DocPage {url: $from_url})
        MATCH (to:DocPage {url: $to_url})
        MERGE (from)-[r:LINKS_TO]->(to)
        SET r.anchor_text = $anchor_text
        """
        await self.client.execute_write(
            query,
            {"from_url": from_url, "to_url": to_url, "anchor_text": anchor_text},
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DocChunk Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def create_chunk(
        self,
        page_id: str,
        content: str,
        embedding: list[float],
        chunk_index: int,
        token_count: int,
        semantic_type: str = "general",
        heading_context: str = "",
    ) -> dict[str, Any]:
        """Create a document chunk with embedding."""
        chunk_id = generate_id("chunk", page_id, str(chunk_index))

        query = """
        MATCH (p:DocPage {id: $page_id})
        MERGE (c:DocChunk {id: $chunk_id})
        ON CREATE SET
            c.page_id = $page_id,
            c.content = $content,
            c.embedding = $embedding,
            c.chunk_index = $chunk_index,
            c.token_count = $token_count,
            c.semantic_type = $semantic_type,
            c.heading_context = $heading_context,
            c.created_at = datetime()
        ON MATCH SET
            c.content = $content,
            c.embedding = $embedding,
            c.token_count = $token_count,
            c.semantic_type = $semantic_type,
            c.heading_context = $heading_context,
            c.updated_at = datetime()
        MERGE (p)-[:HAS_CHUNK]->(c)
        RETURN c {.id, .page_id, .chunk_index, .semantic_type, .token_count} as chunk
        """

        result = await self.client.execute_write_return(
            query,
            {
                "page_id": page_id,
                "chunk_id": chunk_id,
                "content": content,
                "embedding": embedding,
                "chunk_index": chunk_index,
                "token_count": token_count,
                "semantic_type": semantic_type,
                "heading_context": heading_context,
            },
        )

        return result[0]["chunk"] if result else {}

    async def delete_page_chunks(self, page_id: str) -> int:
        """Delete all chunks for a page."""
        query = """
        MATCH (p:DocPage {id: $page_id})-[:HAS_CHUNK]->(c:DocChunk)
        DETACH DELETE c
        RETURN count(c) as deleted
        """
        result = await self.client.execute_write_return(query, {"page_id": page_id})
        return result[0]["deleted"] if result else 0

    # ─────────────────────────────────────────────────────────────────────────
    # Vector Search Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def vector_search(
        self,
        embedding: list[float],
        limit: int = 10,
        min_score: float = 0.7,
        source_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform vector similarity search on chunks.
        
        Args:
            embedding: Query embedding vector
            limit: Maximum results
            min_score: Minimum similarity score
            source_ids: Optional filter by source IDs
            
        Returns:
            List of chunks with scores and page info
        """
        from doc_builder.config import get_settings
        settings = get_settings()

        if source_ids:
            query = f"""
            CALL db.index.vector.queryNodes(
                '{settings.vector_index_name}',
                $limit,
                $embedding
            ) YIELD node, score
            WHERE score >= $min_score
            MATCH (p:DocPage)-[:HAS_CHUNK]->(node)
            WHERE p.source_id IN $source_ids
            RETURN node {{.*}} as chunk,
                   score,
                   p.url as page_url,
                   p.title as page_title,
                   p.source_id as source_id
            ORDER BY score DESC
            """
            params = {
                "embedding": embedding,
                "limit": limit * 2,  # Get more to filter
                "min_score": min_score,
                "source_ids": source_ids,
            }
        else:
            query = f"""
            CALL db.index.vector.queryNodes(
                '{settings.vector_index_name}',
                $limit,
                $embedding
            ) YIELD node, score
            WHERE score >= $min_score
            MATCH (p:DocPage)-[:HAS_CHUNK]->(node)
            RETURN node {{.*}} as chunk,
                   score,
                   p.url as page_url,
                   p.title as page_title,
                   p.source_id as source_id
            ORDER BY score DESC
            """
            params = {
                "embedding": embedding,
                "limit": limit,
                "min_score": min_score,
            }

        result = await self.client.execute_query(query, params)
        return result[:limit]

    async def fulltext_search(
        self,
        query_text: str,
        limit: int = 10,
        source_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Perform fulltext search on chunks.
        
        Args:
            query_text: Search query
            limit: Maximum results
            source_ids: Optional filter by source IDs
            
        Returns:
            List of chunks with scores and page info
        """
        # Escape special characters for Lucene
        escaped = query_text.replace("~", "\\~").replace("*", "\\*")

        if source_ids:
            query = """
            CALL db.index.fulltext.queryNodes('doc_chunk_fulltext', $query)
            YIELD node, score
            MATCH (p:DocPage)-[:HAS_CHUNK]->(node)
            WHERE p.source_id IN $source_ids
            RETURN node {.*} as chunk,
                   score,
                   p.url as page_url,
                   p.title as page_title,
                   p.source_id as source_id
            ORDER BY score DESC
            LIMIT $limit
            """
            params = {"query": escaped, "limit": limit, "source_ids": source_ids}
        else:
            query = """
            CALL db.index.fulltext.queryNodes('doc_chunk_fulltext', $query)
            YIELD node, score
            MATCH (p:DocPage)-[:HAS_CHUNK]->(node)
            RETURN node {.*} as chunk,
                   score,
                   p.url as page_url,
                   p.title as page_title,
                   p.source_id as source_id
            ORDER BY score DESC
            LIMIT $limit
            """
            params = {"query": escaped, "limit": limit}

        return await self.client.execute_query(query, params)

    # ─────────────────────────────────────────────────────────────────────────
    # DocConcept Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def upsert_concept(
        self,
        source_id: str,
        name: str,
        description: str = "",
        category: str = "entity",
        confidence: float = 1.0,
    ) -> dict[str, Any]:
        """Create or update a concept in the ontology."""
        concept_id = generate_id("concept", source_id, name.lower())

        query = """
        MATCH (s:DocSource {id: $source_id})
        MERGE (c:DocConcept {id: $concept_id})
        ON CREATE SET
            c.name = $name,
            c.description = $description,
            c.category = $category,
            c.confidence = $confidence,
            c.source_id = $source_id,
            c.mention_count = 1,
            c.created_at = datetime()
        ON MATCH SET
            c.description = CASE WHEN size($description) > size(c.description)
                           THEN $description ELSE c.description END,
            c.confidence = CASE WHEN $confidence > c.confidence
                          THEN $confidence ELSE c.confidence END,
            c.mention_count = c.mention_count + 1,
            c.updated_at = datetime()
        MERGE (s)-[:DEFINES]->(c)
        RETURN c {.*} as concept
        """

        result = await self.client.execute_write_return(
            query,
            {
                "source_id": source_id,
                "concept_id": concept_id,
                "name": name,
                "description": description,
                "category": category,
                "confidence": confidence,
            },
        )

        return result[0]["concept"] if result else {}

    async def link_chunk_to_concept(
        self,
        chunk_id: str,
        concept_id: str,
        confidence: float = 1.0,
    ) -> None:
        """Create a MENTIONS relationship between chunk and concept."""
        query = """
        MATCH (c:DocChunk {id: $chunk_id})
        MATCH (concept:DocConcept {id: $concept_id})
        MERGE (c)-[r:MENTIONS]->(concept)
        SET r.confidence = $confidence
        """
        await self.client.execute_write(
            query,
            {"chunk_id": chunk_id, "concept_id": concept_id, "confidence": confidence},
        )

    async def create_concept_relation(
        self,
        from_concept_id: str,
        to_concept_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        """Create a relationship between two concepts."""
        query = """
        MATCH (from:DocConcept {id: $from_id})
        MATCH (to:DocConcept {id: $to_id})
        MERGE (from)-[r:RELATES_TO]->(to)
        SET r.type = $type, r.weight = $weight
        """
        await self.client.execute_write(
            query,
            {
                "from_id": from_concept_id,
                "to_id": to_concept_id,
                "type": relation_type,
                "weight": weight,
            },
        )

    async def get_concepts(
        self,
        source_id: str,
        category: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get concepts for a source."""
        if category:
            query = """
            MATCH (s:DocSource {id: $source_id})-[:DEFINES]->(c:DocConcept)
            WHERE c.category = $category
            RETURN c {.*} as concept
            ORDER BY c.mention_count DESC
            LIMIT $limit
            """
            params = {"source_id": source_id, "category": category, "limit": limit}
        else:
            query = """
            MATCH (s:DocSource {id: $source_id})-[:DEFINES]->(c:DocConcept)
            RETURN c {.*} as concept
            ORDER BY c.mention_count DESC
            LIMIT $limit
            """
            params = {"source_id": source_id, "limit": limit}

        result = await self.client.execute_query(query, params)
        return [r["concept"] for r in result]

    async def get_concept_graph(
        self,
        source_id: str,
        concept_name: str | None = None,
        depth: int = 2,
    ) -> dict[str, Any]:
        """Get concept relationships as a graph."""
        if concept_name:
            query = """
            MATCH path = (c:DocConcept {source_id: $source_id})-[:RELATES_TO*1..$depth]-(related)
            WHERE toLower(c.name) CONTAINS toLower($concept_name)
            WITH c, related, relationships(path) as rels
            RETURN collect(DISTINCT c {.*}) + collect(DISTINCT related {.*}) as concepts,
                   [r IN collect(DISTINCT rels) | {
                       from: startNode(r[0]).id,
                       to: endNode(r[0]).id,
                       type: r[0].type,
                       weight: r[0].weight
                   }] as relationships
            """
            params = {"source_id": source_id, "concept_name": concept_name, "depth": depth}
        else:
            query = """
            MATCH (c:DocConcept {source_id: $source_id})
            OPTIONAL MATCH (c)-[r:RELATES_TO]-(related:DocConcept)
            RETURN collect(DISTINCT c {.*}) as concepts,
                   collect(DISTINCT {
                       from: startNode(r).id,
                       to: endNode(r).id,
                       type: r.type,
                       weight: r.weight
                   }) as relationships
            """
            params = {"source_id": source_id}

        result = await self.client.execute_query(query, params)
        if result:
            return {
                "concepts": result[0].get("concepts", []),
                "relationships": [r for r in result[0].get("relationships", []) if r.get("from")],
            }
        return {"concepts": [], "relationships": []}

    # ─────────────────────────────────────────────────────────────────────────
    # DocMetatag Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def create_metatag(
        self,
        page_id: str,
        key: str,
        value: str,
    ) -> dict[str, Any]:
        """Create a metatag for a page."""
        metatag_id = generate_id("meta", page_id, key)

        query = """
        MATCH (p:DocPage {id: $page_id})
        MERGE (m:DocMetatag {id: $metatag_id})
        ON CREATE SET
            m.key = $key,
            m.value = $value
        ON MATCH SET
            m.value = $value
        MERGE (p)-[:HAS_METATAG]->(m)
        RETURN m {.*} as metatag
        """

        result = await self.client.execute_write_return(
            query,
            {"page_id": page_id, "metatag_id": metatag_id, "key": key, "value": value},
        )

        return result[0]["metatag"] if result else {}

    async def get_page_metatags(self, page_id: str) -> list[dict[str, Any]]:
        """Get all metatags for a page."""
        query = """
        MATCH (p:DocPage {id: $page_id})-[:HAS_METATAG]->(m:DocMetatag)
        RETURN m {.*} as metatag
        """
        result = await self.client.execute_query(query, {"page_id": page_id})
        return [r["metatag"] for r in result]

    # ─────────────────────────────────────────────────────────────────────────
    # DocCrawlJob Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def create_crawl_job(self, source_id: str) -> dict[str, Any]:
        """Create a new crawl job."""
        job_id = generate_id("job", source_id, datetime.now(timezone.utc).isoformat())

        query = """
        MATCH (s:DocSource {id: $source_id})
        CREATE (j:DocCrawlJob {
            id: $job_id,
            source_id: $source_id,
            started_at: datetime(),
            status: 'running',
            pages_crawled: 0,
            pages_failed: 0
        })
        MERGE (s)-[:HAS_JOB]->(j)
        RETURN j {.*} as job
        """

        result = await self.client.execute_write_return(
            query,
            {"source_id": source_id, "job_id": job_id},
        )

        return result[0]["job"] if result else {}

    async def update_crawl_job(
        self,
        job_id: str,
        status: str | None = None,
        pages_crawled: int | None = None,
        pages_failed: int | None = None,
        error_message: str | None = None,
    ) -> None:
        """Update a crawl job."""
        sets = []
        params: dict[str, Any] = {"job_id": job_id}

        if status:
            sets.append("j.status = $status")
            params["status"] = status
            if status in ("completed", "failed"):
                sets.append("j.completed_at = datetime()")

        if pages_crawled is not None:
            sets.append("j.pages_crawled = $pages_crawled")
            params["pages_crawled"] = pages_crawled

        if pages_failed is not None:
            sets.append("j.pages_failed = $pages_failed")
            params["pages_failed"] = pages_failed

        if error_message:
            sets.append("j.error_message = $error_message")
            params["error_message"] = error_message

        if sets:
            query = f"""
            MATCH (j:DocCrawlJob {{id: $job_id}})
            SET {', '.join(sets)}
            """
            await self.client.execute_write(query, params)

    async def get_latest_job(self, source_id: str) -> dict[str, Any] | None:
        """Get the latest crawl job for a source."""
        query = """
        MATCH (s:DocSource {id: $source_id})-[:HAS_JOB]->(j:DocCrawlJob)
        RETURN j {.*} as job
        ORDER BY j.started_at DESC
        LIMIT 1
        """
        result = await self.client.execute_query(query, {"source_id": source_id})
        return result[0]["job"] if result else None

    # ─────────────────────────────────────────────────────────────────────────
    # Statistics
    # ─────────────────────────────────────────────────────────────────────────

    async def get_source_stats(self, source_id: str) -> dict[str, Any]:
        """Get statistics for a documentation source."""
        query = """
        MATCH (s:DocSource {id: $source_id})
        OPTIONAL MATCH (s)-[:CONTAINS]->(p:DocPage)
        OPTIONAL MATCH (p)-[:HAS_CHUNK]->(c:DocChunk)
        OPTIONAL MATCH (s)-[:DEFINES]->(concept:DocConcept)
        RETURN s {.*} as source,
               count(DISTINCT p) as page_count,
               count(DISTINCT c) as chunk_count,
               count(DISTINCT concept) as concept_count,
               sum(c.token_count) as total_tokens
        """
        result = await self.client.execute_query(query, {"source_id": source_id})

        if result:
            r = result[0]
            return {
                "source": r["source"],
                "page_count": r["page_count"],
                "chunk_count": r["chunk_count"],
                "concept_count": r["concept_count"],
                "total_tokens": r["total_tokens"] or 0,
            }
        return {}


# Global repository instance
_repo: DocRepository | None = None


def get_repository() -> DocRepository:
    """Get the global repository instance."""
    global _repo
    if _repo is None:
        _repo = DocRepository()
    return _repo
