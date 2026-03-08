"""
MCP Tools for documentation ingestion, search, and management.
"""

import asyncio
import logging
from typing import Any

from doc_builder.config import get_settings
from doc_builder.crawler import CrawlResult, ParsedPage, Spider
from doc_builder.kg import get_repository
from doc_builder.ontology import extract_ontology, get_linker, store_page_metatags
from doc_builder.utils import content_hash, extract_domain, generate_id, truncate_text
from doc_builder.vector import get_indexer, search_documents

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────────
# Tool Response Models
# ─────────────────────────────────────────────────────────────────────────────────


def format_ingest_result(
    source_id: str,
    name: str,
    stats: dict,
) -> dict[str, Any]:
    """Format ingestion result for MCP response."""
    return {
        "source_id": source_id,
        "name": name,
        "status": "completed",
        "stats": stats,
        "message": f"Successfully indexed {stats.get('pages_crawled', 0)} pages with {stats.get('chunks_created', 0)} chunks",
    }


def format_search_results(results: list, query: str) -> dict[str, Any]:
    """Format search results for MCP response."""
    return {
        "query": query,
        "total_results": len(results),
        "results": [
            {
                "content": truncate_text(r.content, 500),
                "score": round(r.score, 3),
                "page_url": r.page_url,
                "page_title": r.page_title,
                "source_id": r.source_id,
                "semantic_type": r.semantic_type,
            }
            for r in results
        ],
    }


def format_context_pack(
    topic: str,
    results: list,
    concepts: list,
    relationships: list,
) -> dict[str, Any]:
    """Format context pack for MCP response."""
    return {
        "topic": topic,
        "relevant_chunks": [
            {
                "content": r.content,
                "page_url": r.page_url,
                "page_title": r.page_title,
                "semantic_type": r.semantic_type,
            }
            for r in results
        ],
        "concepts": [
            {
                "name": c.get("name"),
                "description": c.get("description"),
                "category": c.get("category"),
            }
            for c in concepts
        ],
        "relationships": relationships,
    }


# ─────────────────────────────────────────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────────────────────────────────────────


async def doc_ingest(
    url: str,
    name: str,
    max_depth: int = 2,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """
    Ingest a documentation website.
    
    Crawls the website, extracts content, generates embeddings,
    and builds an ontology graph.
    
    Args:
        url: Root URL of the documentation
        name: Name for this documentation source
        max_depth: Maximum link depth to crawl (1-5)
        include_patterns: URL patterns to include (regex)
        exclude_patterns: URL patterns to exclude (regex)
        
    Returns:
        Ingestion result with statistics
    """
    settings = get_settings()
    repo = get_repository()
    indexer = get_indexer()
    linker = get_linker()

    # Create source
    domain = extract_domain(url)
    source = await repo.create_source(
        root_url=url,
        name=name,
        domain=domain,
        description=f"Documentation from {domain}",
    )
    source_id = source["id"]

    # Update status to crawling
    await repo.update_source_status(source_id, "crawling")

    # Create crawl job
    job = await repo.create_crawl_job(source_id)

    # Initialize spider
    spider = Spider(
        root_url=url,
        doc_name=name,
        max_depth=min(max_depth, settings.crawler_max_depth),
        max_pages=settings.crawler_max_pages,
        max_concurrent=settings.crawler_max_concurrent,
        rate_limit=settings.crawler_rate_limit,
    )

    # Track statistics
    stats = {
        "pages_crawled": 0,
        "pages_failed": 0,
        "chunks_created": 0,
        "concepts_extracted": 0,
        "total_tokens": 0,
    }

    # Collect parsed pages for linking
    parsed_pages: list[ParsedPage] = []

    # Process each crawled page
    async def process_page(result: CrawlResult):
        nonlocal stats

        if not result.success or not result.page:
            stats["pages_failed"] += 1
            return

        page = result.page
        parsed_pages.append(page)
        stats["pages_crawled"] += 1

        try:
            # Store page in graph
            page_data = await repo.upsert_page(
                source_id=source_id,
                url=page.url,
                title=page.title,
                description=page.description,
                content_preview=truncate_text(page.content, 500),
                content_hash=content_hash(page.content),
                depth=result.depth,
                language=page.language,
                word_count=page.word_count,
            )
            page_id = page_data["id"]

            # Store metatags
            await store_page_metatags(page_id, page.metatags)

            # Index content (chunking + embedding)
            index_stats = await indexer.index_page(
                page_id=page_id,
                content=page.content,
                heading_context=page.title,
            )
            stats["chunks_created"] += index_stats.chunks_created
            stats["total_tokens"] += index_stats.total_tokens

            # Extract ontology (concepts + relationships)
            ontology_result = await extract_ontology(
                source_id=source_id,
                source_name=name,
                page_title=page.title,
                page_url=page.url,
                content=page.content,
            )
            stats["concepts_extracted"] += ontology_result.concepts_extracted

            # Update job progress
            await repo.update_crawl_job(
                job_id=job["id"],
                pages_crawled=stats["pages_crawled"],
                pages_failed=stats["pages_failed"],
            )

        except Exception as e:
            logger.error(f"Failed to process page {page.url}: {e}")
            stats["pages_failed"] += 1

    # Register callback and start crawling
    spider.on_page(lambda r: asyncio.create_task(process_page(r)))

    try:
        async for _ in spider.crawl():
            pass
    except Exception as e:
        logger.error(f"Crawl failed: {e}")
        await repo.update_source_status(source_id, "failed")
        await repo.update_crawl_job(job["id"], status="failed", error_message=str(e))
        raise

    # Build relationships
    try:
        link_stats = await linker.build_source_graph(source_id, parsed_pages)
        stats["page_links"] = link_stats.page_links_created
        stats["concept_relations"] = link_stats.concept_relations
    except Exception as e:
        logger.error(f"Failed to build relationships: {e}")

    # Update final status
    await repo.update_source_status(
        source_id,
        "completed",
        total_pages=stats["pages_crawled"],
    )
    await repo.update_crawl_job(job["id"], status="completed")

    return format_ingest_result(source_id, name, stats)


async def doc_search(
    query: str,
    sources: list[str] | None = None,
    limit: int = 10,
    search_mode: str = "hybrid",
) -> dict[str, Any]:
    """
    Search indexed documentation.
    
    Performs semantic and/or fulltext search across indexed documentation.
    
    Args:
        query: Search query in natural language
        sources: Optional list of source IDs to filter
        limit: Maximum number of results (1-50)
        search_mode: "vector", "fulltext", or "hybrid"
        
    Returns:
        Search results with relevance scores
    """
    limit = min(max(1, limit), 50)

    if search_mode not in ("vector", "fulltext", "hybrid"):
        search_mode = "hybrid"

    results = await search_documents(
        query=query,
        limit=limit,
        source_ids=sources,
        search_mode=search_mode,
    )

    return format_search_results(results, query)


async def doc_context(
    topic: str,
    sources: list[str] | None = None,
    include_related: bool = True,
) -> dict[str, Any]:
    """
    Get comprehensive context for a topic.
    
    Returns relevant chunks, concepts, and relationships for a topic.
    Useful for providing context to an LLM for code generation.
    
    Args:
        topic: Topic to get context for (e.g., "authentication in Next.js")
        sources: Optional list of source IDs to filter
        include_related: Include related concepts and relationships
        
    Returns:
        Context pack with chunks, concepts, and relationships
    """
    repo = get_repository()

    # Search for relevant chunks
    results = await search_documents(
        query=topic,
        limit=10,
        source_ids=sources,
        search_mode="hybrid",
    )

    concepts = []
    relationships = []

    if include_related and sources:
        # Get concepts from the specified sources
        for source_id in sources[:3]:  # Limit to avoid too many queries
            source_concepts = await repo.get_concepts(source_id, limit=50)
            concepts.extend(source_concepts)

            # Get concept graph
            graph = await repo.get_concept_graph(source_id, concept_name=topic)
            relationships.extend(graph.get("relationships", []))

    elif include_related and results:
        # Get concepts from result sources
        source_ids = list(set(r.source_id for r in results))
        for source_id in source_ids[:3]:
            source_concepts = await repo.get_concepts(source_id, limit=30)
            concepts.extend(source_concepts)

    # Deduplicate concepts
    seen_ids = set()
    unique_concepts = []
    for c in concepts:
        if c.get("id") not in seen_ids:
            seen_ids.add(c.get("id"))
            unique_concepts.append(c)

    return format_context_pack(topic, results, unique_concepts[:20], relationships[:30])


async def doc_sources() -> dict[str, Any]:
    """
    List all indexed documentation sources.
    
    Returns:
        List of documentation sources with stats
    """
    repo = get_repository()
    sources = await repo.list_sources()

    result = []
    for source in sources:
        # Get stats for each source
        stats = await repo.get_source_stats(source["id"])

        result.append({
            "id": source["id"],
            "name": source["name"],
            "root_url": source["root_url"],
            "domain": source["domain"],
            "status": source.get("status", "unknown"),
            "total_pages": stats.get("page_count", 0),
            "total_chunks": stats.get("chunk_count", 0),
            "total_concepts": stats.get("concept_count", 0),
            "last_crawled": source.get("last_crawled"),
        })

    return {
        "total_sources": len(result),
        "sources": result,
    }


async def doc_refresh(
    source_id: str,
    force: bool = False,
) -> dict[str, Any]:
    """
    Refresh a documentation source.
    
    Re-crawls the source and updates the index.
    
    Args:
        source_id: ID of the source to refresh
        force: Force re-crawl even if content hasn't changed
        
    Returns:
        Refresh result with statistics
    """
    repo = get_repository()

    # Get source info
    source = await repo.get_source(source_id)
    if not source:
        return {"error": f"Source not found: {source_id}"}

    # Re-ingest using the original URL
    return await doc_ingest(
        url=source["root_url"],
        name=source["name"],
        max_depth=get_settings().crawler_max_depth,
    )


async def doc_ontology(
    source_id: str | None = None,
    concept: str | None = None,
) -> dict[str, Any]:
    """
    Explore the extracted ontology.
    
    Returns concepts and their relationships from the knowledge graph.
    
    Args:
        source_id: Optional source ID to filter
        concept: Optional concept name to focus on
        
    Returns:
        Ontology graph with concepts and relationships
    """
    repo = get_repository()

    if source_id:
        # Get concepts for specific source
        concepts = await repo.get_concepts(source_id, limit=100)
        graph = await repo.get_concept_graph(source_id, concept_name=concept)

        return {
            "source_id": source_id,
            "total_concepts": len(concepts),
            "concepts": [
                {
                    "id": c.get("id"),
                    "name": c.get("name"),
                    "description": c.get("description"),
                    "category": c.get("category"),
                    "mention_count": c.get("mention_count", 0),
                }
                for c in concepts
            ],
            "relationships": graph.get("relationships", []),
        }

    else:
        # Get all sources and their concept counts
        sources = await repo.list_sources()
        result = []

        for source in sources:
            stats = await repo.get_source_stats(source["id"])
            result.append({
                "source_id": source["id"],
                "name": source["name"],
                "concept_count": stats.get("concept_count", 0),
            })

        return {
            "sources": result,
            "message": "Specify source_id to explore concepts",
        }


# ─────────────────────────────────────────────────────────────────────────────────
# Tool Registration
# ─────────────────────────────────────────────────────────────────────────────────

TOOLS = {
    "doc_ingest": {
        "function": doc_ingest,
        "description": """Ingest and index a documentation website.

Crawls the website using intelligent link evaluation, extracts content,
generates vector embeddings, and builds a knowledge graph with concepts
and relationships.

Use this to add new documentation sources for semantic search.""",
        "parameters": {
            "url": {"type": "string", "description": "Root URL of the documentation"},
            "name": {"type": "string", "description": "Name for this documentation source"},
            "max_depth": {
                "type": "integer",
                "description": "Maximum link depth to crawl (default: 2)",
                "default": 2,
            },
            "include_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URL patterns to include (regex)",
            },
            "exclude_patterns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "URL patterns to exclude (regex)",
            },
        },
        "required": ["url", "name"],
    },
    "doc_search": {
        "function": doc_search,
        "description": """Search indexed documentation.

Performs semantic (vector) and fulltext search across all indexed
documentation. Returns relevant chunks with source information.

Use this to find specific information in the documentation.""",
        "parameters": {
            "query": {"type": "string", "description": "Search query in natural language"},
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of source IDs to filter",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default: 10, max: 50)",
                "default": 10,
            },
            "search_mode": {
                "type": "string",
                "enum": ["vector", "fulltext", "hybrid"],
                "description": "Search mode (default: hybrid)",
                "default": "hybrid",
            },
        },
        "required": ["query"],
    },
    "doc_context": {
        "function": doc_context,
        "description": """Get comprehensive context for a topic.

Returns relevant documentation chunks along with extracted concepts
and their relationships. Ideal for providing context to an LLM
before code generation.

Use this when you need deep understanding of a topic.""",
        "parameters": {
            "topic": {"type": "string", "description": "Topic to get context for"},
            "sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional source IDs to filter",
            },
            "include_related": {
                "type": "boolean",
                "description": "Include related concepts (default: true)",
                "default": True,
            },
        },
        "required": ["topic"],
    },
    "doc_sources": {
        "function": doc_sources,
        "description": """List all indexed documentation sources.

Returns all documentation sources with their statistics including
page count, chunk count, and concept count.

Use this to see what documentation is available.""",
        "parameters": {},
        "required": [],
    },
    "doc_refresh": {
        "function": doc_refresh,
        "description": """Refresh a documentation source.

Re-crawls and re-indexes a documentation source to update the content.

Use this to update documentation after the source has changed.""",
        "parameters": {
            "source_id": {"type": "string", "description": "ID of the source to refresh"},
            "force": {
                "type": "boolean",
                "description": "Force re-crawl even if unchanged (default: false)",
                "default": False,
            },
        },
        "required": ["source_id"],
    },
    "doc_ontology": {
        "function": doc_ontology,
        "description": """Explore the extracted ontology.

Returns the knowledge graph of concepts and relationships extracted
from the documentation. Useful for understanding the structure and
key entities in the documentation.

Use this to explore what concepts exist in the documentation.""",
        "parameters": {
            "source_id": {"type": "string", "description": "Source ID to explore"},
            "concept": {"type": "string", "description": "Focus on specific concept"},
        },
        "required": [],
    },
}


def get_tools() -> dict:
    """Get all tool definitions."""
    return TOOLS
