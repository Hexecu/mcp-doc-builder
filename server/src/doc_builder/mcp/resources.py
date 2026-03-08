"""
MCP Resources for documentation access.
"""

import logging
from typing import Any

from doc_builder.kg import get_repository
from doc_builder.utils import safe_json_dumps

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────────
# Resource Handlers
# ─────────────────────────────────────────────────────────────────────────────────


async def get_source_pages(source_id: str, limit: int = 100) -> dict[str, Any]:
    """
    Get pages for a documentation source.
    
    Resource URI: doc://sources/{source_id}/pages
    
    Args:
        source_id: Documentation source ID
        limit: Maximum pages to return
        
    Returns:
        List of pages with metadata
    """
    repo = get_repository()
    pages = await repo.list_pages(source_id, limit=limit)

    return {
        "source_id": source_id,
        "total_pages": len(pages),
        "pages": [
            {
                "id": p.get("id"),
                "url": p.get("url"),
                "title": p.get("title"),
                "description": p.get("description"),
                "depth": p.get("depth"),
                "word_count": p.get("word_count"),
                "crawled_at": p.get("crawled_at"),
            }
            for p in pages
        ],
    }


async def get_source_concepts(source_id: str, category: str | None = None) -> dict[str, Any]:
    """
    Get concepts for a documentation source.
    
    Resource URI: doc://sources/{source_id}/concepts
    
    Args:
        source_id: Documentation source ID
        category: Optional category filter
        
    Returns:
        List of concepts
    """
    repo = get_repository()
    concepts = await repo.get_concepts(source_id, category=category, limit=200)

    # Group by category
    by_category: dict[str, list] = {}
    for c in concepts:
        cat = c.get("category", "other")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append({
            "id": c.get("id"),
            "name": c.get("name"),
            "description": c.get("description"),
            "mention_count": c.get("mention_count", 0),
        })

    return {
        "source_id": source_id,
        "total_concepts": len(concepts),
        "by_category": by_category,
    }


async def get_concept_graph(source_id: str, concept_name: str | None = None) -> dict[str, Any]:
    """
    Get the concept relationship graph.
    
    Resource URI: doc://sources/{source_id}/graph
    
    Args:
        source_id: Documentation source ID
        concept_name: Optional concept to focus on
        
    Returns:
        Graph with nodes and edges
    """
    repo = get_repository()
    graph = await repo.get_concept_graph(source_id, concept_name=concept_name)

    return {
        "source_id": source_id,
        "focus": concept_name,
        "nodes": graph.get("concepts", []),
        "edges": graph.get("relationships", []),
    }


async def get_source_stats(source_id: str) -> dict[str, Any]:
    """
    Get statistics for a documentation source.
    
    Resource URI: doc://sources/{source_id}/stats
    
    Args:
        source_id: Documentation source ID
        
    Returns:
        Statistics including counts and totals
    """
    repo = get_repository()
    stats = await repo.get_source_stats(source_id)
    source = stats.get("source", {})

    return {
        "source_id": source_id,
        "name": source.get("name"),
        "status": source.get("status"),
        "pages": stats.get("page_count", 0),
        "chunks": stats.get("chunk_count", 0),
        "concepts": stats.get("concept_count", 0),
        "total_tokens": stats.get("total_tokens", 0),
        "last_crawled": source.get("last_crawled"),
    }


async def get_page_content(page_id: str) -> dict[str, Any]:
    """
    Get full content for a page.
    
    Resource URI: doc://pages/{page_id}
    
    Args:
        page_id: Page ID
        
    Returns:
        Page with full content and chunks
    """
    repo = get_repository()
    page = await repo.get_page(page_id)

    if not page:
        return {"error": f"Page not found: {page_id}"}

    metatags = await repo.get_page_metatags(page_id)

    return {
        "id": page.get("id"),
        "url": page.get("url"),
        "title": page.get("title"),
        "description": page.get("description"),
        "content_preview": page.get("content_preview"),
        "language": page.get("language"),
        "word_count": page.get("word_count"),
        "depth": page.get("depth"),
        "metatags": {m.get("key"): m.get("value") for m in metatags},
    }


# ─────────────────────────────────────────────────────────────────────────────────
# Resource Registry
# ─────────────────────────────────────────────────────────────────────────────────

RESOURCES = {
    "doc://sources/{source_id}/pages": {
        "handler": get_source_pages,
        "description": "Get pages for a documentation source",
        "parameters": {
            "source_id": {"type": "string", "description": "Documentation source ID"},
            "limit": {"type": "integer", "description": "Maximum pages to return"},
        },
    },
    "doc://sources/{source_id}/concepts": {
        "handler": get_source_concepts,
        "description": "Get concepts for a documentation source",
        "parameters": {
            "source_id": {"type": "string", "description": "Documentation source ID"},
            "category": {"type": "string", "description": "Filter by category"},
        },
    },
    "doc://sources/{source_id}/graph": {
        "handler": get_concept_graph,
        "description": "Get the concept relationship graph",
        "parameters": {
            "source_id": {"type": "string", "description": "Documentation source ID"},
            "concept_name": {"type": "string", "description": "Focus on specific concept"},
        },
    },
    "doc://sources/{source_id}/stats": {
        "handler": get_source_stats,
        "description": "Get statistics for a documentation source",
        "parameters": {
            "source_id": {"type": "string", "description": "Documentation source ID"},
        },
    },
    "doc://pages/{page_id}": {
        "handler": get_page_content,
        "description": "Get full content for a page",
        "parameters": {
            "page_id": {"type": "string", "description": "Page ID"},
        },
    },
}


def get_resources() -> dict:
    """Get all resource definitions."""
    return RESOURCES
