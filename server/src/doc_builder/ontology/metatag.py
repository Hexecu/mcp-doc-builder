"""
Metatag extraction from HTML pages.
"""

import logging
from dataclasses import dataclass
from typing import Any

from doc_builder.crawler.parser import ExtractedMetatag, ParsedPage
from doc_builder.kg import get_repository

logger = logging.getLogger(__name__)


@dataclass
class ProcessedMetatag:
    """A processed metatag with normalized key."""

    key: str
    value: str
    category: str  # "opengraph", "twitter", "standard", "schema"


# Important metatags to extract
IMPORTANT_METATAGS = {
    # Standard
    "title",
    "description",
    "keywords",
    "author",
    "robots",
    "canonical",
    # Open Graph
    "og:title",
    "og:description",
    "og:type",
    "og:url",
    "og:image",
    "og:site_name",
    # Twitter Cards
    "twitter:card",
    "twitter:title",
    "twitter:description",
    "twitter:image",
    # Documentation specific
    "docsearch:language",
    "docsearch:version",
    "algolia-site-verification",
    "generator",
}


def categorize_metatag(key: str) -> str:
    """Categorize a metatag by its key."""
    if key.startswith("og:"):
        return "opengraph"
    elif key.startswith("twitter:"):
        return "twitter"
    elif key.startswith("article:"):
        return "opengraph"
    elif key.startswith("schema:") or key.startswith("itemprop"):
        return "schema"
    else:
        return "standard"


def process_metatags(
    metatags: list[ExtractedMetatag],
    filter_important: bool = True,
) -> list[ProcessedMetatag]:
    """
    Process and normalize metatags.
    
    Args:
        metatags: Raw extracted metatags
        filter_important: Only keep important metatags
        
    Returns:
        List of ProcessedMetatag objects
    """
    processed = []
    seen_keys = set()

    for meta in metatags:
        key = meta.key.lower().strip()
        value = meta.value.strip()

        if not key or not value:
            continue

        # Skip duplicates
        if key in seen_keys:
            continue
        seen_keys.add(key)

        # Filter to important metatags if requested
        if filter_important and key not in IMPORTANT_METATAGS:
            # But keep all og: and twitter: tags
            if not key.startswith(("og:", "twitter:")):
                continue

        category = categorize_metatag(key)

        processed.append(ProcessedMetatag(
            key=key,
            value=value[:1000],  # Limit value length
            category=category,
        ))

    return processed


async def store_page_metatags(
    page_id: str,
    metatags: list[ExtractedMetatag],
) -> int:
    """
    Store metatags for a page in the graph.
    
    Args:
        page_id: The page ID
        metatags: Raw metatags from parsing
        
    Returns:
        Number of metatags stored
    """
    repo = get_repository()
    processed = process_metatags(metatags)
    stored = 0

    for meta in processed:
        try:
            await repo.create_metatag(
                page_id=page_id,
                key=meta.key,
                value=meta.value,
            )
            stored += 1
        except Exception as e:
            logger.warning(f"Failed to store metatag {meta.key}: {e}")

    return stored


def extract_structured_data(page: ParsedPage) -> dict[str, Any]:
    """
    Extract structured data from metatags.
    
    Creates a summary dictionary from metatags.
    
    Args:
        page: Parsed page with metatags
        
    Returns:
        Dictionary of structured data
    """
    data: dict[str, Any] = {
        "title": page.title,
        "description": page.description,
        "url": page.url,
        "language": page.language,
    }

    # Process metatags into structured data
    for meta in page.metatags:
        key = meta.key.lower()

        # Open Graph
        if key == "og:type":
            data["type"] = meta.value
        elif key == "og:site_name":
            data["site_name"] = meta.value
        elif key == "og:image":
            data["image"] = meta.value

        # Standard
        elif key == "keywords":
            data["keywords"] = [k.strip() for k in meta.value.split(",")]
        elif key == "author":
            data["author"] = meta.value
        elif key == "canonical":
            data["canonical_url"] = meta.value

        # Documentation specific
        elif key == "docsearch:version" or key == "version":
            data["version"] = meta.value
        elif key == "generator":
            data["generator"] = meta.value

    return data


def get_page_summary_from_metatags(page: ParsedPage) -> str:
    """
    Generate a summary string from page metatags.
    
    Useful for providing context to LLM.
    
    Args:
        page: Parsed page
        
    Returns:
        Summary string
    """
    parts = []

    if page.title:
        parts.append(f"Title: {page.title}")
    if page.description:
        parts.append(f"Description: {page.description}")

    # Add keywords if available
    for meta in page.metatags:
        if meta.key.lower() == "keywords" and meta.value:
            parts.append(f"Keywords: {meta.value}")
            break

    # Add type if available
    for meta in page.metatags:
        if meta.key.lower() == "og:type" and meta.value:
            parts.append(f"Type: {meta.value}")
            break

    return "\n".join(parts)
