"""
Utility functions for serialization and common operations.
"""

import hashlib
import json
import re
from datetime import datetime
from typing import Any
from urllib.parse import urljoin, urlparse

from neo4j.time import DateTime as Neo4jDateTime


def neo4j_serializer(obj: Any) -> Any:
    """
    Serialize Neo4j types and other objects to JSON-compatible format.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, Neo4jDateTime):
        return obj.to_native().isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def safe_json_dumps(obj: Any, **kwargs) -> str:
    """
    Safely serialize object to JSON string.
    
    Args:
        obj: Object to serialize
        **kwargs: Additional arguments for json.dumps
        
    Returns:
        JSON string
    """
    return json.dumps(obj, default=neo4j_serializer, **kwargs)


def safe_json_loads(s: str) -> Any:
    """
    Safely parse JSON string, returning empty dict on error.
    
    Args:
        s: JSON string to parse
        
    Returns:
        Parsed object or empty dict
    """
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}


def generate_id(prefix: str, *components: str) -> str:
    """
    Generate a deterministic ID from components.
    
    Args:
        prefix: ID prefix (e.g., "page", "chunk")
        *components: Components to hash
        
    Returns:
        ID string like "page_abc123"
    """
    content = "|".join(str(c) for c in components)
    hash_part = hashlib.sha256(content.encode()).hexdigest()[:12]
    return f"{prefix}_{hash_part}"


def content_hash(content: str) -> str:
    """
    Generate a hash of content for change detection.
    
    Args:
        content: Text content to hash
        
    Returns:
        SHA256 hash string
    """
    return hashlib.sha256(content.encode()).hexdigest()


def normalize_url(url: str, base_url: str | None = None) -> str:
    """
    Normalize a URL, resolving relative paths.
    
    Args:
        url: URL to normalize
        base_url: Base URL for resolving relative paths
        
    Returns:
        Normalized absolute URL
    """
    # Strip whitespace and fragments
    url = url.strip()
    
    # Remove fragment
    url = url.split("#")[0]
    
    # Resolve relative URLs
    if base_url and not url.startswith(("http://", "https://", "//")):
        url = urljoin(base_url, url)
    
    # Parse and reconstruct to normalize
    parsed = urlparse(url)
    
    # Ensure scheme
    if not parsed.scheme:
        url = f"https:{url}" if url.startswith("//") else f"https://{url}"
        parsed = urlparse(url)
    
    # Remove trailing slash from path (except root)
    path = parsed.path.rstrip("/") if parsed.path != "/" else "/"
    
    # Reconstruct
    normalized = f"{parsed.scheme}://{parsed.netloc}{path}"
    if parsed.query:
        normalized += f"?{parsed.query}"
    
    return normalized


def extract_domain(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: URL to parse
        
    Returns:
        Domain string (e.g., "docs.example.com")
    """
    parsed = urlparse(url)
    return parsed.netloc


def is_same_domain(url1: str, url2: str) -> bool:
    """
    Check if two URLs are from the same domain.
    
    Args:
        url1: First URL
        url2: Second URL
        
    Returns:
        True if same domain
    """
    return extract_domain(url1) == extract_domain(url2)


def is_valid_doc_url(url: str) -> bool:
    """
    Check if URL is valid for documentation crawling.
    
    Args:
        url: URL to validate
        
    Returns:
        True if URL is valid for crawling
    """
    # Must be HTTP(S)
    if not url.startswith(("http://", "https://")):
        return False
    
    parsed = urlparse(url)
    
    # Skip common non-doc patterns
    skip_patterns = [
        r"/api/",           # API endpoints
        r"/login",          # Auth pages
        r"/signup",
        r"/logout",
        r"/admin",
        r"/cart",
        r"/checkout",
        r"\.(pdf|zip|tar|gz|exe|dmg|pkg|deb|rpm)$",  # Downloads
        r"\.(png|jpg|jpeg|gif|svg|ico|webp)$",        # Images
        r"\.(css|js|woff|woff2|ttf|eot)$",            # Assets
        r"\.(mp4|mp3|avi|mov|wav)$",                  # Media
    ]
    
    path = parsed.path.lower()
    for pattern in skip_patterns:
        if re.search(pattern, path):
            return False
    
    return True


def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


def clean_text(text: str) -> str:
    """
    Clean text by normalizing whitespace.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple whitespace with single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading/trailing whitespace
    return text.strip()


def extract_title_from_url(url: str) -> str:
    """
    Extract a title-like string from URL path.
    
    Args:
        url: URL to parse
        
    Returns:
        Title-like string
    """
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    
    if not path:
        return parsed.netloc
    
    # Get last path segment
    segments = path.split("/")
    last = segments[-1]
    
    # Remove extension
    last = re.sub(r"\.[^.]+$", "", last)
    
    # Convert kebab/snake case to title
    last = re.sub(r"[-_]", " ", last)
    
    return last.title()
