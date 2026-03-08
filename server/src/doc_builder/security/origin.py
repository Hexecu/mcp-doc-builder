"""
Origin validation for MCP server.
"""

import fnmatch
import logging
from urllib.parse import urlparse

from doc_builder.config import get_settings

logger = logging.getLogger(__name__)


def is_origin_allowed(origin: str) -> bool:
    """
    Check if an origin is allowed.
    
    Args:
        origin: Origin URL or hostname
        
    Returns:
        True if origin is allowed
    """
    settings = get_settings()
    allowed = settings.allowed_origins_list

    if not allowed:
        return True  # No restrictions if no origins configured

    # Parse origin
    if "://" in origin:
        parsed = urlparse(origin)
        host = parsed.hostname or origin
    else:
        host = origin

    # Check against allowed patterns
    for pattern in allowed:
        # Support wildcards
        if fnmatch.fnmatch(host, pattern):
            return True

        # Support localhost variants
        if pattern in ("localhost", "127.0.0.1") and host in ("localhost", "127.0.0.1"):
            return True

    logger.warning(f"Origin not allowed: {origin}")
    return False


class OriginMiddleware:
    """
    ASGI middleware for origin validation.
    """

    def __init__(self, app):
        self.app = app
        self.settings = get_settings()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Get Origin header
        headers = dict(scope.get("headers", []))
        origin = headers.get(b"origin", b"").decode()

        if origin and not is_origin_allowed(origin):
            from starlette.responses import JSONResponse
            response = JSONResponse(
                {"error": "Origin not allowed"},
                status_code=403,
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
