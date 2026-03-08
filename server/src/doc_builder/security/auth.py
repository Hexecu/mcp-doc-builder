"""
Authentication middleware for MCP server.
"""

import logging
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from doc_builder.config import get_settings

logger = logging.getLogger(__name__)


def get_auth_middleware() -> Callable | None:
    """
    Get authentication middleware if token is configured.
    
    Returns:
        Middleware function or None if auth is disabled
    """
    settings = get_settings()

    if not settings.doc_mcp_token:
        logger.info("Authentication disabled (no token configured)")
        return None

    async def auth_middleware(request: Request, call_next):
        """Verify bearer token in Authorization header."""
        # Skip auth for health check
        if request.url.path == "/health":
            return await call_next(request)

        # Get Authorization header
        auth_header = request.headers.get("Authorization", "")

        if not auth_header.startswith("Bearer "):
            return JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
            )

        token = auth_header[7:]  # Remove "Bearer " prefix

        if token != settings.doc_mcp_token:
            logger.warning(f"Invalid auth token from {request.client.host}")
            return JSONResponse(
                {"error": "Invalid token"},
                status_code=401,
            )

        return await call_next(request)

    return auth_middleware


class AuthMiddleware:
    """
    ASGI middleware for authentication.
    """

    def __init__(self, app):
        self.app = app
        self.settings = get_settings()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Skip auth if no token configured
        if not self.settings.doc_mcp_token:
            await self.app(scope, receive, send)
            return

        # Skip auth for health check
        path = scope.get("path", "")
        if path == "/health":
            await self.app(scope, receive, send)
            return

        # Check Authorization header
        headers = dict(scope.get("headers", []))
        auth_header = headers.get(b"authorization", b"").decode()

        if not auth_header.startswith("Bearer "):
            response = JSONResponse(
                {"error": "Missing or invalid Authorization header"},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        token = auth_header[7:]

        if token != self.settings.doc_mcp_token:
            logger.warning("Invalid auth token")
            response = JSONResponse(
                {"error": "Invalid token"},
                status_code=401,
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)
