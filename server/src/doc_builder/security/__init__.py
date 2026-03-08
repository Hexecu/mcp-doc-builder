"""
Security module for authentication and origin validation.
"""

from doc_builder.security.auth import AuthMiddleware, get_auth_middleware
from doc_builder.security.origin import OriginMiddleware, is_origin_allowed

__all__ = [
    "AuthMiddleware",
    "get_auth_middleware",
    "OriginMiddleware",
    "is_origin_allowed",
]
