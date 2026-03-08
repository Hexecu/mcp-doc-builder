"""
MCP module for server tools, resources, and prompts.
"""

from doc_builder.mcp.prompts import PROMPTS, get_prompts, render_prompt
from doc_builder.mcp.resources import RESOURCES, get_resources
from doc_builder.mcp.tools import TOOLS, get_tools

__all__ = [
    "TOOLS",
    "get_tools",
    "RESOURCES",
    "get_resources",
    "PROMPTS",
    "get_prompts",
    "render_prompt",
]
