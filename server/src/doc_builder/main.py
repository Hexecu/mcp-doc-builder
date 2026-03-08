"""
Main entry point for the MCP Doc Builder server.
Supports both STDIO and HTTP transport modes.
"""

import argparse
import asyncio
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    GetPromptResult,
    Prompt,
    PromptArgument,
    PromptMessage,
    Resource,
    TextContent,
    Tool,
)

from doc_builder.config import get_settings
from doc_builder.kg import close_neo4j_client, get_neo4j_client
from doc_builder.kg.apply_schema import apply_schema
from doc_builder.mcp import PROMPTS, RESOURCES, TOOLS, render_prompt
from doc_builder.utils import safe_json_dumps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("doc-builder")

    # ─────────────────────────────────────────────────────────────────────────
    # Tool Handlers
    # ─────────────────────────────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """List available tools."""
        tools = []
        for name, config in TOOLS.items():
            tools.append(Tool(
                name=name,
                description=config["description"],
                inputSchema={
                    "type": "object",
                    "properties": config.get("parameters", {}),
                    "required": config.get("required", []),
                },
            ))
        return tools

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute a tool and return results."""
        if name not in TOOLS:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        tool = TOOLS[name]
        func = tool["function"]

        try:
            result = await func(**arguments)
            return [TextContent(type="text", text=safe_json_dumps(result, indent=2))]
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    # ─────────────────────────────────────────────────────────────────────────
    # Resource Handlers
    # ─────────────────────────────────────────────────────────────────────────

    @server.list_resources()
    async def list_resources() -> list[Resource]:
        """List available resources."""
        resources = []
        for uri_template, config in RESOURCES.items():
            resources.append(Resource(
                uri=uri_template,
                name=uri_template,
                description=config["description"],
                mimeType="application/json",
            ))
        return resources

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        """Read a resource by URI."""
        # Parse the URI and match to a handler
        # URI format: doc://sources/{source_id}/pages
        
        import re
        
        for uri_template, config in RESOURCES.items():
            # Convert template to regex
            pattern = uri_template.replace("{", "(?P<").replace("}", ">[^/]+)")
            pattern = f"^{pattern}$"
            
            match = re.match(pattern, uri)
            if match:
                handler = config["handler"]
                params = match.groupdict()
                
                try:
                    result = await handler(**params)
                    return safe_json_dumps(result, indent=2)
                except Exception as e:
                    logger.error(f"Resource read failed: {e}")
                    return safe_json_dumps({"error": str(e)})
        
        return safe_json_dumps({"error": f"Unknown resource: {uri}"})

    # ─────────────────────────────────────────────────────────────────────────
    # Prompt Handlers
    # ─────────────────────────────────────────────────────────────────────────

    @server.list_prompts()
    async def list_prompts() -> list[Prompt]:
        """List available prompts."""
        prompts = []
        for name, config in PROMPTS.items():
            prompts.append(Prompt(
                name=name,
                description=config["description"],
                arguments=[
                    PromptArgument(
                        name=arg["name"],
                        description=arg.get("description", ""),
                        required=arg.get("required", False),
                    )
                    for arg in config.get("arguments", [])
                ],
            ))
        return prompts

    @server.get_prompt()
    async def get_prompt(name: str, arguments: dict[str, Any] | None = None) -> GetPromptResult:
        """Get a prompt by name."""
        if name not in PROMPTS:
            return GetPromptResult(
                description=f"Unknown prompt: {name}",
                messages=[],
            )

        content = render_prompt(name, arguments)
        
        return GetPromptResult(
            description=PROMPTS[name]["description"],
            messages=[
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=content),
                )
            ],
        )

    return server


async def initialize() -> None:
    """Initialize server dependencies."""
    settings = get_settings()
    
    logger.info(f"Initializing Doc Builder MCP Server")
    logger.info(f"LLM Mode: {settings.llm_mode}")
    logger.info(f"Neo4j URI: {settings.neo4j_uri}")

    # Connect to Neo4j
    client = get_neo4j_client()
    await client.connect()

    # Apply schema
    try:
        await apply_schema()
        logger.info("Schema applied successfully")
    except Exception as e:
        logger.warning(f"Schema application warning: {e}")


async def cleanup() -> None:
    """Cleanup server resources."""
    await close_neo4j_client()
    logger.info("Server cleanup complete")


async def run_stdio() -> None:
    """Run server in STDIO mode."""
    server = create_server()

    try:
        await initialize()
        
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await cleanup()


async def run_http(host: str, port: int) -> None:
    """Run server in HTTP mode."""
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.routing import Route
    from starlette.responses import JSONResponse

    server = create_server()
    await initialize()

    sse = SseServerTransport("/messages/")

    async def handle_sse(request):
        async with sse.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )

    async def handle_messages(request):
        await sse.handle_post_message(request.scope, request.receive, request._send)

    async def health_check(request):
        return JSONResponse({"status": "healthy"})

    routes = [
        Route("/sse", endpoint=handle_sse),
        Route("/messages/", endpoint=handle_messages, methods=["POST"]),
        Route("/health", endpoint=health_check),
    ]

    app = Starlette(routes=routes, on_shutdown=[cleanup])

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server_instance = uvicorn.Server(config)
    await server_instance.serve()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MCP Doc Builder - Documentation Indexing Server"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode (default: stdio)",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP host (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="HTTP port (default: 8001)",
    )

    args = parser.parse_args()

    if args.transport == "stdio":
        asyncio.run(run_stdio())
    else:
        asyncio.run(run_http(args.host, args.port))


if __name__ == "__main__":
    main()
