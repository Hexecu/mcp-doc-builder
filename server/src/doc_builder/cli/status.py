"""
Health check and status command for MCP Doc Builder.
"""

import asyncio
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def check_neo4j() -> tuple[bool, str]:
    """Check Neo4j connection."""
    try:
        from doc_builder.kg import get_neo4j_client
        
        client = get_neo4j_client()
        health = await client.health_check()
        
        if health.get("status") == "healthy":
            return True, "Connected"
        else:
            return False, health.get("error", "Unknown error")
    except Exception as e:
        return False, str(e)


async def check_llm() -> tuple[bool, str]:
    """Check LLM connection."""
    try:
        from doc_builder.llm import get_llm_client
        
        client = get_llm_client()
        health = await client.health_check()
        
        if health.get("status") == "healthy":
            return True, f"Connected ({health.get('model', 'unknown')})"
        else:
            return False, health.get("error", "Unknown error")
    except Exception as e:
        return False, str(e)


async def check_embeddings() -> tuple[bool, str]:
    """Check embedding service."""
    try:
        from doc_builder.vector import get_embedder
        
        embedder = get_embedder()
        health = await embedder.health_check()
        
        if health.get("status") == "healthy":
            return True, f"Working ({health.get('dimensions', '?')} dims)"
        else:
            return False, health.get("error", "Unknown error")
    except Exception as e:
        return False, str(e)


async def get_source_stats() -> list[dict]:
    """Get statistics for all sources."""
    try:
        from doc_builder.kg import get_neo4j_client, get_repository
        
        # Ensure connection
        client = get_neo4j_client()
        await client.connect()
        
        repo = get_repository()
        sources = await repo.list_sources()
        
        stats = []
        for source in sources:
            source_stats = await repo.get_source_stats(source["id"])
            stats.append({
                "name": source.get("name", "Unknown"),
                "status": source.get("status", "unknown"),
                "pages": source_stats.get("page_count", 0),
                "chunks": source_stats.get("chunk_count", 0),
                "concepts": source_stats.get("concept_count", 0),
            })
        
        return stats
    except Exception as e:
        return []


async def run_doctor() -> bool:
    """Run comprehensive health checks."""
    console.print()
    console.print(Panel.fit(
        "[bold blue]MCP Doc Builder - Health Check[/bold blue]",
        border_style="blue",
    ))
    console.print()

    all_healthy = True

    # Create status table
    table = Table(title="Service Status")
    table.add_column("Service", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Details")

    # Check Neo4j
    console.print("[yellow]Checking Neo4j...[/yellow]", end=" ")
    neo4j_ok, neo4j_msg = await check_neo4j()
    status = "[green]OK[/green]" if neo4j_ok else "[red]FAIL[/red]"
    table.add_row("Neo4j", status, neo4j_msg)
    console.print(status)
    if not neo4j_ok:
        all_healthy = False

    # Check LLM
    console.print("[yellow]Checking LLM...[/yellow]", end=" ")
    llm_ok, llm_msg = await check_llm()
    status = "[green]OK[/green]" if llm_ok else "[red]FAIL[/red]"
    table.add_row("LLM", status, llm_msg)
    console.print(status)
    if not llm_ok:
        all_healthy = False

    # Check Embeddings
    console.print("[yellow]Checking Embeddings...[/yellow]", end=" ")
    embed_ok, embed_msg = await check_embeddings()
    status = "[green]OK[/green]" if embed_ok else "[red]FAIL[/red]"
    table.add_row("Embeddings", status, embed_msg)
    console.print(status)
    if not embed_ok:
        all_healthy = False

    console.print()
    console.print(table)

    # Get source statistics
    if neo4j_ok:
        console.print()
        stats = await get_source_stats()
        
        if stats:
            stats_table = Table(title="Documentation Sources")
            stats_table.add_column("Name", style="cyan")
            stats_table.add_column("Status")
            stats_table.add_column("Pages", justify="right")
            stats_table.add_column("Chunks", justify="right")
            stats_table.add_column("Concepts", justify="right")

            for source in stats:
                status_style = "green" if source["status"] == "completed" else "yellow"
                stats_table.add_row(
                    source["name"],
                    f"[{status_style}]{source['status']}[/{status_style}]",
                    str(source["pages"]),
                    str(source["chunks"]),
                    str(source["concepts"]),
                )

            console.print(stats_table)
        else:
            console.print("[dim]No documentation sources indexed yet.[/dim]")

    # Summary
    console.print()
    if all_healthy:
        console.print(Panel.fit(
            "[bold green]All systems operational![/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel.fit(
            "[bold red]Some services are not working.[/bold red]\n"
            "Run `doc-mcp-setup` to configure.",
            border_style="red",
        ))

    return all_healthy


async def run_status() -> None:
    """Show basic status."""
    from doc_builder.config import get_settings
    
    settings = get_settings()
    
    console.print()
    console.print("[bold]MCP Doc Builder Status[/bold]")
    console.print()
    console.print(f"  LLM Mode: {settings.llm_mode}")
    console.print(f"  Neo4j URI: {settings.neo4j_uri}")
    console.print(f"  Server: {settings.mcp_host}:{settings.mcp_port}")
    console.print()


def main():
    """Main entry point for status command."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MCP Doc Builder Status")
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run comprehensive health checks",
    )
    
    args = parser.parse_args()
    
    if args.doctor:
        healthy = asyncio.run(run_doctor())
        sys.exit(0 if healthy else 1)
    else:
        asyncio.run(run_status())


if __name__ == "__main__":
    main()
