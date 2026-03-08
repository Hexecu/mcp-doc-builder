"""
CLI command for indexing documentation.
Supports interactive mode, presets, and custom URLs.
"""

import argparse
import asyncio
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from doc_builder.cli.presets import (
    DOC_PRESETS,
    DocPreset,
    estimate_time_minutes,
    get_all_presets,
    get_preset,
    get_preset_keys,
)

console = Console()

# State file for tracking indexing progress and errors
STATE_FILE = Path.home() / ".doc-builder" / "index_state.json"

# Timeout for indexing (1 hour)
INDEX_TIMEOUT_SECONDS = 3600


class IndexState:
    """Manages indexing state for progress tracking and error recovery."""

    def __init__(self):
        self.state: dict[str, Any] = self._load_state()

    def _load_state(self) -> dict[str, Any]:
        """Load state from file."""
        if STATE_FILE.exists():
            try:
                return json.loads(STATE_FILE.read_text())
            except Exception:
                pass
        return {"sources": {}, "failed_pages": {}}

    def _save_state(self) -> None:
        """Save state to file."""
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.state, indent=2, default=str))

    def start_source(self, source_id: str, name: str, url: str) -> None:
        """Mark a source as started."""
        self.state["sources"][source_id] = {
            "name": name,
            "url": url,
            "status": "indexing",
            "started_at": datetime.now().isoformat(),
            "pages_crawled": 0,
            "pages_failed": 0,
            "chunks_created": 0,
            "concepts_extracted": 0,
        }
        self._save_state()

    def update_source(self, source_id: str, **kwargs) -> None:
        """Update source progress."""
        if source_id in self.state["sources"]:
            self.state["sources"][source_id].update(kwargs)
            self._save_state()

    def complete_source(self, source_id: str, stats: dict) -> None:
        """Mark a source as completed."""
        if source_id in self.state["sources"]:
            self.state["sources"][source_id].update({
                "status": "completed",
                "completed_at": datetime.now().isoformat(),
                **stats,
            })
            self._save_state()

    def fail_source(self, source_id: str, error: str) -> None:
        """Mark a source as failed."""
        if source_id in self.state["sources"]:
            self.state["sources"][source_id].update({
                "status": "failed",
                "error": error,
                "failed_at": datetime.now().isoformat(),
            })
            self._save_state()

    def add_failed_page(self, source_id: str, url: str, error: str) -> None:
        """Record a failed page."""
        if source_id not in self.state["failed_pages"]:
            self.state["failed_pages"][source_id] = []
        self.state["failed_pages"][source_id].append({
            "url": url,
            "error": error,
            "timestamp": datetime.now().isoformat(),
        })
        self._save_state()

    def get_failed_pages(self, source_id: str | None = None) -> dict:
        """Get failed pages, optionally filtered by source."""
        if source_id:
            return {source_id: self.state["failed_pages"].get(source_id, [])}
        return self.state["failed_pages"]

    def clear_failed_pages(self, source_id: str) -> None:
        """Clear failed pages for a source."""
        if source_id in self.state["failed_pages"]:
            del self.state["failed_pages"][source_id]
            self._save_state()

    def get_incomplete_sources(self) -> list[dict]:
        """Get sources that were interrupted."""
        incomplete = []
        for source_id, info in self.state["sources"].items():
            if info.get("status") == "indexing":
                incomplete.append({"source_id": source_id, **info})
        return incomplete


def print_header():
    """Print command header."""
    console.print()
    console.print(Panel.fit(
        "[bold blue]MCP Doc Builder - Documentation Indexer[/bold blue]\n"
        "Index documentation for semantic search",
        border_style="blue",
    ))
    console.print()


def list_presets():
    """Display available presets."""
    table = Table(title="Available Documentation Presets")
    table.add_column("Key", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("URL")
    table.add_column("Depth", justify="center")
    table.add_column("Est. Pages", justify="right")

    for preset in get_all_presets():
        table.add_row(
            preset.key,
            preset.name,
            preset.url[:40] + "..." if len(preset.url) > 40 else preset.url,
            str(preset.max_depth),
            f"~{preset.estimated_pages}",
        )

    console.print(table)
    console.print()
    console.print("[dim]Use: doc-mcp-index --preset langchain,nextjs[/dim]")
    console.print("[dim]Or:  doc-mcp-index --interactive[/dim]")


def show_failed_pages(state: IndexState, source_id: str | None = None):
    """Display failed pages."""
    failed = state.get_failed_pages(source_id)

    if not any(failed.values()):
        console.print("[green]No failed pages recorded.[/green]")
        return

    for sid, pages in failed.items():
        if not pages:
            continue

        source_info = state.state["sources"].get(sid, {})
        source_name = source_info.get("name", sid)

        table = Table(title=f"Failed Pages: {source_name}")
        table.add_column("URL", style="cyan")
        table.add_column("Error", style="red")
        table.add_column("Time")

        for page in pages[-20:]:  # Show last 20
            table.add_row(
                page["url"][:60] + "..." if len(page["url"]) > 60 else page["url"],
                page["error"][:40] + "..." if len(page["error"]) > 40 else page["error"],
                page["timestamp"][:19],
            )

        console.print(table)
        console.print(f"[dim]Total failed: {len(pages)}[/dim]")
        console.print()


def interactive_select() -> list[tuple[str, str, int]]:
    """
    Interactive preset selection with multi-select.
    
    Returns:
        List of (name, url, max_depth) tuples
    """
    presets = get_all_presets()
    selected: set[str] = set()

    while True:
        console.clear()
        print_header()
        console.print("[bold]Select documentation to index:[/bold]\n")

        # Display presets with selection status
        for i, preset in enumerate(presets, 1):
            checkbox = "[green]✓[/green]" if preset.key in selected else "[ ]"
            console.print(
                f"  {checkbox} {i:2}. [cyan]{preset.name:25}[/cyan] "
                f"[dim]{preset.url[:35]}...[/dim]"
            )

        console.print()
        console.print(f"  [bold]A[/bold].  Add custom URL")
        console.print(f"  [bold]C[/bold].  Clear selection")
        console.print(f"  [bold]Enter[/bold] Confirm and start indexing")
        console.print()

        # Show selection summary
        if selected:
            selected_presets = [p for p in presets if p.key in selected]
            min_time, max_time = estimate_time_minutes(selected_presets)
            console.print(
                f"[green]Selected: {len(selected)} sources | "
                f"Estimated time: {min_time}-{max_time} minutes[/green]"
            )
        else:
            console.print("[dim]No sources selected[/dim]")

        console.print()
        choice = Prompt.ask(
            "Enter number to toggle, or command",
            default="",
        ).strip().upper()

        if choice == "":
            # Confirm selection
            if not selected:
                console.print("[yellow]Please select at least one source.[/yellow]")
                Prompt.ask("Press Enter to continue")
                continue
            break

        elif choice == "A":
            # Add custom URL
            custom = add_custom_url()
            if custom:
                return list(get_selected_docs(selected, presets)) + [custom]

        elif choice == "C":
            selected.clear()

        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                key = presets[idx].key
                if key in selected:
                    selected.remove(key)
                else:
                    selected.add(key)

    return list(get_selected_docs(selected, presets))


def get_selected_docs(
    selected: set[str],
    presets: list[DocPreset],
) -> list[tuple[str, str, int]]:
    """Convert selected keys to doc tuples."""
    result = []
    for preset in presets:
        if preset.key in selected:
            result.append((preset.name, preset.url, preset.max_depth))
    return result


def add_custom_url() -> tuple[str, str, int] | None:
    """Prompt for custom URL."""
    console.print()
    console.print("[bold]Add Custom Documentation URL[/bold]")
    console.print()

    url = Prompt.ask("Documentation URL").strip()
    if not url:
        return None

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    name = Prompt.ask("Name for this documentation")
    if not name:
        return None

    depth = IntPrompt.ask("Max crawl depth", default=2)
    depth = max(1, min(5, depth))

    return (name, url, depth)


async def index_documentation(
    docs: list[tuple[str, str, int]],
    state: IndexState,
    retry_failed: bool = False,
) -> dict[str, Any]:
    """
    Index multiple documentation sources.
    
    Args:
        docs: List of (name, url, max_depth) tuples
        state: IndexState for progress tracking
        retry_failed: Whether to retry previously failed pages
        
    Returns:
        Summary statistics
    """
    from doc_builder.config import get_settings
    from doc_builder.crawler import CrawlResult, Spider
    from doc_builder.kg import get_neo4j_client, get_repository
    from doc_builder.kg.apply_schema import apply_schema
    from doc_builder.ontology import extract_ontology, get_linker, store_page_metatags
    from doc_builder.utils import content_hash, extract_domain, generate_id, truncate_text
    from doc_builder.vector import get_indexer

    settings = get_settings()
    
    # Connect to Neo4j
    console.print("[yellow]Connecting to Neo4j...[/yellow]")
    client = get_neo4j_client()
    await client.connect()
    
    # Apply schema
    console.print("[yellow]Applying database schema...[/yellow]")
    await apply_schema()
    
    repo = get_repository()
    indexer = get_indexer()
    linker = get_linker()

    total_stats = {
        "sources_completed": 0,
        "sources_failed": 0,
        "total_pages": 0,
        "total_chunks": 0,
        "total_concepts": 0,
        "failed_pages": 0,
    }

    # Setup signal handler for graceful interruption
    interrupted = False

    def handle_interrupt(signum, frame):
        nonlocal interrupted
        interrupted = True
        console.print("\n[yellow]Interrupting... saving progress...[/yellow]")

    signal.signal(signal.SIGINT, handle_interrupt)

    # Process each documentation source
    for doc_name, doc_url, max_depth in docs:
        if interrupted:
            break

        console.print()
        console.print(Panel(f"[bold]Indexing: {doc_name}[/bold]\n{doc_url}"))

        # Create source
        domain = extract_domain(doc_url)
        source = await repo.create_source(
            root_url=doc_url,
            name=doc_name,
            domain=domain,
            description=f"Documentation from {domain}",
        )
        source_id = source["id"]

        state.start_source(source_id, doc_name, doc_url)

        # Update status to crawling
        await repo.update_source_status(source_id, "crawling")

        # Create crawl job
        job = await repo.create_crawl_job(source_id)

        # Stats for this source
        stats = {
            "pages_crawled": 0,
            "pages_failed": 0,
            "chunks_created": 0,
            "concepts_extracted": 0,
        }

        # Collected pages for relationship building
        crawled_pages = []

        # Progress display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            # Create progress task (estimate based on preset or default)
            preset = get_preset(doc_name.lower().replace(" ", "-"))
            estimated = preset.estimated_pages if preset else 100
            task = progress.add_task(f"Crawling {doc_name}", total=estimated)

            # Initialize spider
            spider = Spider(
                root_url=doc_url,
                doc_name=doc_name,
                max_depth=max_depth,
                max_pages=settings.crawler_max_pages,
                max_concurrent=settings.crawler_max_concurrent,
                rate_limit=settings.crawler_rate_limit,
            )

            async def process_page(result: CrawlResult):
                nonlocal stats

                if interrupted:
                    return

                progress.update(task, advance=1)

                if not result.success or not result.page:
                    stats["pages_failed"] += 1
                    total_stats["failed_pages"] += 1
                    if result.error:
                        state.add_failed_page(source_id, result.url, result.error)
                    return

                page = result.page
                crawled_pages.append(page)
                stats["pages_crawled"] += 1

                try:
                    # Store page
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

                    # Index content
                    index_stats = await indexer.index_page(
                        page_id=page_id,
                        content=page.content,
                        heading_context=page.title,
                    )
                    stats["chunks_created"] += index_stats.chunks_created

                    # Extract ontology
                    ontology_result = await extract_ontology(
                        source_id=source_id,
                        source_name=doc_name,
                        page_title=page.title,
                        page_url=page.url,
                        content=page.content,
                    )
                    stats["concepts_extracted"] += ontology_result.concepts_extracted

                    # Update progress description
                    progress.update(
                        task,
                        description=f"Crawling {doc_name} | {stats['chunks_created']} chunks",
                    )

                    # Update state periodically
                    state.update_source(source_id, **stats)

                except Exception as e:
                    stats["pages_failed"] += 1
                    state.add_failed_page(source_id, page.url, str(e))

            # Register callback - wrap in sync function
            def on_page_callback(r: CrawlResult) -> None:
                asyncio.create_task(process_page(r))

            spider.on_page(on_page_callback)

            # Run crawler with timeout
            try:
                crawl_task = asyncio.create_task(spider.crawl_all())
                await asyncio.wait_for(crawl_task, timeout=INDEX_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                console.print(f"[yellow]Timeout reached for {doc_name}[/yellow]")
                state.fail_source(source_id, "Timeout after 1 hour")
                spider.stop()
            except Exception as e:
                console.print(f"[red]Crawl error: {e}[/red]")
                state.fail_source(source_id, str(e))

            # Update progress to actual count
            progress.update(task, completed=stats["pages_crawled"], total=stats["pages_crawled"])

        # Build relationships if not interrupted
        if not interrupted and crawled_pages:
            console.print("[dim]Building relationships...[/dim]")
            try:
                await linker.build_source_graph(source_id, crawled_pages)
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to build relationships: {e}[/yellow]")

        # Update final status
        if not interrupted:
            await repo.update_source_status(
                source_id,
                "completed",
                total_pages=stats["pages_crawled"],
            )
            await repo.update_crawl_job(job["id"], status="completed")
            state.complete_source(source_id, stats)
            total_stats["sources_completed"] += 1
        else:
            state.update_source(source_id, status="interrupted", **stats)

        # Accumulate totals
        total_stats["total_pages"] += stats["pages_crawled"]
        total_stats["total_chunks"] += stats["chunks_created"]
        total_stats["total_concepts"] += stats["concepts_extracted"]

        # Show source summary
        console.print(
            f"[green]✓ {doc_name}:[/green] "
            f"{stats['pages_crawled']} pages, "
            f"{stats['chunks_created']} chunks, "
            f"{stats['concepts_extracted']} concepts"
        )
        if stats["pages_failed"] > 0:
            console.print(
                f"  [yellow]⚠ {stats['pages_failed']} pages failed[/yellow]"
            )

    # Close Neo4j connection
    from doc_builder.kg import close_neo4j_client
    await close_neo4j_client()

    return total_stats


async def retry_failed_pages(state: IndexState, source_id: str) -> dict:
    """Retry indexing failed pages for a source."""
    failed = state.get_failed_pages(source_id).get(source_id, [])

    if not failed:
        console.print("[green]No failed pages to retry.[/green]")
        return {"retried": 0, "success": 0, "failed": 0}

    console.print(f"[yellow]Retrying {len(failed)} failed pages...[/yellow]")

    from doc_builder.kg import get_neo4j_client, get_repository
    from doc_builder.ontology import extract_ontology, store_page_metatags
    from doc_builder.utils import content_hash, truncate_text
    from doc_builder.vector import get_indexer
    from doc_builder.crawler.parser import HTMLParser

    import aiohttp

    client = get_neo4j_client()
    await client.connect()

    repo = get_repository()
    indexer = get_indexer()
    parser = HTMLParser()

    source = await repo.get_source(source_id)
    if not source:
        console.print(f"[red]Source not found: {source_id}[/red]")
        return {"retried": 0, "success": 0, "failed": 0}

    stats = {"retried": len(failed), "success": 0, "failed": 0}
    new_failed = []

    async with aiohttp.ClientSession() as session:
        for page_info in failed:
            url = page_info["url"]
            try:
                timeout = aiohttp.ClientTimeout(total=30)
                async with session.get(url, timeout=timeout) as response:
                    if response.status != 200:
                        raise Exception(f"HTTP {response.status}")
                    html = await response.text()

                page = parser.parse(html, url)

                # Store page
                page_data = await repo.upsert_page(
                    source_id=source_id,
                    url=page.url,
                    title=page.title,
                    description=page.description,
                    content_preview=truncate_text(page.content, 500),
                    content_hash=content_hash(page.content),
                    depth=0,  # Unknown depth for retry
                    language=page.language,
                    word_count=page.word_count,
                )
                page_id = page_data["id"]

                # Store metatags
                await store_page_metatags(page_id, page.metatags)

                # Index content
                await indexer.index_page(
                    page_id=page_id,
                    content=page.content,
                    heading_context=page.title,
                )

                # Extract ontology
                await extract_ontology(
                    source_id=source_id,
                    source_name=source["name"],
                    page_title=page.title,
                    page_url=page.url,
                    content=page.content,
                )

                stats["success"] += 1
                console.print(f"[green]✓[/green] {url[:60]}")

            except Exception as e:
                stats["failed"] += 1
                new_failed.append({
                    "url": url,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                })
                console.print(f"[red]✗[/red] {url[:60]} - {e}")

    # Update failed pages list
    state.state["failed_pages"][source_id] = new_failed
    state._save_state()

    from doc_builder.kg import close_neo4j_client
    await close_neo4j_client()

    return stats


def show_status(state: IndexState):
    """Show indexing status for all sources."""
    sources = state.state.get("sources", {})

    if not sources:
        console.print("[dim]No indexing history found.[/dim]")
        return

    table = Table(title="Indexing History")
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("Pages", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Date")

    for source_id, info in sources.items():
        status = info.get("status", "unknown")
        status_style = {
            "completed": "[green]completed[/green]",
            "failed": "[red]failed[/red]",
            "indexing": "[yellow]indexing[/yellow]",
            "interrupted": "[yellow]interrupted[/yellow]",
        }.get(status, status)

        failed_count = len(state.state.get("failed_pages", {}).get(source_id, []))

        table.add_row(
            info.get("name", "Unknown"),
            status_style,
            str(info.get("pages_crawled", 0)),
            str(info.get("chunks_created", 0)),
            str(failed_count) if failed_count else "-",
            info.get("completed_at", info.get("started_at", ""))[:10],
        )

    console.print(table)


def main():
    """Main entry point for doc-mcp-index command."""
    # Load environment variables from .env file
    from pathlib import Path
    from dotenv import load_dotenv
    
    # Try to find .env file in project directory
    project_dir = Path(__file__).parent.parent.parent.parent.parent
    env_file = project_dir / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        # Try current directory
        load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Index documentation for MCP Doc Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  doc-mcp-index --interactive          Interactive preset selection
  doc-mcp-index --preset langchain     Index LangChain docs
  doc-mcp-index --preset langchain,nextjs  Index multiple presets
  doc-mcp-index --url https://docs.example.com --name "My Docs"
  doc-mcp-index --list-presets         Show available presets
  doc-mcp-index --status               Show indexing history
  doc-mcp-index --show-failed          Show failed pages
  doc-mcp-index --retry <source_id>    Retry failed pages
        """,
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode with multi-select",
    )
    parser.add_argument(
        "--preset", "-p",
        type=str,
        help="Preset name(s), comma-separated (e.g., langchain,nextjs)",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Custom documentation URL",
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name for custom documentation (required with --url)",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=2,
        help="Max crawl depth (default: 2)",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available documentation presets",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show indexing status and history",
    )
    parser.add_argument(
        "--show-failed",
        action="store_true",
        help="Show failed pages",
    )
    parser.add_argument(
        "--retry",
        type=str,
        metavar="SOURCE_ID",
        help="Retry failed pages for a source",
    )
    parser.add_argument(
        "--clear-failed",
        type=str,
        metavar="SOURCE_ID",
        help="Clear failed pages for a source",
    )

    args = parser.parse_args()
    state = IndexState()

    # Handle info commands
    if args.list_presets:
        print_header()
        list_presets()
        return

    if args.status:
        print_header()
        show_status(state)
        return

    if args.show_failed:
        print_header()
        show_failed_pages(state)
        return

    if args.clear_failed:
        state.clear_failed_pages(args.clear_failed)
        console.print(f"[green]Cleared failed pages for {args.clear_failed}[/green]")
        return

    if args.retry:
        print_header()
        stats = asyncio.run(retry_failed_pages(state, args.retry))
        console.print()
        console.print(
            f"Retry complete: {stats['success']} success, "
            f"{stats['failed']} still failed"
        )
        return

    # Collect docs to index
    docs: list[tuple[str, str, int]] = []

    if args.interactive:
        print_header()
        docs = interactive_select()

    elif args.preset:
        print_header()
        preset_keys = [k.strip() for k in args.preset.split(",")]
        for key in preset_keys:
            preset = get_preset(key)
            if preset:
                docs.append((preset.name, preset.url, preset.max_depth))
            else:
                console.print(f"[yellow]Unknown preset: {key}[/yellow]")

    elif args.url:
        if not args.name:
            console.print("[red]Error: --name is required with --url[/red]")
            sys.exit(1)
        print_header()
        docs.append((args.name, args.url, args.depth))

    else:
        parser.print_help()
        return

    if not docs:
        console.print("[yellow]No documentation selected.[/yellow]")
        return

    # Confirm
    console.print()
    console.print("[bold]Documentation to index:[/bold]")
    for name, url, depth in docs:
        console.print(f"  • {name} (depth={depth})")
        console.print(f"    [dim]{url}[/dim]")

    console.print()
    if not Confirm.ask("Start indexing?", default=True):
        console.print("[dim]Cancelled.[/dim]")
        return

    # Run indexing
    console.print()
    try:
        stats = asyncio.run(index_documentation(docs, state))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Progress has been saved.[/yellow]")
        console.print("Run [cyan]doc-mcp-index --status[/cyan] to see progress.")
        console.print("Run [cyan]doc-mcp-index --show-failed[/cyan] to see failed pages.")
        return

    # Show summary
    console.print()
    console.print(Panel.fit(
        f"[bold green]Indexing Complete![/bold green]\n\n"
        f"Sources: {stats['sources_completed']} completed\n"
        f"Pages: {stats['total_pages']}\n"
        f"Chunks: {stats['total_chunks']}\n"
        f"Concepts: {stats['total_concepts']}\n"
        f"Failed pages: {stats['failed_pages']}",
        border_style="green",
    ))

    if stats["failed_pages"] > 0:
        console.print()
        console.print(
            "[yellow]Some pages failed. Run [cyan]doc-mcp-index --show-failed[/cyan] "
            "to see details.[/yellow]"
        )
        console.print(
            "[yellow]Run [cyan]doc-mcp-index --retry <source_id>[/cyan] "
            "to retry failed pages.[/yellow]"
        )


if __name__ == "__main__":
    main()
