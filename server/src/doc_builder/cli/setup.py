"""
Interactive setup wizard for MCP Doc Builder.
"""

import asyncio
import os
import secrets
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

console = Console()


def print_header():
    """Print setup header."""
    console.print()
    console.print(Panel.fit(
        "[bold blue]MCP Doc Builder Setup[/bold blue]\n"
        "Intelligent Documentation Indexing & Search",
        border_style="blue",
    ))
    console.print()


def select_option(prompt: str, options: list[str], default: int = 1) -> int:
    """Display numbered options and get selection."""
    console.print(f"\n[bold]{prompt}[/bold]")
    for i, option in enumerate(options, 1):
        marker = "[green]>[/green]" if i == default else " "
        console.print(f"  {marker} {i}. {option}")

    while True:
        try:
            choice = IntPrompt.ask(
                "Select option",
                default=default,
                show_default=True,
            )
            if 1 <= choice <= len(options):
                return choice
            console.print("[red]Invalid selection[/red]")
        except KeyboardInterrupt:
            console.print("\n[yellow]Setup cancelled[/yellow]")
            sys.exit(0)


async def test_neo4j_connection(uri: str, user: str, password: str) -> bool:
    """Test Neo4j connection."""
    try:
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        await driver.verify_connectivity()
        await driver.close()
        return True
    except Exception as e:
        console.print(f"[red]Connection failed: {e}[/red]")
        return False


async def test_llm_connection(mode: str, config: dict) -> bool:
    """Test LLM connection."""
    try:
        import litellm

        if mode == "litellm" and config.get("base_url"):
            response = await litellm.acompletion(
                model=config.get("model", "gemini-2.5-flash"),
                messages=[{"role": "user", "content": "Say ok"}],
                api_base=config["base_url"],
                api_key=config.get("api_key"),
                max_tokens=10,
            )
        else:
            response = await litellm.acompletion(
                model=f"gemini/{config.get('model', 'gemini-2.5-flash')}",
                messages=[{"role": "user", "content": "Say ok"}],
                api_key=config.get("api_key"),
                max_tokens=10,
            )
        return True
    except Exception as e:
        console.print(f"[red]Connection failed: {e}[/red]")
        return False


def configure_neo4j() -> dict:
    """Configure Neo4j settings."""
    console.print("\n[bold cyan]Neo4j Configuration[/bold cyan]")

    choice = select_option(
        "Where is your Neo4j database?",
        [
            "Local Docker (will be started with docker-compose)",
            "Remote Neo4j (Neo4j Aura or self-hosted)",
        ],
        default=1,
    )

    if choice == 1:
        # Local Docker
        password = Prompt.ask(
            "Neo4j password",
            default=secrets.token_urlsafe(16),
            password=True,
        )
        return {
            "uri": "bolt://localhost:7688",  # Different port from kg-memory
            "user": "neo4j",
            "password": password,
            "docker": True,
        }
    else:
        # Remote
        uri = Prompt.ask(
            "Neo4j URI",
            default="neo4j+s://xxxxx.databases.neo4j.io",
        )
        user = Prompt.ask("Username", default="neo4j")
        password = Prompt.ask("Password", password=True)

        return {
            "uri": uri,
            "user": user,
            "password": password,
            "docker": False,
        }


def configure_llm() -> dict:
    """Configure LLM settings."""
    console.print("\n[bold cyan]LLM Configuration[/bold cyan]")

    choice = select_option(
        "How do you want to connect to the LLM?",
        [
            "LiteLLM Gateway (recommended for enterprise)",
            "Gemini Direct (requires API key from AI Studio)",
        ],
        default=1,
    )

    if choice == 1:
        # LiteLLM Gateway
        base_url = Prompt.ask(
            "LiteLLM Gateway URL",
            default="https://your-litellm-gateway.com/",
        )
        api_key = Prompt.ask("API Key", password=True)
        model = Prompt.ask("Model name", default="gemini-2.5-flash")

        return {
            "mode": "litellm",
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        }
    else:
        # Gemini Direct
        api_key = Prompt.ask("Gemini API Key", password=True)
        model = Prompt.ask("Model name", default="gemini-2.5-flash")

        return {
            "mode": "gemini_direct",
            "api_key": api_key,
            "model": model,
        }


def configure_crawler() -> dict:
    """Configure crawler settings."""
    console.print("\n[bold cyan]Crawler Configuration[/bold cyan]")

    max_depth = IntPrompt.ask(
        "Maximum crawl depth (hops from root URL)",
        default=2,
    )
    max_depth = max(1, min(5, max_depth))

    rate_limit = Prompt.ask(
        "Rate limit (seconds between requests)",
        default="1.0",
    )
    rate_limit = float(rate_limit)

    max_pages = IntPrompt.ask(
        "Maximum pages per source",
        default=500,
    )

    return {
        "max_depth": max_depth,
        "rate_limit": rate_limit,
        "max_pages": max_pages,
    }


def configure_security() -> dict:
    """Configure security settings."""
    console.print("\n[bold cyan]Security Configuration[/bold cyan]")

    enable_auth = Confirm.ask(
        "Enable authentication token?",
        default=True,
    )

    if enable_auth:
        token = Prompt.ask(
            "Auth token",
            default=secrets.token_urlsafe(32),
            password=True,
        )
    else:
        token = None

    return {
        "token": token,
    }


def write_env_file(config: dict, env_path: Path) -> None:
    """Write configuration to .env file."""
    lines = [
        "# MCP Doc Builder Configuration",
        "# Generated by doc-mcp-setup",
        "",
        "# Neo4j",
        f"NEO4J_URI={config['neo4j']['uri']}",
        f"NEO4J_USERNAME={config['neo4j']['user']}",
        f"NEO4J_PASSWORD={config['neo4j']['password']}",
        "",
        "# LLM",
        f"LLM_MODE={config['llm']['mode']}",
    ]

    if config['llm']['mode'] == "litellm":
        lines.extend([
            f"LITELLM_BASE_URL={config['llm'].get('base_url', '')}",
            f"LITELLM_API_KEY={config['llm'].get('api_key', '')}",
            f"LITELLM_MODEL={config['llm'].get('model', 'gemini-2.5-flash')}",
        ])
    else:
        lines.extend([
            f"GEMINI_API_KEY={config['llm'].get('api_key', '')}",
            f"GEMINI_MODEL={config['llm'].get('model', 'gemini-2.5-flash')}",
        ])

    lines.extend([
        "",
        "# Crawler",
        f"CRAWLER_MAX_DEPTH={config['crawler']['max_depth']}",
        f"CRAWLER_RATE_LIMIT={config['crawler']['rate_limit']}",
        f"CRAWLER_MAX_PAGES={config['crawler']['max_pages']}",
        "",
        "# Server",
        "MCP_HOST=127.0.0.1",
        "MCP_PORT=8001",
        "LOG_LEVEL=INFO",
    ])

    if config['security']['token']:
        lines.extend([
            "",
            "# Security",
            f"DOC_MCP_TOKEN={config['security']['token']}",
        ])

    env_path.write_text("\n".join(lines) + "\n")


def write_mcp_config(config: dict, config_path: Path) -> None:
    """Write VS Code MCP configuration."""
    import json

    # Find Python path
    python_path = sys.executable

    mcp_config = {
        "mcpServers": {
            "doc-builder": {
                "command": python_path,
                "args": ["-m", "doc_builder", "--transport", "stdio"],
                "env": {
                    "NEO4J_URI": config['neo4j']['uri'],
                    "NEO4J_USERNAME": config['neo4j']['user'],
                    "NEO4J_PASSWORD": config['neo4j']['password'],
                    "LLM_MODE": config['llm']['mode'],
                },
            }
        }
    }

    # Add LLM config
    if config['llm']['mode'] == "litellm":
        mcp_config["mcpServers"]["doc-builder"]["env"].update({
            "LITELLM_BASE_URL": config['llm'].get('base_url', ''),
            "LITELLM_API_KEY": config['llm'].get('api_key', ''),
            "LITELLM_MODEL": config['llm'].get('model', ''),
        })
    else:
        mcp_config["mcpServers"]["doc-builder"]["env"].update({
            "GEMINI_API_KEY": config['llm'].get('api_key', ''),
            "GEMINI_MODEL": config['llm'].get('model', ''),
        })

    config_path.write_text(json.dumps(mcp_config, indent=2))


def main():
    """Main setup wizard entry point."""
    print_header()

    # Collect configuration
    config = {}

    # Neo4j
    config['neo4j'] = configure_neo4j()

    # Test Neo4j if not local Docker
    if not config['neo4j'].get('docker'):
        console.print("\n[yellow]Testing Neo4j connection...[/yellow]")
        if asyncio.run(test_neo4j_connection(
            config['neo4j']['uri'],
            config['neo4j']['user'],
            config['neo4j']['password'],
        )):
            console.print("[green]Neo4j connection successful![/green]")
        else:
            if not Confirm.ask("Continue anyway?"):
                sys.exit(1)

    # LLM
    config['llm'] = configure_llm()

    # Test LLM
    console.print("\n[yellow]Testing LLM connection...[/yellow]")
    if asyncio.run(test_llm_connection(config['llm']['mode'], config['llm'])):
        console.print("[green]LLM connection successful![/green]")
    else:
        if not Confirm.ask("Continue anyway?"):
            sys.exit(1)

    # Crawler
    config['crawler'] = configure_crawler()

    # Security
    config['security'] = configure_security()

    # Write configuration files
    console.print("\n[bold cyan]Writing Configuration Files[/bold cyan]")

    # Determine paths
    server_dir = Path(__file__).parent.parent.parent.parent
    project_dir = server_dir.parent

    # Write .env
    env_path = project_dir / ".env"
    write_env_file(config, env_path)
    console.print(f"  [green]✓[/green] Written: {env_path}")

    # Write .vscode/mcp.json
    vscode_dir = project_dir / ".vscode"
    vscode_dir.mkdir(exist_ok=True)
    mcp_config_path = vscode_dir / "mcp.json"
    write_mcp_config(config, mcp_config_path)
    console.print(f"  [green]✓[/green] Written: {mcp_config_path}")

    # Summary
    console.print()
    console.print(Panel.fit(
        "[bold green]Setup Complete![/bold green]\n\n"
        "Next steps:\n"
        f"1. {'Run `docker compose up -d` to start Neo4j' if config['neo4j'].get('docker') else 'Neo4j is already configured'}\n"
        "2. Run `doc-mcp-status --doctor` to verify setup\n"
        "3. Use `python -m doc_builder` to start the server\n"
        "4. Or configure your IDE to use the MCP server",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
