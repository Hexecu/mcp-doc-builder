"""
Interactive setup wizard for MCP Doc Builder.
Configures LLM, Neo4j, security, and Antigravity integration.
"""

import asyncio
import json
import os
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()

# Shared configuration paths
USER_CONFIG_DIR = Path.home() / ".doc-builder"
USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
ENV_FILE = USER_CONFIG_DIR / ".env"
ANTIGRAVITY_CONFIG = Path.home() / ".gemini" / "antigravity" / "mcp_config.json"


class SetupWizard:
    """Interactive setup wizard."""

    def __init__(self) -> None:
        self.config: dict[str, str] = {}
        self.docker_available = False

    def run(self) -> None:
        """Run the setup wizard."""
        self._print_header()
        self._check_prerequisites()

        # Step 1: LLM Configuration
        self._step_llm()

        # Step 2: Neo4j Configuration
        self._step_neo4j()

        # Step 3: Security
        self._step_security()

        # Step 4: Generate .env
        self._step_generate_env()

        # Step 5: Start Neo4j (if Docker)
        if self.config.get("NEO4J_URI", "").startswith("bolt://localhost"):
            self._step_start_neo4j()

        # Step 6: Apply Schema
        self._step_apply_schema()

        # Step 7: Configure Antigravity
        self._step_antigravity()

        # Step 8: Summary
        self._step_summary()

    def _print_header(self) -> None:
        """Print wizard header."""
        console.print()
        console.print(Panel.fit(
            "[bold blue]MCP Doc Builder - Setup Wizard[/bold blue]\n"
            "Intelligent Documentation Indexing & Semantic Search",
            border_style="blue",
        ))
        console.print()

    def _check_prerequisites(self) -> None:
        """Check system prerequisites."""
        console.print("[bold]Checking prerequisites...[/bold]\n")

        # Check Python version
        py_version = sys.version_info
        if py_version >= (3, 11):
            console.print(f"  [green]✓[/green] Python {py_version.major}.{py_version.minor}")
        else:
            console.print(f"  [yellow]⚠[/yellow] Python {py_version.major}.{py_version.minor} (3.11+ recommended)")

        # Check Docker
        while True:
            self.docker_available = self._check_docker()
            if self.docker_available:
                console.print("  [green]✓[/green] Docker available")
                break
                
            console.print("  [yellow]⚠[/yellow] Docker engine is not running or not installed.")
            retry = questionary.confirm("Would you like to start Docker Desktop and retry?", default=True).ask()
            if not retry:
                console.print("  [dim]Proceeding without Docker (you will need to use a remote Neo4j instance).[/dim]")
                break
                
            console.print("[dim]Waiting for Docker to start...[/dim]")
            time.sleep(3)

        console.print()

    def _check_docker(self) -> bool:
        """Check if Docker is available and running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _prompt_required(self, prompt: str, default: str | None = None) -> str:
        """Prompt for required input using questionary."""
        while True:
            value = str(questionary.text(prompt, default=default or "").ask() or "")
            if value and value.strip():
                return value.strip()
            console.print("[red]This field is required[/red]")

    def _prompt_secret(self, prompt: str, default: str | None = None) -> str:
        """Prompt for secret input (masked)."""
        while True:
            value = str(questionary.password(prompt, default=default or "").ask() or "")
            if value and value.strip():
                return value.strip()
            console.print("[red]This field is required[/red]")

    def _mask_secret(self, secret: str) -> str:
        """Mask a secret for display."""
        if len(secret) <= 8:
            return "*" * len(secret)
        return secret[:4] + "*" * (len(secret) - 8) + secret[-4:]

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1: LLM Configuration
    # ─────────────────────────────────────────────────────────────────────────

    def _step_llm(self) -> None:
        """Configure LLM provider."""
        console.print(Panel("[bold]Step 1: LLM Configuration[/bold]", border_style="blue"))

        choice = questionary.select(
            "Select LLM provider:",
            choices=[
                "Gemini Direct (Google AI Studio) - Recommended",
                "LiteLLM Gateway (Enterprise/Proxy)"
            ]
        ).ask()

        if choice and "Gemini Direct" in choice:
            self._configure_gemini_direct()
        else:
            self._configure_litellm()

        console.print()

    def _configure_gemini_direct(self) -> None:
        """Configure Gemini Direct mode."""
        console.print("\n[dim]Get your API key from: https://aistudio.google.com/apikey[/dim]")

        self.config["LLM_MODE"] = "gemini_direct"
        self.config["GEMINI_API_KEY"] = self._prompt_secret("Gemini API Key")
        self.config["GEMINI_MODEL"] = questionary.text(
            "Model name",
            default="gemini-2.5-flash",
        ).ask().strip()

        # Embedding model
        self.config["EMBEDDING_MODEL"] = "gemini-embedding-001"
        self.config["EMBEDDING_DIMENSIONS"] = "3072"

        console.print(f"\n  [green]✓[/green] Gemini Direct configured")
        console.print(f"    Model: {self.config['GEMINI_MODEL']}")
        console.print(f"    Embedding: {self.config['EMBEDDING_MODEL']}")

    def _configure_litellm(self) -> None:
        """Configure LiteLLM Gateway mode."""
        console.print("\n[dim]Enter your LiteLLM Gateway details[/dim]")

        self.config["LLM_MODE"] = "litellm"
        self.config["LITELLM_BASE_URL"] = self._prompt_required(
            "LiteLLM Gateway URL",
            default="https://your-gateway.com/",
        )
        self.config["LITELLM_API_KEY"] = self._prompt_secret("LiteLLM API Key")
        self.config["LITELLM_MODEL"] = questionary.text(
            "Model name",
            default="gemini-2.5-flash",
        ).ask().strip()

        # Embedding via gateway
        self.config["EMBEDDING_MODEL"] = questionary.text(
            "Embedding model",
            default="text-embedding-3-small",
        ).ask().strip()
        self.config["EMBEDDING_DIMENSIONS"] = questionary.text(
            "Embedding dimensions",
            default="1536",
        ).ask().strip()

        console.print(f"\n  [green]✓[/green] LiteLLM Gateway configured")
        console.print(f"    URL: {self.config['LITELLM_BASE_URL']}")
        console.print(f"    Model: {self.config['LITELLM_MODEL']}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2: Neo4j Configuration
    # ─────────────────────────────────────────────────────────────────────────

    def _step_neo4j(self) -> None:
        """Configure Neo4j database."""
        console.print(Panel("[bold]Step 2: Neo4j Configuration[/bold]", border_style="blue"))

        options = []
        if self.docker_available:
            options.append("Docker (local) - Recommended")
        options.append("Remote Neo4j (Aura or self-hosted)")

        choice = questionary.select("Select Neo4j setup:", choices=options).ask()

        if choice and "Docker" in choice:
            self._configure_neo4j_docker()
        else:
            self._configure_neo4j_remote()

        console.print()

    def _configure_neo4j_docker(self) -> None:
        """Configure local Docker Neo4j."""
        console.print("\n[dim]Using local Docker container (port 7688)[/dim]")

        self.config["NEO4J_URI"] = "bolt://localhost:7688"
        self.config["NEO4J_USERNAME"] = "neo4j"

        # Generate or use existing password
        self.config["NEO4J_PASSWORD"] = questionary.text(
            "Neo4j password",
            default="password123",
        ).ask().strip()

        console.print(f"\n  [green]✓[/green] Neo4j Docker configured")
        console.print(f"    URI: {self.config['NEO4J_URI']}")
        console.print(f"    Browser: http://localhost:7475")

    def _configure_neo4j_remote(self) -> None:
        """Configure remote Neo4j."""
        console.print("\n[dim]Enter your Neo4j connection details[/dim]")

        self.config["NEO4J_URI"] = self._prompt_required(
            "Neo4j URI",
            default="neo4j+s://xxxxx.databases.neo4j.io",
        )
        self.config["NEO4J_USERNAME"] = questionary.text(
            "Username",
            default="neo4j",
        ).ask().strip()
        self.config["NEO4J_PASSWORD"] = self._prompt_secret("Password")

        console.print(f"\n  [green]✓[/green] Remote Neo4j configured")
        console.print(f"    URI: {self.config['NEO4J_URI']}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3: Security
    # ─────────────────────────────────────────────────────────────────────────

    def _step_security(self) -> None:
        """Configure security settings."""
        console.print(Panel("[bold]Step 3: Security Configuration[/bold]", border_style="blue"))

        if questionary.confirm("Enable token authentication?", default=False).ask():
            token = secrets.token_urlsafe(32)
            self.config["DOC_MCP_TOKEN"] = token
            console.print(f"\n  [green]✓[/green] Token generated: {self._mask_secret(token)}")
        else:
            console.print("  [dim]Token authentication disabled[/dim]")

        # Crawler settings
        console.print("\n[bold]Crawler Settings:[/bold]")
        self.config["CRAWLER_MAX_DEPTH"] = questionary.text(
            "Max crawl depth",
            default="2",
        ).ask().strip()
        self.config["CRAWLER_RATE_LIMIT"] = questionary.text(
            "Rate limit (seconds between requests)",
            default="1.0",
        ).ask().strip()
        self.config["CRAWLER_MAX_PAGES"] = questionary.text(
            "Max pages per source",
            default="500",
        ).ask().strip()

        console.print()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4: Generate .env
    # ─────────────────────────────────────────────────────────────────────────

    def _step_generate_env(self) -> None:
        """Generate .env file."""
        console.print(Panel("[bold]Step 4: Generate Configuration[/bold]", border_style="blue"))

        # Backup existing .env
        if ENV_FILE.exists():
            backup = ENV_FILE.with_suffix(f".env.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.copy(ENV_FILE, backup)
            console.print(f"  [dim]Backed up existing .env to {backup.name}[/dim]")

        # Generate .env content
        lines = [
            "# MCP Doc Builder Configuration",
            f"# Generated by setup wizard on {datetime.now().isoformat()}",
            "",
            "# LLM Configuration",
            f"LLM_MODE={self.config.get('LLM_MODE', 'gemini_direct')}",
        ]

        if self.config.get("LLM_MODE") == "gemini_direct":
            lines.extend([
                f"GEMINI_API_KEY={self.config.get('GEMINI_API_KEY', '')}",
                f"GEMINI_MODEL={self.config.get('GEMINI_MODEL', 'gemini-2.5-flash')}",
            ])
        else:
            lines.extend([
                f"LITELLM_BASE_URL={self.config.get('LITELLM_BASE_URL', '')}",
                f"LITELLM_API_KEY={self.config.get('LITELLM_API_KEY', '')}",
                f"LITELLM_MODEL={self.config.get('LITELLM_MODEL', '')}",
            ])

        lines.extend([
            "",
            "# Embedding Configuration",
            f"EMBEDDING_MODEL={self.config.get('EMBEDDING_MODEL', 'gemini-embedding-001')}",
            f"EMBEDDING_DIMENSIONS={self.config.get('EMBEDDING_DIMENSIONS', '3072')}",
            "",
            "# Neo4j Configuration",
            f"NEO4J_URI={self.config.get('NEO4J_URI', 'bolt://localhost:7688')}",
            f"NEO4J_USERNAME={self.config.get('NEO4J_USERNAME', 'neo4j')}",
            f"NEO4J_PASSWORD={self.config.get('NEO4J_PASSWORD', '')}",
            "",
            "# Crawler Configuration",
            f"CRAWLER_MAX_DEPTH={self.config.get('CRAWLER_MAX_DEPTH', '2')}",
            f"CRAWLER_RATE_LIMIT={self.config.get('CRAWLER_RATE_LIMIT', '1.0')}",
            f"CRAWLER_MAX_PAGES={self.config.get('CRAWLER_MAX_PAGES', '500')}",
            "",
            "# Server Configuration",
            "MCP_HOST=127.0.0.1",
            "MCP_PORT=8001",
            "LOG_LEVEL=INFO",
        ])

        if self.config.get("DOC_MCP_TOKEN"):
            lines.extend([
                "",
                "# Security",
                f"DOC_MCP_TOKEN={self.config.get('DOC_MCP_TOKEN', '')}",
            ])

        # Write .env file
        ENV_FILE.write_text("\n".join(lines) + "\n")
        console.print(f"\n  [green]✓[/green] Configuration saved to {ENV_FILE}")

    # ─────────────────────────────────────────────────────────────────────────
    # Step 5: Start Neo4j
    # ─────────────────────────────────────────────────────────────────────────

    def _step_start_neo4j(self) -> None:
        """Start Neo4j Docker container."""
        console.print(Panel("[bold]Step 5: Start Neo4j[/bold]", border_style="blue"))

        if not questionary.confirm("Start Neo4j Docker container?", default=True).ask():
            console.print("  [dim]Skipped. Run manually.[/dim]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Starting Neo4j...", total=None)

            try:
                # Setup data directory
                data_dir = USER_CONFIG_DIR / "neo4j_data"
                data_dir.mkdir(parents=True, exist_ok=True)
                password = self.config.get("NEO4J_PASSWORD", "password123")
                
                # Cleanup existing container named doc-builder-neo4j without stopping it if not needed,
                # actually it's better to force remove it if starting anew
                subprocess.run(
                    ["docker", "rm", "-f", "doc-builder-neo4j"],
                    capture_output=True,
                )

                # Pure docker run command (works without docker-compose.yml available locally)
                result = subprocess.run(
                    [
                        "docker", "run", "-d",
                        "--name", "doc-builder-neo4j",
                        "-p", "7688:7687",
                        "-p", "7475:7474",
                        "-e", f"NEO4J_AUTH=neo4j/{password}",
                        "-e", "NEO4J_apoc_export_file_enabled=true",
                        "-e", "NEO4J_apoc_import_file_enabled=true",
                        "-e", "NEO4J_apoc_import_file_use__neo4j__config=true",
                        "-e", "NEO4J_PLUGINS=[\"apoc\"]",
                        "-v", f"{data_dir.absolute()}:/data",
                        "--restart", "unless-stopped",
                        "neo4j:5"
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if result.returncode == 0:
                    progress.update(task, description="Waiting for Neo4j to be ready...")

                    # Wait for Neo4j to be healthy
                    for _ in range(40):
                        check = subprocess.run(
                            ["docker", "inspect", "-f", "{{.State.Status}}", "doc-builder-neo4j"],
                            capture_output=True,
                            text=True,
                        )
                        if "running" in check.stdout.lower():
                            # Let's wait a couple of seconds for the DB to be actually responsive
                            time.sleep(3)
                            break
                        time.sleep(2)

                    console.print("\n  [green]✓[/green] Neo4j started successfully")
                    console.print("    Browser: http://localhost:7475")
                else:
                    console.print(f"\n  [red]✗[/red] Failed to start Neo4j")
                    console.print(f"    {result.stderr}")

            except Exception as e:
                console.print(f"\n  [red]✗[/red] Error starting Neo4j: {e}")

        console.print()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 6: Apply Schema
    # ─────────────────────────────────────────────────────────────────────────

    def _step_apply_schema(self) -> None:
        """Apply Neo4j schema."""
        console.print(Panel("[bold]Step 6: Apply Database Schema[/bold]", border_style="blue"))

        if not questionary.confirm("Apply Neo4j schema now?", default=True).ask():
            console.print("  [dim]Skipped. Schema will be applied on first use.[/dim]")
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Applying schema...", total=None)

            try:
                # Load environment
                from dotenv import load_dotenv
                load_dotenv(ENV_FILE, override=True)

                # Apply schema
                from doc_builder.kg.neo4j import get_neo4j_client
                from doc_builder.kg.apply_schema import apply_schema

                async def run_schema() -> dict[str, Any]:
                    client = get_neo4j_client()
                    await client.connect()
                    result = await apply_schema()
                    await client.close()
                    return result

                result = asyncio.get_event_loop().run_until_complete(run_schema())

                console.print(f"\n  [green]✓[/green] Schema applied successfully")
                console.print(f"    Constraints: {result.get('success', 0)}")
                console.print(f"    Vector index: {result.get('vector_index', 'unknown')}")

            except Exception as e:
                console.print(f"\n  [yellow]⚠[/yellow] Schema application warning: {e}")
                console.print("    [dim]Schema will be applied on first use[/dim]")

        console.print()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 7: Configure Antigravity
    # ─────────────────────────────────────────────────────────────────────────

    def _step_antigravity(self) -> None:
        """Configure Antigravity MCP integration."""
        console.print(Panel("[bold]Step 7: Configure Antigravity[/bold]", border_style="blue"))

        if not ANTIGRAVITY_CONFIG.parent.exists():
            console.print("  [dim]Antigravity config directory not found. Skipping.[/dim]")
            return

        if not questionary.confirm("Configure Antigravity MCP integration?", default=True).ask():
            console.print("  [dim]Skipped.[/dim]")
            return

        try:
            # Load existing config
            if ANTIGRAVITY_CONFIG.exists():
                config = json.loads(ANTIGRAVITY_CONFIG.read_text())
            else:
                config = {"mcpServers": {}}

            # Find Python path
            python_path = sys.executable

            # Build doc-builder config
            doc_builder_config: dict[str, Any] = {
                "command": python_path,
                "args": ["-m", "doc_builder", "--transport", "stdio"],
                "env": {
                    "LOG_LEVEL": "INFO",
                    "LLM_MODE": self.config.get("LLM_MODE", "gemini_direct"),
                    "NEO4J_URI": self.config.get("NEO4J_URI", "bolt://localhost:7688"),
                    "NEO4J_USERNAME": self.config.get("NEO4J_USERNAME", "neo4j"),
                    "NEO4J_PASSWORD": self.config.get("NEO4J_PASSWORD", ""),
                    "EMBEDDING_MODEL": self.config.get("EMBEDDING_MODEL", "gemini-embedding-001"),
                    "EMBEDDING_DIMENSIONS": self.config.get("EMBEDDING_DIMENSIONS", "3072"),
                },
                "disabledTools": [],
                "disabled": False,
            }

            # Add LLM-specific config
            if self.config.get("LLM_MODE") == "gemini_direct":
                doc_builder_config["env"]["GEMINI_API_KEY"] = self.config.get("GEMINI_API_KEY", "")
                doc_builder_config["env"]["GEMINI_MODEL"] = self.config.get("GEMINI_MODEL", "gemini-2.5-flash")
            else:
                doc_builder_config["env"]["LITELLM_BASE_URL"] = self.config.get("LITELLM_BASE_URL", "")
                doc_builder_config["env"]["LITELLM_API_KEY"] = self.config.get("LITELLM_API_KEY", "")
                doc_builder_config["env"]["LITELLM_MODEL"] = self.config.get("LITELLM_MODEL", "")

            # Update config
            config["mcpServers"]["doc-builder"] = doc_builder_config

            # Backup and save
            if ANTIGRAVITY_CONFIG.exists():
                backup = ANTIGRAVITY_CONFIG.with_suffix(
                    f".json.backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                shutil.copy(ANTIGRAVITY_CONFIG, backup)

            ANTIGRAVITY_CONFIG.write_text(json.dumps(config, indent=2))

            console.print(f"\n  [green]✓[/green] Antigravity configured")
            console.print(f"    Config: {ANTIGRAVITY_CONFIG}")
            console.print("    [dim]Restart Antigravity to load the new MCP server[/dim]")

        except Exception as e:
            console.print(f"\n  [red]✗[/red] Failed to configure Antigravity: {e}")

        console.print()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 8: Summary
    # ─────────────────────────────────────────────────────────────────────────

    def _step_summary(self) -> None:
        """Show setup summary."""
        console.print(Panel("[bold green]Setup Complete![/bold green]", border_style="green"))

        table = Table(title="Configuration Summary", show_header=False)
        table.add_column("Setting", style="cyan")
        table.add_column("Value")

        table.add_row("LLM Mode", self.config.get("LLM_MODE", ""))
        if self.config.get("LLM_MODE") == "gemini_direct":
            table.add_row("Model", self.config.get("GEMINI_MODEL", ""))
        else:
            table.add_row("Gateway", self.config.get("LITELLM_BASE_URL", ""))
            table.add_row("Model", self.config.get("LITELLM_MODEL", ""))

        table.add_row("Embedding", self.config.get("EMBEDDING_MODEL", ""))
        table.add_row("Neo4j", self.config.get("NEO4J_URI", ""))
        table.add_row("Max Depth", self.config.get("CRAWLER_MAX_DEPTH", ""))
        table.add_row("Max Pages", self.config.get("CRAWLER_MAX_PAGES", ""))

        console.print()
        console.print(table)

        console.print()
        console.print("[bold]Next Steps:[/bold]")
        console.print()
        console.print("  1. [cyan]Index documentation:[/cyan]")
        console.print("     doc-mcp-index --interactive")
        console.print()
        console.print("  2. [cyan]Check status:[/cyan]")
        console.print("     doc-mcp-status --doctor")
        console.print()
        console.print("  3. [cyan]Restart Antigravity[/cyan] to load the MCP server")
        console.print()
        console.print("  4. [cyan]Use in Antigravity:[/cyan]")
        console.print("     doc_search query=\"your search query\"")
        console.print()


def main() -> None:
    """Main entry point for setup wizard."""
    wizard = SetupWizard()
    try:
        wizard.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Setup cancelled[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Setup failed: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()
