# MCP Doc Builder

Intelligent Documentation Scraping, Vectorization, and Semantic Search for AI Coding Assistants.

## Overview

MCP Doc Builder is a Model Context Protocol (MCP) server that provides:

- **Intelligent Web Scraping**: LLM-guided crawler that intelligently decides which documentation pages to index
- **Semantic Vectorization**: Gemini text-embedding-004 for semantic search across documentation
- **Dynamic Ontology**: Automatically extracts concepts and relationships from documentation
- **Knowledge Graph**: Neo4j-based storage with full graph traversal capabilities
- **Hybrid Search**: Combined vector similarity and fulltext search for optimal results

## Features

### Intelligent Crawling
- LLM-powered link evaluation decides which pages to follow
- Respects rate limits to avoid overwhelming documentation servers
- Configurable depth (1-5 hops from root URL)
- Smart content extraction with trafilatura

### Semantic Search
- Gemini text-embedding-004 for 768-dimensional vectors
- Neo4j Vector Index for fast similarity search
- Fulltext search with Lucene
- Hybrid search combining both methods

### Dynamic Ontology
- Automatic concept extraction (APIs, patterns, entities)
- Relationship inference (uses, extends, requires, etc.)
- Chunk-to-concept linking
- Concept co-occurrence analysis

### MCP Integration
- 6 tools for complete documentation management
- Resources for graph exploration
- Workflow prompts for common tasks

## Quick Start

### 1. Prerequisites

- Python 3.11+
- Docker (for Neo4j)
- LiteLLM Gateway or Gemini API key

### 2. Installation

You can install `doc-builder-mcp` globally using `pipx` (recommended) or in a local virtual environment.

#### Option 1: One-Line Install (Recommended)

```bash
# Install the package
pipx install doc-builder-mcp

# Run the interactive Setup Wizard
doc-mcp-setup
```

The wizard will:
1.  Check for Docker and Neo4j.
2.  Ask for your **LiteLLM / Gemini Credentials**.
3.  Configure the **LLM Mode** (LiteLLM vs Gemini Direct).
4.  Generate a secure `.env` file.

<details>
<summary>❓ Don't have <code>pipx</code>? Click here to install it</summary>

**macOS:**
```bash
brew install pipx
pipx ensurepath
```

**Windows:**
```bash
winget install pipx
pipx ensurepath
```

**Linux (Debian/Ubuntu):**
```bash
sudo apt install pipx
pipx ensurepath
```

*Restart your terminal after installing pipx.*

</details>

#### Alternative: Standard Pip

If you prefer not to use pipx:
```bash
pip install doc-builder-mcp
doc-mcp-setup
```

#### Option 2: Manual Development Setup

If you want to contribute or modify the code:

```bash
git clone https://github.com/Hexecu/mcp-doc-builder.git
cd mcp-doc-builder
make full-setup
```

### 3. Setup

Run the interactive setup wizard:

```bash
doc-mcp-setup
```

Or manually configure:

```bash
cp ../.env.example ../.env
# Edit .env with your configuration
```

### 4. Start Neo4j

Start the Neo4j database natively with docker or using the provided Makefile:

```bash
make neo4j-up
```

*This uses the `docker-compose.yml` to start the Neo4j instance.*

### 5. Run the Server

```bash
# STDIO mode (for IDE integration)
make server-stdio

# HTTP mode (for API access)
make server
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEO4J_URI` | Neo4j connection URI | `bolt://localhost:7688` |
| `NEO4J_USERNAME` | Neo4j username | `neo4j` |
| `NEO4J_PASSWORD` | Neo4j password | - |
| `LLM_MODE` | `litellm`, `gemini_direct`, or `both` | `litellm` |
| `LITELLM_BASE_URL` | LiteLLM Gateway URL | - |
| `LITELLM_API_KEY` | LiteLLM API key | - |
| `LITELLM_MODEL` | Model name | `gemini-2.5-flash` |
| `CRAWLER_MAX_DEPTH` | Maximum crawl depth | `2` |
| `CRAWLER_RATE_LIMIT` | Seconds between requests | `1.0` |
| `CRAWLER_MAX_PAGES` | Max pages per source | `500` |

## MCP Tools

### doc_ingest
Ingest and index a documentation website.

```json
{
  "url": "https://nextjs.org/docs",
  "name": "Next.js Docs",
  "max_depth": 2
}
```

### doc_search
Search indexed documentation.

```json
{
  "query": "how to use React hooks",
  "limit": 10,
  "search_mode": "hybrid"
}
```

### doc_context
Get comprehensive context for a topic.

```json
{
  "topic": "authentication in Next.js",
  "include_related": true
}
```

### doc_sources
List all indexed documentation sources.

### doc_refresh
Refresh/re-index a documentation source.

```json
{
  "source_id": "source_abc123",
  "force": false
}
```

### doc_ontology
Explore extracted concepts and relationships.

```json
{
  "source_id": "source_abc123",
  "concept": "useState"
}
```

## IDE Integration

### VS Code / Cursor / Windsurf

Add to `.vscode/mcp.json`:

```json
{
  "mcpServers": {
    "doc-builder": {
      "command": "python",
      "args": ["-m", "doc_builder", "--transport", "stdio"],
      "env": {
        "NEO4J_URI": "bolt://localhost:7688",
        "NEO4J_USERNAME": "neo4j",
        "NEO4J_PASSWORD": "your-password",
        "LLM_MODE": "litellm",
        "LITELLM_BASE_URL": "https://your-gateway.com/",
        "LITELLM_API_KEY": "your-key",
        "LITELLM_MODEL": "gemini-2.5-flash"
      }
    }
  }
}
```

## Architecture

```
mcp-doc-builder/
├── docker-compose.yml        # Neo4j container
├── .env.example              # Configuration template
└── server/
    ├── pyproject.toml        # Python package
    └── src/doc_builder/
        ├── main.py           # MCP server entry
        ├── config.py         # Settings
        ├── cli/              # Setup wizard & status
        ├── crawler/          # Web scraping
        │   ├── spider.py     # Async crawler
        │   ├── parser.py     # HTML parsing
        │   └── agent.py      # LLM link evaluation
        ├── vector/           # Vectorization
        │   ├── embedder.py   # Gemini embeddings
        │   ├── chunker.py    # Smart chunking
        │   └── indexer.py    # Neo4j vector index
        ├── ontology/         # Knowledge extraction
        │   ├── extractor.py  # Concept extraction
        │   ├── metatag.py    # Metatag processing
        │   └── linker.py     # Relationship building
        ├── kg/               # Neo4j graph
        │   ├── neo4j.py      # Async client
        │   ├── repo.py       # Query repository
        │   └── schema.cypher # Database schema
        ├── llm/              # LLM integration
        │   ├── client.py     # LiteLLM wrapper
        │   └── prompts/      # Prompt templates
        ├── mcp/              # MCP protocol
        │   ├── tools.py      # Tool definitions
        │   ├── resources.py  # Resource handlers
        │   └── prompts.py    # Workflow prompts
        └── security/         # Auth & validation
```

## Graph Schema

### Nodes (Doc* prefixed for namespace separation)

- **DocSource**: Documentation root (URL, name, status)
- **DocPage**: Individual pages with metadata
- **DocChunk**: Vectorized content chunks with embeddings
- **DocConcept**: Extracted concepts (APIs, patterns, entities)
- **DocMetatag**: Page metatags (og:*, twitter:*, etc.)
- **DocCrawlJob**: Crawl job tracking

### Relationships

- `(DocSource)-[:CONTAINS]->(DocPage)`
- `(DocPage)-[:LINKS_TO]->(DocPage)`
- `(DocPage)-[:HAS_CHUNK]->(DocChunk)`
- `(DocChunk)-[:MENTIONS]->(DocConcept)`
- `(DocConcept)-[:RELATES_TO]->(DocConcept)`

## CLI Commands

```bash
# Interactive setup
doc-mcp-setup

# Health check
doc-mcp-status --doctor

# Run server
doc-mcp
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy src/

# Linting
ruff check src/
```

## License

MIT

## Related Projects

- [MCP KG Memory](../mcp-kg-memory): Knowledge graph memory for AI coding assistants
- [Model Context Protocol](https://modelcontextprotocol.io): MCP specification
