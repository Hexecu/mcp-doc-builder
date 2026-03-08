"""
MCP Prompts for documentation workflows.
"""

from typing import Any


# ─────────────────────────────────────────────────────────────────────────────────
# Workflow Prompts
# ─────────────────────────────────────────────────────────────────────────────────

SEARCH_DOCUMENTATION_PROMPT = """Search Documentation Workflow

You have access to indexed documentation through the MCP Doc Builder.

## Available Tools

1. **doc_sources** - List all indexed documentation sources
2. **doc_search** - Search for specific information
3. **doc_context** - Get comprehensive context for a topic
4. **doc_ontology** - Explore concepts and relationships

## Workflow

1. First, use `doc_sources` to see what documentation is available
2. Use `doc_search` with your query to find relevant content
3. If you need deeper context, use `doc_context` with the topic
4. Use `doc_ontology` to understand related concepts

## Example

User asks: "How do I set up authentication in Next.js?"

1. Search: `doc_search(query="authentication setup Next.js")`
2. Get context: `doc_context(topic="Next.js authentication")`
3. The results will include relevant code examples and concepts

## Tips

- Use natural language queries for best results
- Combine search with context for comprehensive understanding
- Check the semantic_type field to prioritize code examples or explanations
"""


ADD_DOCUMENTATION_PROMPT = """Add Documentation Workflow

Use this workflow to index new documentation sources.

## Steps

1. **Identify the documentation URL**
   - Use the root URL of the documentation site
   - Example: "https://nextjs.org/docs"

2. **Choose a descriptive name**
   - This will be used to identify the source
   - Example: "Next.js Official Docs"

3. **Set appropriate depth**
   - depth=1: Only the root page and direct links
   - depth=2: Recommended for most documentation
   - depth=3: For comprehensive indexing

4. **Run the ingestion**
   ```
   doc_ingest(
       url="https://nextjs.org/docs",
       name="Next.js Official Docs",
       max_depth=2
   )
   ```

5. **Monitor progress**
   - The tool will return statistics when complete
   - Large documentation sites may take several minutes

## After Indexing

- Use `doc_sources` to verify the source was added
- Use `doc_search` to test searching the new content
- Use `doc_ontology` to explore extracted concepts
"""


EXPLORE_CONCEPTS_PROMPT = """Explore Documentation Concepts

This workflow helps you understand the structure of indexed documentation.

## Getting Started

1. **List sources**
   ```
   doc_sources()
   ```

2. **Explore a source's ontology**
   ```
   doc_ontology(source_id="source_xxx")
   ```

3. **Focus on a specific concept**
   ```
   doc_ontology(source_id="source_xxx", concept="useState")
   ```

## Understanding the Output

The ontology includes:

- **Concepts**: Key entities extracted from the documentation
  - Categories: api, pattern, entity, action, config, tool, concept
  - Mention count indicates importance

- **Relationships**: How concepts relate to each other
  - uses, extends, implements, requires
  - returns, contains, configures
  - produces, consumes, triggers

## Use Cases

1. **Understanding API structure**
   - Find all "api" category concepts
   - See how they relate to each other

2. **Learning patterns**
   - Filter for "pattern" concepts
   - Understand best practices

3. **Dependency mapping**
   - Follow "requires" relationships
   - Understand what depends on what
"""


# ─────────────────────────────────────────────────────────────────────────────────
# Prompt Registry
# ─────────────────────────────────────────────────────────────────────────────────

PROMPTS = {
    "SearchDocumentation": {
        "description": "Workflow for searching and finding information in documentation",
        "content": SEARCH_DOCUMENTATION_PROMPT,
        "arguments": [],
    },
    "AddDocumentation": {
        "description": "Workflow for adding new documentation sources",
        "content": ADD_DOCUMENTATION_PROMPT,
        "arguments": [
            {
                "name": "url",
                "description": "URL of the documentation to add",
                "required": False,
            }
        ],
    },
    "ExploreConcepts": {
        "description": "Workflow for exploring extracted concepts and relationships",
        "content": EXPLORE_CONCEPTS_PROMPT,
        "arguments": [
            {
                "name": "source_id",
                "description": "Optional source ID to focus on",
                "required": False,
            }
        ],
    },
}


def get_prompts() -> dict:
    """Get all prompt definitions."""
    return PROMPTS


def render_prompt(prompt_name: str, arguments: dict[str, Any] | None = None) -> str:
    """
    Render a prompt with optional arguments.
    
    Args:
        prompt_name: Name of the prompt
        arguments: Optional arguments to include
        
    Returns:
        Rendered prompt content
    """
    if prompt_name not in PROMPTS:
        return f"Unknown prompt: {prompt_name}"

    prompt = PROMPTS[prompt_name]
    content = prompt["content"]

    # Add any provided arguments as context
    if arguments:
        context_lines = ["\n## Context"]
        for key, value in arguments.items():
            context_lines.append(f"- {key}: {value}")
        content += "\n".join(context_lines)

    return content
