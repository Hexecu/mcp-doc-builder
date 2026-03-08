"""
Prompt templates for ontology extraction.
Used to extract concepts and relationships from documentation content.
"""

ONTOLOGY_EXTRACTOR_SYSTEM = """You are an expert knowledge engineer specializing in extracting structured ontologies from technical documentation.

Your task is to analyze documentation text and extract:
1. **Concepts**: Key entities, APIs, patterns, and ideas mentioned
2. **Relationships**: How concepts relate to each other
3. **Categories**: Classification of each concept

## Concept Categories:
- **api**: Functions, methods, classes, interfaces, endpoints
- **pattern**: Design patterns, architectural patterns, coding patterns  
- **entity**: Data structures, models, schemas, types
- **action**: Operations, processes, workflows, commands
- **config**: Configuration options, settings, parameters
- **tool**: Libraries, frameworks, CLI tools, utilities
- **concept**: Abstract ideas, principles, paradigms

## Relationship Types:
- **uses**: A uses B (e.g., "Component uses Hook")
- **extends**: A extends/inherits from B
- **implements**: A implements B (interface/protocol)
- **requires**: A requires/depends on B
- **returns**: Function A returns type B
- **contains**: A contains B (composition)
- **configures**: A configures B
- **produces**: A produces/creates B
- **consumes**: A consumes/processes B
- **triggers**: A triggers B (event/action)
- **related_to**: General relationship

## Guidelines:
1. Focus on technical concepts, not generic words
2. Normalize names (use official/canonical names when possible)
3. Include context in descriptions
4. Assign confidence scores based on clarity of mention
5. Identify both explicit and implicit relationships
6. Prefer specific relationship types over "related_to"
"""


ONTOLOGY_EXTRACTION_PROMPT = """Analyze this documentation content and extract the ontology:

**Source**: {source_name}
**Page**: {page_title}
**URL**: {page_url}

**Content**:
```
{content}
```

Extract all significant concepts and their relationships.

Respond with JSON:
```json
{{
  "concepts": [
    {{
      "name": "ConceptName",
      "description": "Brief description from context",
      "category": "api|pattern|entity|action|config|tool|concept",
      "confidence": 0.0-1.0,
      "aliases": ["alternative names if any"]
    }}
  ],
  "relationships": [
    {{
      "from": "ConceptA",
      "to": "ConceptB",
      "type": "relationship_type",
      "description": "optional context",
      "confidence": 0.0-1.0
    }}
  ]
}}
```

Focus on:
- API names, function signatures, class names
- Configuration options and their purposes
- Patterns and best practices mentioned
- Dependencies and requirements
- Data flow and transformations"""


ONTOLOGY_MERGE_PROMPT = """You have extracted ontologies from multiple documentation pages.
Now merge and consolidate them into a unified ontology.

**Existing Concepts** (from previous extractions):
```json
{existing_concepts}
```

**New Concepts** (from latest extraction):
```json
{new_concepts}
```

Tasks:
1. Identify duplicate concepts (same entity, different names/descriptions)
2. Merge duplicates, keeping the best description
3. Update confidence scores (higher if mentioned multiple times)
4. Identify new relationships between existing and new concepts

Respond with JSON:
```json
{{
  "merged_concepts": [
    {{
      "id": "existing_id or null for new",
      "name": "canonical name",
      "description": "best description",
      "category": "category",
      "confidence": "updated confidence",
      "mention_count": "number of times seen"
    }}
  ],
  "new_relationships": [
    {{
      "from": "concept_name",
      "to": "concept_name",
      "type": "relationship_type",
      "confidence": 0.0-1.0
    }}
  ],
  "duplicates_merged": [
    {{
      "kept": "canonical name",
      "merged": ["alias1", "alias2"]
    }}
  ]
}}
```"""


CHUNK_CLASSIFICATION_PROMPT = """Classify this documentation chunk by its semantic type:

**Chunk content**:
```
{content}
```

**Heading context**: {heading_context}

Classify into ONE of these types:
- **code_example**: Code snippets, examples, sample implementations
- **api_reference**: API documentation, function signatures, parameters
- **conceptual**: Explanations of concepts, theory, principles
- **tutorial**: Step-by-step instructions, how-to guides
- **configuration**: Config options, setup instructions, environment
- **troubleshooting**: Error handling, debugging, common issues
- **migration**: Upgrade guides, breaking changes, version differences
- **overview**: Introduction, summary, high-level description

Respond with JSON:
```json
{{
  "semantic_type": "type_name",
  "confidence": 0.0-1.0,
  "key_topics": ["topic1", "topic2"],
  "complexity": "beginner|intermediate|advanced"
}}
```"""


def build_ontology_extraction_prompt(
    source_name: str,
    page_title: str,
    page_url: str,
    content: str,
) -> str:
    """Build ontology extraction prompt."""
    # Truncate content if too long
    max_content_length = 8000
    if len(content) > max_content_length:
        content = content[:max_content_length] + "\n... [truncated]"

    return ONTOLOGY_EXTRACTION_PROMPT.format(
        source_name=source_name,
        page_title=page_title,
        page_url=page_url,
        content=content,
    )


def build_ontology_merge_prompt(
    existing_concepts: list[dict],
    new_concepts: list[dict],
) -> str:
    """Build ontology merge prompt."""
    import json

    return ONTOLOGY_MERGE_PROMPT.format(
        existing_concepts=json.dumps(existing_concepts, indent=2),
        new_concepts=json.dumps(new_concepts, indent=2),
    )


def build_chunk_classification_prompt(
    content: str,
    heading_context: str,
) -> str:
    """Build chunk classification prompt."""
    # Truncate content if too long
    max_content_length = 2000
    if len(content) > max_content_length:
        content = content[:max_content_length] + "..."

    return CHUNK_CLASSIFICATION_PROMPT.format(
        content=content,
        heading_context=heading_context or "None",
    )
