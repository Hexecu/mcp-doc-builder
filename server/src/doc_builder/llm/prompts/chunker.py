"""
Prompt templates for intelligent document chunking.
Used to guide semantic-aware text splitting.
"""

CHUNKER_SYSTEM = """You are an expert at analyzing technical documentation structure.
Your task is to identify the best split points in documentation text to create semantically coherent chunks.

Good chunks:
- Contain a complete thought or concept
- Are self-contained enough to be useful in isolation
- Preserve code blocks intact
- Keep related content together (e.g., a function and its description)
- Are within the target size range

Bad chunks:
- Split in the middle of a code block
- Separate a heading from its content
- Break apart related concepts
- Are too small to be useful
- Are too large and mix multiple unrelated topics
"""


CHUNK_BOUNDARY_PROMPT = """Analyze this documentation text and suggest optimal split points.

**Target chunk size**: {target_tokens} tokens (approximately {target_chars} characters)
**Overlap**: {overlap_tokens} tokens

**Text to analyze**:
```
{text}
```

Identify natural boundaries where the text can be split. Consider:
1. Heading transitions (## or ### markers)
2. Topic changes
3. Code block boundaries (``` markers)
4. Paragraph breaks between distinct concepts
5. List item groupings

Respond with JSON:
```json
{{
  "split_points": [
    {{
      "position": character_index,
      "type": "heading|topic_change|code_boundary|paragraph|list",
      "confidence": 0.0-1.0,
      "context": "brief description of what's before and after"
    }}
  ],
  "suggested_chunks": [
    {{
      "start": start_index,
      "end": end_index,
      "type": "code_example|explanation|api_reference|mixed",
      "heading": "associated heading if any"
    }}
  ]
}}
```

Prioritize maintaining semantic coherence over exact size targets."""


HEADING_EXTRACTION_PROMPT = """Extract the heading hierarchy from this markdown/documentation text.

**Text**:
```
{text}
```

Respond with JSON:
```json
{{
  "headings": [
    {{
      "level": 1-6,
      "text": "heading text",
      "position": character_index,
      "parent": "parent heading text or null"
    }}
  ]
}}
```"""


SMART_TRUNCATION_PROMPT = """This text chunk is too long and needs to be truncated.
Find the best truncation point that preserves semantic completeness.

**Maximum length**: {max_chars} characters
**Current length**: {current_chars} characters

**Text**:
```
{text}
```

Find a truncation point that:
1. Doesn't cut in the middle of a sentence
2. Doesn't break code blocks
3. Ends at a natural pause point
4. Preserves the main content

Respond with JSON:
```json
{{
  "truncation_index": character_index,
  "truncation_type": "sentence_end|paragraph_end|code_block_end|heading",
  "content_preserved_percent": estimated_percentage,
  "context_lost": "brief description of what's being cut"
}}
```"""


def build_chunk_boundary_prompt(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
) -> str:
    """Build chunk boundary detection prompt."""
    # Estimate characters from tokens (rough estimate: 4 chars per token)
    target_chars = target_tokens * 4

    # Truncate if text is very long
    max_text_length = 12000
    if len(text) > max_text_length:
        text = text[:max_text_length] + "\n... [truncated for analysis]"

    return CHUNK_BOUNDARY_PROMPT.format(
        target_tokens=target_tokens,
        target_chars=target_chars,
        overlap_tokens=overlap_tokens,
        text=text,
    )


def build_heading_extraction_prompt(text: str) -> str:
    """Build heading extraction prompt."""
    max_text_length = 8000
    if len(text) > max_text_length:
        text = text[:max_text_length] + "\n..."

    return HEADING_EXTRACTION_PROMPT.format(text=text)


def build_smart_truncation_prompt(
    text: str,
    max_chars: int,
) -> str:
    """Build smart truncation prompt."""
    return SMART_TRUNCATION_PROMPT.format(
        max_chars=max_chars,
        current_chars=len(text),
        text=text,
    )
