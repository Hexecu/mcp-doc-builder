"""
Prompt templates for LLM operations.
"""

from doc_builder.llm.prompts.chunker import (
    CHUNKER_SYSTEM,
    build_chunk_boundary_prompt,
    build_heading_extraction_prompt,
    build_smart_truncation_prompt,
)
from doc_builder.llm.prompts.crawler_agent import (
    CRAWLER_AGENT_SYSTEM,
    build_batch_evaluation_prompt,
    build_link_evaluation_prompt,
)
from doc_builder.llm.prompts.ontology import (
    ONTOLOGY_EXTRACTOR_SYSTEM,
    build_chunk_classification_prompt,
    build_ontology_extraction_prompt,
    build_ontology_merge_prompt,
)

__all__ = [
    # Crawler
    "CRAWLER_AGENT_SYSTEM",
    "build_link_evaluation_prompt",
    "build_batch_evaluation_prompt",
    # Ontology
    "ONTOLOGY_EXTRACTOR_SYSTEM",
    "build_ontology_extraction_prompt",
    "build_ontology_merge_prompt",
    "build_chunk_classification_prompt",
    # Chunker
    "CHUNKER_SYSTEM",
    "build_chunk_boundary_prompt",
    "build_heading_extraction_prompt",
    "build_smart_truncation_prompt",
]
