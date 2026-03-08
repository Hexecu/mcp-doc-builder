"""
LLM integration module.
"""

from doc_builder.llm.client import LLMClient, get_llm_client
from doc_builder.llm.schemas import (
    BatchLinkResult,
    ChunkBoundaryResult,
    ChunkClassification,
    ExtractedConcept,
    ExtractedRelationship,
    HeadingExtractionResult,
    LinkDecision,
    LinkEvaluationResult,
    OntologyExtractionResult,
    OntologyMergeResult,
)

__all__ = [
    "LLMClient",
    "get_llm_client",
    # Schemas
    "LinkDecision",
    "LinkEvaluationResult",
    "BatchLinkResult",
    "ExtractedConcept",
    "ExtractedRelationship",
    "OntologyExtractionResult",
    "OntologyMergeResult",
    "ChunkClassification",
    "ChunkBoundaryResult",
    "HeadingExtractionResult",
]
