"""
Pydantic schemas for LLM responses.
"""

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────────
# Crawler Agent Schemas
# ─────────────────────────────────────────────────────────────────────────────────


class LinkDecision(BaseModel):
    """Decision for a single link."""

    url: str
    action: str = Field(description="'follow' or 'skip'")
    reason: str = Field(default="")
    priority: float = Field(default=0.5, ge=0.0, le=1.0)


class LinkEvaluationResult(BaseModel):
    """Result of link evaluation."""

    decisions: list[LinkDecision]


class BatchLinkResult(BaseModel):
    """Result of batch link evaluation."""

    follow: list[str] = Field(default_factory=list)
    skip: list[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────────
# Ontology Extraction Schemas
# ─────────────────────────────────────────────────────────────────────────────────


class ExtractedConcept(BaseModel):
    """A concept extracted from documentation."""

    name: str
    description: str = Field(default="")
    category: str = Field(
        default="concept",
        description="api, pattern, entity, action, config, tool, concept",
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    aliases: list[str] = Field(default_factory=list)


class ExtractedRelationship(BaseModel):
    """A relationship between concepts."""

    from_concept: str = Field(alias="from")
    to_concept: str = Field(alias="to")
    type: str = Field(description="Relationship type")
    description: str = Field(default="")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    class Config:
        populate_by_name = True


class OntologyExtractionResult(BaseModel):
    """Result of ontology extraction."""

    concepts: list[ExtractedConcept] = Field(default_factory=list)
    relationships: list[ExtractedRelationship] = Field(default_factory=list)


class MergedConcept(BaseModel):
    """A concept after merging duplicates."""

    id: str | None = Field(default=None)
    name: str
    description: str = Field(default="")
    category: str = Field(default="concept")
    confidence: float = Field(default=0.8)
    mention_count: int = Field(default=1)


class DuplicateMerge(BaseModel):
    """Record of merged duplicates."""

    kept: str
    merged: list[str]


class OntologyMergeResult(BaseModel):
    """Result of ontology merge."""

    merged_concepts: list[MergedConcept] = Field(default_factory=list)
    new_relationships: list[ExtractedRelationship] = Field(default_factory=list)
    duplicates_merged: list[DuplicateMerge] = Field(default_factory=list)


class ChunkClassification(BaseModel):
    """Classification of a document chunk."""

    semantic_type: str = Field(
        default="conceptual",
        description="code_example, api_reference, conceptual, tutorial, etc.",
    )
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    key_topics: list[str] = Field(default_factory=list)
    complexity: str = Field(
        default="intermediate",
        description="beginner, intermediate, advanced",
    )


# ─────────────────────────────────────────────────────────────────────────────────
# Chunker Schemas
# ─────────────────────────────────────────────────────────────────────────────────


class SplitPoint(BaseModel):
    """A suggested split point in text."""

    position: int
    type: str = Field(description="heading, topic_change, code_boundary, etc.")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    context: str = Field(default="")


class SuggestedChunk(BaseModel):
    """A suggested chunk boundary."""

    start: int
    end: int
    type: str = Field(default="mixed")
    heading: str = Field(default="")


class ChunkBoundaryResult(BaseModel):
    """Result of chunk boundary detection."""

    split_points: list[SplitPoint] = Field(default_factory=list)
    suggested_chunks: list[SuggestedChunk] = Field(default_factory=list)


class HeadingInfo(BaseModel):
    """Information about a heading."""

    level: int = Field(ge=1, le=6)
    text: str
    position: int
    parent: str | None = Field(default=None)


class HeadingExtractionResult(BaseModel):
    """Result of heading extraction."""

    headings: list[HeadingInfo] = Field(default_factory=list)


class TruncationResult(BaseModel):
    """Result of smart truncation."""

    truncation_index: int
    truncation_type: str
    content_preserved_percent: float
    context_lost: str = Field(default="")
