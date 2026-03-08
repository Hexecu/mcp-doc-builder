"""
LLM-powered ontology extraction from documentation content.
Extracts concepts and relationships to build a dynamic knowledge graph.
"""

import logging
from dataclasses import dataclass
from typing import Any

from doc_builder.kg import get_repository
from doc_builder.llm import get_llm_client
from doc_builder.llm.prompts import (
    ONTOLOGY_EXTRACTOR_SYSTEM,
    build_chunk_classification_prompt,
    build_ontology_extraction_prompt,
)
from doc_builder.llm.schemas import (
    ChunkClassification,
    ExtractedConcept,
    ExtractedRelationship,
    OntologyExtractionResult,
)
from doc_builder.utils import generate_id

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of ontology extraction for a page."""

    concepts_extracted: int
    relationships_extracted: int
    concepts: list[dict]
    relationships: list[dict]


class OntologyExtractor:
    """
    Extracts concepts and relationships from documentation using LLM.
    """

    def __init__(self):
        self.llm = get_llm_client()
        self.repo = get_repository()

    async def extract_from_content(
        self,
        source_id: str,
        source_name: str,
        page_title: str,
        page_url: str,
        content: str,
    ) -> ExtractionResult:
        """
        Extract ontology from page content.
        
        Args:
            source_id: Documentation source ID
            source_name: Name of the documentation
            page_title: Page title
            page_url: Page URL
            content: Page content
            
        Returns:
            ExtractionResult with extracted concepts and relationships
        """
        if not content or len(content) < 100:
            return ExtractionResult(
                concepts_extracted=0,
                relationships_extracted=0,
                concepts=[],
                relationships=[],
            )

        # Build prompt
        prompt = build_ontology_extraction_prompt(
            source_name=source_name,
            page_title=page_title,
            page_url=page_url,
            content=content,
        )

        try:
            # Extract using LLM
            result = await self.llm.complete_structured(
                messages=[
                    {"role": "system", "content": ONTOLOGY_EXTRACTOR_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_model=OntologyExtractionResult,
                model=self.llm.settings.reason_model,
                temperature=0.3,
            )

            # Store concepts
            stored_concepts = []
            concept_id_map: dict[str, str] = {}

            for concept in result.concepts:
                try:
                    stored = await self.repo.upsert_concept(
                        source_id=source_id,
                        name=concept.name,
                        description=concept.description,
                        category=concept.category,
                        confidence=concept.confidence,
                    )
                    stored_concepts.append(stored)
                    concept_id_map[concept.name.lower()] = stored.get("id", "")
                except Exception as e:
                    logger.warning(f"Failed to store concept {concept.name}: {e}")

            # Store relationships
            stored_relationships = []

            for rel in result.relationships:
                try:
                    from_id = concept_id_map.get(rel.from_concept.lower())
                    to_id = concept_id_map.get(rel.to_concept.lower())

                    if from_id and to_id:
                        await self.repo.create_concept_relation(
                            from_concept_id=from_id,
                            to_concept_id=to_id,
                            relation_type=rel.type,
                            weight=rel.confidence,
                        )
                        stored_relationships.append({
                            "from": rel.from_concept,
                            "to": rel.to_concept,
                            "type": rel.type,
                        })
                except Exception as e:
                    logger.warning(f"Failed to store relationship: {e}")

            return ExtractionResult(
                concepts_extracted=len(stored_concepts),
                relationships_extracted=len(stored_relationships),
                concepts=stored_concepts,
                relationships=stored_relationships,
            )

        except Exception as e:
            logger.error(f"Ontology extraction failed: {e}")
            return ExtractionResult(
                concepts_extracted=0,
                relationships_extracted=0,
                concepts=[],
                relationships=[],
            )

    async def classify_chunk(
        self,
        content: str,
        heading_context: str = "",
    ) -> ChunkClassification:
        """
        Classify a chunk's semantic type.
        
        Args:
            content: Chunk content
            heading_context: Parent heading context
            
        Returns:
            ChunkClassification with type and metadata
        """
        prompt = build_chunk_classification_prompt(
            content=content,
            heading_context=heading_context,
        )

        try:
            result = await self.llm.complete_structured(
                messages=[
                    {"role": "system", "content": "You are a documentation analyzer."},
                    {"role": "user", "content": prompt},
                ],
                response_model=ChunkClassification,
                model=self.llm.settings.fast_model,
                temperature=0.2,
            )
            return result
        except Exception as e:
            logger.warning(f"Chunk classification failed: {e}")
            return ChunkClassification(
                semantic_type="general",
                confidence=0.5,
                key_topics=[],
                complexity="intermediate",
            )

    async def link_chunk_to_concepts(
        self,
        chunk_id: str,
        content: str,
        source_id: str,
    ) -> int:
        """
        Link a chunk to relevant concepts.
        
        Args:
            chunk_id: The chunk ID
            content: Chunk content
            source_id: Source ID for concept lookup
            
        Returns:
            Number of links created
        """
        # Get existing concepts for this source
        concepts = await self.repo.get_concepts(source_id, limit=500)

        if not concepts:
            return 0

        # Simple keyword matching for concept linking
        # In production, could use embeddings for better matching
        content_lower = content.lower()
        links_created = 0

        for concept in concepts:
            concept_name = concept.get("name", "").lower()
            if concept_name and concept_name in content_lower:
                try:
                    await self.repo.link_chunk_to_concept(
                        chunk_id=chunk_id,
                        concept_id=concept["id"],
                        confidence=0.8,
                    )
                    links_created += 1
                except Exception as e:
                    logger.debug(f"Failed to link chunk to concept: {e}")

        return links_created


# Global extractor instance
_extractor: OntologyExtractor | None = None


def get_extractor() -> OntologyExtractor:
    """Get the global ontology extractor."""
    global _extractor
    if _extractor is None:
        _extractor = OntologyExtractor()
    return _extractor


async def extract_ontology(
    source_id: str,
    source_name: str,
    page_title: str,
    page_url: str,
    content: str,
) -> ExtractionResult:
    """
    Convenience function to extract ontology from content.
    
    Args:
        source_id: Documentation source ID
        source_name: Name of the documentation
        page_title: Page title
        page_url: Page URL
        content: Page content
        
    Returns:
        ExtractionResult
    """
    extractor = get_extractor()
    return await extractor.extract_from_content(
        source_id=source_id,
        source_name=source_name,
        page_title=page_title,
        page_url=page_url,
        content=content,
    )
