"""
LLM-powered ontology extraction from documentation content.
Extracts concepts and relationships to build a dynamic knowledge graph.

Features:
- Smart retry with exponential backoff
- Partial extraction on validation failures
- Parallel processing for multiple pages
- JSON repair for malformed responses
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
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

# Configuration
MAX_RETRIES = 3
PARALLEL_EXTRACTIONS = 2  # Conservative to avoid rate limits
BACKOFF_BASE = 2  # Exponential backoff base


@dataclass
class ExtractionResult:
    """Result of ontology extraction for a page."""

    concepts_extracted: int
    relationships_extracted: int
    concepts: list[dict] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class OntologyExtractor:
    """
    Extracts concepts and relationships from documentation using LLM.
    
    Features:
    - Smart retry with exponential backoff
    - JSON repair for malformed responses
    - Partial extraction on validation failures
    - Parallel processing for batches
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
        Extract ontology from page content with smart retry.
        
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
            )

        # Build prompt
        prompt = build_ontology_extraction_prompt(
            source_name=source_name,
            page_title=page_title,
            page_url=page_url,
            content=content,
        )

        errors: list[str] = []
        result = None

        # Retry loop with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                result = await self._extract_with_llm(prompt, attempt)
                if result:
                    break
            except json.JSONDecodeError as e:
                errors.append(f"JSON parse error (attempt {attempt + 1}): {str(e)[:100]}")
                logger.warning(f"JSON parse error on attempt {attempt + 1}: {e}")
                
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BACKOFF_BASE ** attempt)
                    continue
                    
            except Exception as e:
                error_msg = str(e)
                errors.append(f"Extraction error (attempt {attempt + 1}): {error_msg[:100]}")
                
                # Check for rate limit
                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    wait_time = BACKOFF_BASE ** (attempt + 2)  # Longer wait for rate limits
                    logger.warning(f"Rate limit hit, waiting {wait_time}s")
                    await asyncio.sleep(wait_time)
                elif attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BACKOFF_BASE ** attempt)
                    
                if attempt == MAX_RETRIES - 1:
                    logger.error(f"Ontology extraction failed after {MAX_RETRIES} attempts: {e}")

        # If we have no result, return empty
        if not result:
            return ExtractionResult(
                concepts_extracted=0,
                relationships_extracted=0,
                errors=errors,
            )

        # Store extracted data
        return await self._store_extraction(source_id, result, errors)

    async def _extract_with_llm(
        self,
        prompt: str,
        attempt: int = 0,
    ) -> OntologyExtractionResult | None:
        """
        Perform LLM extraction with JSON repair fallback.
        
        Args:
            prompt: The extraction prompt
            attempt: Current attempt number (affects prompt)
            
        Returns:
            Parsed OntologyExtractionResult or None
        """
        messages = [
            {"role": "system", "content": ONTOLOGY_EXTRACTOR_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        
        # On retry, add explicit JSON instruction
        if attempt > 0:
            messages.append({
                "role": "user",
                "content": "IMPORTANT: Respond with valid JSON only. No markdown, no code blocks, just pure JSON."
            })

        try:
            result = await self.llm.complete_structured(
                messages=messages,
                response_model=OntologyExtractionResult,
                model=self.llm.settings.reason_model,
                temperature=0.3,
            )
            return result
            
        except json.JSONDecodeError as e:
            # Try to repair JSON
            logger.debug(f"Attempting JSON repair")
            return await self._try_repair_json(messages)
            
        except Exception as e:
            # Check if it's a validation error - try partial extraction
            error_str = str(e)
            if "validation" in error_str.lower():
                logger.debug(f"Validation error, attempting partial extraction")
                return await self._try_partial_extraction(messages)
            raise

    async def _try_repair_json(
        self,
        messages: list[dict],
    ) -> OntologyExtractionResult | None:
        """
        Try to get a clean JSON response from LLM.
        
        Args:
            messages: The conversation messages
            
        Returns:
            Parsed result or None
        """
        try:
            # Ask for raw text response
            response = await self.llm.complete(
                messages=messages + [{
                    "role": "user",
                    "content": "Please provide the JSON response again, ensuring it is valid JSON with no markdown formatting."
                }],
                model=self.llm.settings.fast_model,
                temperature=0.1,
            )
            
            # Clean up common JSON issues
            cleaned = self._clean_json_response(response)
            
            # Try to parse
            data = json.loads(cleaned)
            return OntologyExtractionResult(**data)
            
        except Exception as e:
            logger.debug(f"JSON repair failed: {e}")
            return None

    def _clean_json_response(self, response: str) -> str:
        """
        Clean up common JSON formatting issues.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Cleaned JSON string
        """
        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        # Find JSON object boundaries
        start = response.find('{')
        end = response.rfind('}')
        
        if start >= 0 and end > start:
            response = response[start:end + 1]
        
        # Fix common issues
        response = response.replace('\n', ' ')
        response = re.sub(r',\s*}', '}', response)  # Trailing commas
        response = re.sub(r',\s*]', ']', response)  # Trailing commas in arrays
        
        return response.strip()

    async def _try_partial_extraction(
        self,
        messages: list[dict],
    ) -> OntologyExtractionResult | None:
        """
        Try to extract partial data when validation fails.
        
        Args:
            messages: The conversation messages
            
        Returns:
            Partial result or None
        """
        try:
            response = await self.llm.complete(
                messages=messages,
                model=self.llm.settings.fast_model,
                temperature=0.1,
            )
            
            cleaned = self._clean_json_response(response)
            data = json.loads(cleaned)
            
            # Extract what we can
            concepts = []
            relationships = []
            
            # Try to get concepts
            raw_concepts = data.get("concepts", [])
            for c in raw_concepts:
                if isinstance(c, dict) and c.get("name"):
                    try:
                        concepts.append(ExtractedConcept(
                            name=c.get("name", ""),
                            description=c.get("description", ""),
                            category=c.get("category", "concept"),
                            confidence=float(c.get("confidence", 0.8)),
                            aliases=c.get("aliases", []),
                        ))
                    except Exception:
                        pass
            
            # Try to get relationships
            raw_rels = data.get("relationships", [])
            for r in raw_rels:
                if isinstance(r, dict):
                    # Handle both alias names (from/to) and actual names (from_concept/to_concept)
                    from_val = r.get("from") or r.get("from_concept")
                    to_val = r.get("to") or r.get("to_concept")
                    type_val = r.get("type")
                    
                    if from_val and to_val and type_val:
                        try:
                            relationships.append(ExtractedRelationship(
                                **{"from": from_val, "to": to_val},
                                type=type_val,
                                description=r.get("description", ""),
                                confidence=float(r.get("confidence", 0.8)),
                            ))
                        except Exception:
                            pass
            
            if concepts or relationships:
                logger.info(f"Partial extraction: {len(concepts)} concepts, {len(relationships)} relationships")
                return OntologyExtractionResult(
                    concepts=concepts,
                    relationships=relationships,
                )
                
            return None
            
        except Exception as e:
            logger.debug(f"Partial extraction failed: {e}")
            return None

    async def _store_extraction(
        self,
        source_id: str,
        result: OntologyExtractionResult,
        errors: list[str],
    ) -> ExtractionResult:
        """
        Store extracted concepts and relationships in the graph.
        
        Args:
            source_id: Documentation source ID
            result: Extraction result to store
            errors: Any errors encountered
            
        Returns:
            ExtractionResult with storage stats
        """
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
                logger.debug(f"Failed to store concept {concept.name}: {e}")

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
                logger.debug(f"Failed to store relationship: {e}")

        return ExtractionResult(
            concepts_extracted=len(stored_concepts),
            relationships_extracted=len(stored_relationships),
            concepts=stored_concepts,
            relationships=stored_relationships,
            errors=errors,
        )

    async def extract_batch(
        self,
        pages: list[dict],
        source_id: str,
        source_name: str,
    ) -> list[ExtractionResult]:
        """
        Extract ontology from multiple pages in parallel.
        
        Args:
            pages: List of page dicts with title, url, content
            source_id: Documentation source ID
            source_name: Name of the documentation
            
        Returns:
            List of ExtractionResults
        """
        semaphore = asyncio.Semaphore(PARALLEL_EXTRACTIONS)
        
        async def extract_with_semaphore(page: dict) -> ExtractionResult:
            async with semaphore:
                return await self.extract_from_content(
                    source_id=source_id,
                    source_name=source_name,
                    page_title=page.get("title", ""),
                    page_url=page.get("url", ""),
                    content=page.get("content", ""),
                )
        
        tasks = [extract_with_semaphore(page) for page in pages]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                logger.error(f"Batch extraction failed for page {i}: {result}")
                final_results.append(ExtractionResult(
                    concepts_extracted=0,
                    relationships_extracted=0,
                    errors=[str(result)],
                ))
            else:
                final_results.append(result)
        
        return final_results

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
            logger.debug(f"Chunk classification failed: {e}")
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
        concepts = await self.repo.get_concepts(source_id, limit=500)

        if not concepts:
            return 0

        content_lower = content.lower()
        links_created = 0

        for concept in concepts:
            concept_name = concept.get("name", "").lower()
            if concept_name and len(concept_name) > 2 and concept_name in content_lower:
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


async def extract_ontology_batch(
    pages: list[dict],
    source_id: str,
    source_name: str,
) -> list[ExtractionResult]:
    """
    Convenience function to extract ontology from multiple pages.
    
    Args:
        pages: List of page dicts
        source_id: Documentation source ID
        source_name: Name of the documentation
        
    Returns:
        List of ExtractionResults
    """
    extractor = get_extractor()
    return await extractor.extract_batch(
        pages=pages,
        source_id=source_id,
        source_name=source_name,
    )
