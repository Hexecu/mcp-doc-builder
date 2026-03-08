"""
Relationship linker for connecting entities in the knowledge graph.
Handles page-to-page links, chunk-to-concept links, and concept relationships.
"""

import logging
from dataclasses import dataclass
from typing import Any

from doc_builder.crawler.parser import ExtractedLink, ParsedPage
from doc_builder.kg import get_repository
from doc_builder.utils import generate_id

logger = logging.getLogger(__name__)


@dataclass
class LinkStats:
    """Statistics for linking operations."""

    page_links_created: int = 0
    chunk_concept_links: int = 0
    concept_relations: int = 0


class RelationshipLinker:
    """
    Handles relationship creation in the knowledge graph.
    """

    def __init__(self):
        self.repo = get_repository()

    async def link_pages(
        self,
        source_page: ParsedPage,
        crawled_urls: set[str],
    ) -> int:
        """
        Create LINKS_TO relationships between pages.
        
        Args:
            source_page: The source page with links
            crawled_urls: Set of URLs that have been crawled
            
        Returns:
            Number of links created
        """
        links_created = 0

        for link in source_page.links:
            # Only link to pages we've actually crawled
            if link.url not in crawled_urls:
                continue

            try:
                await self.repo.create_page_link(
                    from_url=source_page.url,
                    to_url=link.url,
                    anchor_text=link.anchor_text,
                )
                links_created += 1
            except Exception as e:
                logger.debug(f"Failed to create page link: {e}")

        return links_created

    async def link_chunks_to_concepts(
        self,
        source_id: str,
        page_id: str,
    ) -> int:
        """
        Link chunks from a page to relevant concepts.
        
        Uses keyword matching to find concept mentions in chunks.
        
        Args:
            source_id: Documentation source ID
            page_id: Page ID to process
            
        Returns:
            Number of links created
        """
        # Get concepts for this source
        concepts = await self.repo.get_concepts(source_id, limit=1000)

        if not concepts:
            return 0

        # Build concept lookup
        concept_keywords = {}
        for concept in concepts:
            name = concept.get("name", "").lower()
            if name:
                concept_keywords[name] = concept["id"]
                # Also add aliases if we tracked them
                for alias in concept.get("aliases", []):
                    concept_keywords[alias.lower()] = concept["id"]

        # Get chunks for the page
        from doc_builder.kg.neo4j import get_neo4j_client
        client = get_neo4j_client()

        query = """
        MATCH (p:DocPage {id: $page_id})-[:HAS_CHUNK]->(c:DocChunk)
        RETURN c.id as chunk_id, c.content as content
        """
        chunks = await client.execute_query(query, {"page_id": page_id})

        links_created = 0

        for chunk in chunks:
            content_lower = chunk["content"].lower()
            chunk_id = chunk["chunk_id"]

            # Find concept mentions
            for keyword, concept_id in concept_keywords.items():
                if keyword in content_lower:
                    try:
                        await self.repo.link_chunk_to_concept(
                            chunk_id=chunk_id,
                            concept_id=concept_id,
                            confidence=0.7,
                        )
                        links_created += 1
                    except Exception as e:
                        logger.debug(f"Failed to link chunk to concept: {e}")

        return links_created

    async def infer_concept_relationships(
        self,
        source_id: str,
    ) -> int:
        """
        Infer relationships between concepts based on co-occurrence.
        
        If two concepts appear in the same chunk frequently, they're likely related.
        
        Args:
            source_id: Documentation source ID
            
        Returns:
            Number of relationships created
        """
        from doc_builder.kg.neo4j import get_neo4j_client
        client = get_neo4j_client()

        # Find concepts that co-occur in chunks
        query = """
        MATCH (c1:DocConcept {source_id: $source_id})<-[:MENTIONS]-(chunk:DocChunk)-[:MENTIONS]->(c2:DocConcept {source_id: $source_id})
        WHERE c1.id < c2.id  // Avoid duplicates
        WITH c1, c2, count(chunk) as co_occurrences
        WHERE co_occurrences >= 2  // Minimum co-occurrences
        RETURN c1.id as concept1_id, c2.id as concept2_id, co_occurrences
        ORDER BY co_occurrences DESC
        LIMIT 100
        """

        co_occurrences = await client.execute_query(query, {"source_id": source_id})
        relations_created = 0

        for item in co_occurrences:
            try:
                # Calculate weight based on co-occurrences
                weight = min(1.0, item["co_occurrences"] / 10)

                await self.repo.create_concept_relation(
                    from_concept_id=item["concept1_id"],
                    to_concept_id=item["concept2_id"],
                    relation_type="related_to",
                    weight=weight,
                )
                relations_created += 1
            except Exception as e:
                logger.debug(f"Failed to create concept relation: {e}")

        return relations_created

    async def build_source_graph(
        self,
        source_id: str,
        pages: list[ParsedPage],
    ) -> LinkStats:
        """
        Build the complete relationship graph for a source.
        
        Args:
            source_id: Documentation source ID
            pages: List of crawled pages
            
        Returns:
            LinkStats with operation results
        """
        stats = LinkStats()
        crawled_urls = {page.url for page in pages}

        # Create page-to-page links
        for page in pages:
            links = await self.link_pages(page, crawled_urls)
            stats.page_links_created += links

        logger.info(f"Created {stats.page_links_created} page-to-page links")

        # Link chunks to concepts
        for page in pages:
            page_id = generate_id("page", page.url)
            links = await self.link_chunks_to_concepts(source_id, page_id)
            stats.chunk_concept_links += links

        logger.info(f"Created {stats.chunk_concept_links} chunk-to-concept links")

        # Infer concept relationships
        relations = await self.infer_concept_relationships(source_id)
        stats.concept_relations = relations

        logger.info(f"Inferred {stats.concept_relations} concept relationships")

        return stats


# Global linker instance
_linker: RelationshipLinker | None = None


def get_linker() -> RelationshipLinker:
    """Get the global relationship linker."""
    global _linker
    if _linker is None:
        _linker = RelationshipLinker()
    return _linker
