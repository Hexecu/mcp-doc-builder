"""
Ontology module for concept extraction and relationship building.
"""

from doc_builder.ontology.extractor import (
    ExtractionResult,
    OntologyExtractor,
    extract_ontology,
    get_extractor,
)
from doc_builder.ontology.linker import (
    LinkStats,
    RelationshipLinker,
    get_linker,
)
from doc_builder.ontology.metatag import (
    ProcessedMetatag,
    extract_structured_data,
    get_page_summary_from_metatags,
    process_metatags,
    store_page_metatags,
)

__all__ = [
    # Extractor
    "OntologyExtractor",
    "ExtractionResult",
    "extract_ontology",
    "get_extractor",
    # Linker
    "RelationshipLinker",
    "LinkStats",
    "get_linker",
    # Metatag
    "ProcessedMetatag",
    "process_metatags",
    "store_page_metatags",
    "extract_structured_data",
    "get_page_summary_from_metatags",
]
