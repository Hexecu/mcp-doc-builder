// ═══════════════════════════════════════════════════════════════════════════════
// MCP Doc Builder - Neo4j Schema
// ═══════════════════════════════════════════════════════════════════════════════
// Documentation graph with vector embeddings and dynamic ontology
// All labels prefixed with "Doc" to enable sharing with other KG namespaces

// ─────────────────────────────────────────────────────────────────────────────────
// Node Constraints (uniqueness)
// ─────────────────────────────────────────────────────────────────────────────────

// Documentation Source (root URL)
CREATE CONSTRAINT doc_source_id IF NOT EXISTS
FOR (s:DocSource) REQUIRE s.id IS UNIQUE;

// Web Page
CREATE CONSTRAINT doc_page_id IF NOT EXISTS
FOR (p:DocPage) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT doc_page_url IF NOT EXISTS
FOR (p:DocPage) REQUIRE p.url IS UNIQUE;

// Document Chunk (for vectorization)
CREATE CONSTRAINT doc_chunk_id IF NOT EXISTS
FOR (c:DocChunk) REQUIRE c.id IS UNIQUE;

// Metatag
CREATE CONSTRAINT doc_metatag_id IF NOT EXISTS
FOR (m:DocMetatag) REQUIRE m.id IS UNIQUE;

// Concept (ontology)
CREATE CONSTRAINT doc_concept_id IF NOT EXISTS
FOR (c:DocConcept) REQUIRE c.id IS UNIQUE;

// Relationship Type (dynamic ontology)
CREATE CONSTRAINT doc_reltype_id IF NOT EXISTS
FOR (r:DocRelationType) REQUIRE r.id IS UNIQUE;

// Crawl Job
CREATE CONSTRAINT doc_crawljob_id IF NOT EXISTS
FOR (j:DocCrawlJob) REQUIRE j.id IS UNIQUE;

// ─────────────────────────────────────────────────────────────────────────────────
// Indexes for Performance
// ─────────────────────────────────────────────────────────────────────────────────

// DocSource indexes
CREATE INDEX doc_source_domain IF NOT EXISTS
FOR (s:DocSource) ON (s.domain);

CREATE INDEX doc_source_status IF NOT EXISTS
FOR (s:DocSource) ON (s.status);

// DocPage indexes
CREATE INDEX doc_page_source IF NOT EXISTS
FOR (p:DocPage) ON (p.source_id);

CREATE INDEX doc_page_depth IF NOT EXISTS
FOR (p:DocPage) ON (p.depth);

CREATE INDEX doc_page_crawled IF NOT EXISTS
FOR (p:DocPage) ON (p.crawled_at);

// DocChunk indexes
CREATE INDEX doc_chunk_page IF NOT EXISTS
FOR (c:DocChunk) ON (c.page_id);

CREATE INDEX doc_chunk_type IF NOT EXISTS
FOR (c:DocChunk) ON (c.semantic_type);

// DocConcept indexes
CREATE INDEX doc_concept_category IF NOT EXISTS
FOR (c:DocConcept) ON (c.category);

CREATE INDEX doc_concept_source IF NOT EXISTS
FOR (c:DocConcept) ON (c.source_id);

// ─────────────────────────────────────────────────────────────────────────────────
// Fulltext Indexes for Search
// ─────────────────────────────────────────────────────────────────────────────────

// Page content search
CREATE FULLTEXT INDEX doc_page_fulltext IF NOT EXISTS
FOR (p:DocPage) ON EACH [p.title, p.description, p.content_preview];

// Chunk content search
CREATE FULLTEXT INDEX doc_chunk_fulltext IF NOT EXISTS
FOR (c:DocChunk) ON EACH [c.content];

// Concept search
CREATE FULLTEXT INDEX doc_concept_fulltext IF NOT EXISTS
FOR (c:DocConcept) ON EACH [c.name, c.description];

// ─────────────────────────────────────────────────────────────────────────────────
// Vector Index for Semantic Search
// ─────────────────────────────────────────────────────────────────────────────────

// Vector index on DocChunk embeddings (3072 dimensions for gemini-embedding-001)
// NOTE: This must be created separately after data exists, using:
// CALL db.index.vector.createNodeIndex(
//   'doc_chunk_embeddings',
//   'DocChunk',
//   'embedding',
//   3072,
//   'cosine'
// )

// ─────────────────────────────────────────────────────────────────────────────────
// Sample Data Model Reference
// ─────────────────────────────────────────────────────────────────────────────────

// DocSource {
//   id: string,           // "source_abc123"
//   root_url: string,     // "https://docs.example.com"
//   domain: string,       // "docs.example.com"
//   name: string,         // "Example Docs"
//   description: string,  // Optional description
//   last_crawled: datetime,
//   total_pages: int,
//   status: string        // "pending", "crawling", "completed", "failed"
// }

// DocPage {
//   id: string,           // "page_abc123"
//   url: string,          // "https://docs.example.com/guide/intro"
//   source_id: string,    // Reference to DocSource
//   title: string,
//   description: string,
//   content_preview: string,  // First ~500 chars
//   content_hash: string,     // SHA256 for change detection
//   crawled_at: datetime,
//   depth: int,           // Hop count from root
//   language: string,     // "en", "it", etc.
//   word_count: int
// }

// DocChunk {
//   id: string,           // "chunk_abc123"
//   page_id: string,      // Reference to DocPage
//   content: string,      // Chunk text
//   embedding: list<float>,  // 768-dim vector
//   chunk_index: int,     // Position in page
//   token_count: int,
//   semantic_type: string,   // "code", "explanation", "api_reference", etc.
//   heading_context: string  // Parent heading(s) for context
// }

// DocMetatag {
//   id: string,           // "meta_abc123"
//   key: string,          // "og:title", "keywords", etc.
//   value: string
// }

// DocConcept {
//   id: string,           // "concept_abc123"
//   name: string,         // "React Hook"
//   description: string,
//   category: string,     // "api", "pattern", "entity", "action"
//   confidence: float,    // 0.0-1.0
//   source_id: string,    // Which DocSource defined this
//   mention_count: int    // How many times mentioned
// }

// DocRelationType {
//   id: string,           // "reltype_abc123"
//   name: string,         // "uses", "extends", "requires"
//   description: string,
//   source_concept_type: string,
//   target_concept_type: string
// }

// DocCrawlJob {
//   id: string,           // "job_abc123"
//   source_id: string,
//   started_at: datetime,
//   completed_at: datetime,
//   status: string,       // "running", "completed", "failed"
//   pages_crawled: int,
//   pages_failed: int,
//   error_message: string
// }

// ─────────────────────────────────────────────────────────────────────────────────
// Relationships
// ─────────────────────────────────────────────────────────────────────────────────

// (DocSource)-[:CONTAINS]->(DocPage)
// (DocPage)-[:LINKS_TO {anchor_text: string}]->(DocPage)
// (DocPage)-[:HAS_CHUNK]->(DocChunk)
// (DocPage)-[:HAS_METATAG]->(DocMetatag)
// (DocChunk)-[:MENTIONS {confidence: float}]->(DocConcept)
// (DocConcept)-[:RELATES_TO {type: string, weight: float}]->(DocConcept)
// (DocSource)-[:DEFINES]->(DocConcept)
// (DocSource)-[:HAS_JOB]->(DocCrawlJob)
