"""
Smart document chunking with semantic awareness.
Splits documents into chunks suitable for embedding and retrieval.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Sequence

from doc_builder.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Hard limit to stay well under embedding model context window (8192 for text-embedding-3-small)
ABSOLUTE_MAX_TOKENS = 2000


@dataclass
class Chunk:
    """A document chunk ready for embedding."""

    content: str
    index: int
    token_count: int
    semantic_type: str = "general"
    heading_context: str = ""
    start_char: int = 0
    end_char: int = 0


@dataclass
class ChunkerConfig:
    """Configuration for the chunker."""

    target_size: int = 500  # Target tokens per chunk (reduced from 800)
    overlap: int = 50  # Overlap tokens between chunks (reduced from 100)
    min_size: int = 50  # Minimum chunk size
    max_size: int = 1000  # Maximum chunk size (reduced from 1500)


class SmartChunker:
    """
    Intelligent document chunker that respects semantic boundaries.
    
    Features:
    - Respects heading boundaries
    - Keeps code blocks intact
    - Maintains context with overlapping
    - Classifies chunk types
    """

    # Patterns for identifying semantic boundaries
    HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```", re.MULTILINE)
    LIST_PATTERN = re.compile(r"^[-*+]\s+.+$", re.MULTILINE)

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        settings: Settings | None = None,
    ):
        """
        Initialize the chunker.
        
        Args:
            config: Chunker configuration
            settings: Application settings
        """
        settings = settings or get_settings()
        self.config = config or ChunkerConfig(
            target_size=settings.vector_chunk_size,
            overlap=settings.vector_chunk_overlap,
        )
        self._encoder = None

    def _get_encoder(self):
        """Get or create tiktoken encoder."""
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                logger.warning("tiktoken not available, using character-based estimation")
                self._encoder = None
        return self._encoder

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.
        
        Args:
            text: Text to count
            
        Returns:
            Token count
        """
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        # Fallback: rough estimate
        return len(text) // 4

    def chunk(
        self,
        content: str,
        heading_context: str = "",
    ) -> list[Chunk]:
        """
        Split content into semantic chunks.
        
        Args:
            content: Text content to chunk
            heading_context: Optional parent heading context
            
        Returns:
            List of Chunk objects
        """
        if not content or not content.strip():
            return []

        # First, identify all code blocks and protect them
        protected_content, code_blocks = self._protect_code_blocks(content)

        # Find natural split points
        split_points = self._find_split_points(protected_content)

        # Create initial chunks based on split points
        raw_chunks = self._split_at_points(protected_content, split_points)

        # Restore code blocks
        raw_chunks = [self._restore_code_blocks(c, code_blocks) for c in raw_chunks]

        # Merge small chunks and split large ones
        sized_chunks = self._enforce_size_limits(raw_chunks)

        # HARD LIMIT: Ensure no chunk exceeds absolute max tokens
        safe_chunks = self._enforce_absolute_limit(sized_chunks)

        # Add overlap between chunks
        overlapped_chunks = self._add_overlap(safe_chunks)

        # Create final Chunk objects
        chunks = []
        char_offset = 0

        for i, text in enumerate(overlapped_chunks):
            token_count = self.count_tokens(text)
            semantic_type = self._classify_chunk(text)

            chunks.append(Chunk(
                content=text,
                index=i,
                token_count=token_count,
                semantic_type=semantic_type,
                heading_context=heading_context,
                start_char=char_offset,
                end_char=char_offset + len(text),
            ))

            # Update offset (accounting for overlap)
            char_offset += len(text) - (self.config.overlap * 4)  # Rough char estimate

        return chunks

    def _protect_code_blocks(self, content: str) -> tuple[str, dict[str, str]]:
        """Replace code blocks with placeholders."""
        code_blocks = {}
        counter = 0

        def replacer(match):
            nonlocal counter
            placeholder = f"__CODE_BLOCK_{counter}__"
            code_blocks[placeholder] = match.group(0)
            counter += 1
            return placeholder

        protected = self.CODE_BLOCK_PATTERN.sub(replacer, content)
        return protected, code_blocks

    def _restore_code_blocks(self, text: str, code_blocks: dict[str, str]) -> str:
        """Restore code blocks from placeholders."""
        for placeholder, code in code_blocks.items():
            text = text.replace(placeholder, code)
        return text

    def _find_split_points(self, content: str) -> list[int]:
        """
        Find natural split points in content.
        
        Returns positions where splitting is appropriate.
        """
        split_points = []

        # Heading positions (high priority)
        for match in self.HEADING_PATTERN.finditer(content):
            split_points.append((match.start(), 3))  # Priority 3 (highest)

        # Double newlines (paragraph breaks)
        for match in re.finditer(r"\n\n+", content):
            split_points.append((match.start(), 2))

        # Single newlines after sentences
        for match in re.finditer(r"\.\n", content):
            split_points.append((match.end(), 1))

        # Sort by position, then priority (descending)
        split_points.sort(key=lambda x: (x[0], -x[1]))

        # Return just positions, removing duplicates
        positions = []
        last_pos = -1
        for pos, _ in split_points:
            if pos > last_pos + 50:  # Minimum distance between splits
                positions.append(pos)
                last_pos = pos

        return positions

    def _split_at_points(self, content: str, split_points: list[int]) -> list[str]:
        """Split content at the given points."""
        if not split_points:
            return [content]

        chunks = []
        start = 0

        for point in split_points:
            if point > start:
                chunk = content[start:point].strip()
                if chunk:
                    chunks.append(chunk)
                start = point

        # Add final chunk
        if start < len(content):
            chunk = content[start:].strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    def _enforce_size_limits(self, chunks: list[str]) -> list[str]:
        """Merge small chunks and split large ones."""
        result = []
        current = ""

        for chunk in chunks:
            token_count = self.count_tokens(chunk)

            # If chunk is too large, split it
            if token_count > self.config.max_size:
                # Flush current
                if current:
                    result.append(current)
                    current = ""

                # Split large chunk
                split_chunks = self._split_large_chunk(chunk)
                result.extend(split_chunks)
                continue

            # If adding this chunk exceeds target, start new chunk
            combined = f"{current}\n\n{chunk}" if current else chunk
            combined_tokens = self.count_tokens(combined)

            if combined_tokens > self.config.target_size:
                if current:
                    result.append(current)
                current = chunk
            else:
                current = combined

        # Flush remaining
        if current:
            result.append(current)

        return result

    def _split_large_chunk(self, chunk: str) -> list[str]:
        """Split a chunk that exceeds max size."""
        result = []
        target_chars = self.config.target_size * 4  # Rough estimate

        # Split by sentences
        sentences = re.split(r"(?<=[.!?])\s+", chunk)
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) > target_chars:
                if current:
                    result.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}" if current else sentence

        if current:
            result.append(current.strip())

        return result

    def _enforce_absolute_limit(self, chunks: list[str]) -> list[str]:
        """
        Enforce absolute maximum token limit on all chunks.
        
        This is a safety net to ensure no chunk exceeds the embedding model's
        context window, even after all other processing.
        """
        result = []
        
        for chunk in chunks:
            token_count = self.count_tokens(chunk)
            
            if token_count <= ABSOLUTE_MAX_TOKENS:
                result.append(chunk)
            else:
                # Force split the chunk
                logger.debug(f"Chunk exceeds {ABSOLUTE_MAX_TOKENS} tokens ({token_count}), force splitting")
                split_chunks = self._force_split_chunk(chunk, ABSOLUTE_MAX_TOKENS)
                result.extend(split_chunks)
        
        return result

    def _force_split_chunk(self, chunk: str, max_tokens: int) -> list[str]:
        """
        Force split a chunk that exceeds the absolute limit.
        
        Uses aggressive splitting to ensure compliance.
        """
        result = []
        encoder = self._get_encoder()
        
        if encoder:
            # Token-based splitting
            tokens = encoder.encode(chunk)
            
            for i in range(0, len(tokens), max_tokens - 50):  # Small buffer
                chunk_tokens = tokens[i:i + max_tokens - 50]
                chunk_text = encoder.decode(chunk_tokens)
                if chunk_text.strip():
                    result.append(chunk_text.strip())
        else:
            # Character-based fallback
            max_chars = max_tokens * 4
            
            for i in range(0, len(chunk), max_chars - 200):
                chunk_text = chunk[i:i + max_chars - 200]
                # Try to end at sentence boundary
                last_period = chunk_text.rfind('.')
                if last_period > len(chunk_text) // 2:
                    chunk_text = chunk_text[:last_period + 1]
                if chunk_text.strip():
                    result.append(chunk_text.strip())
        
        return result if result else [chunk[:max_tokens * 4]]  # Fallback truncation

    def _add_overlap(self, chunks: list[str]) -> list[str]:
        """Add overlapping content between chunks."""
        if len(chunks) <= 1 or self.config.overlap <= 0:
            return chunks

        result = []
        overlap_chars = self.config.overlap * 4  # Rough estimate

        for i, chunk in enumerate(chunks):
            if i > 0:
                # Add end of previous chunk as prefix
                prev_chunk = chunks[i - 1]
                overlap_text = prev_chunk[-overlap_chars:]

                # Try to start at a word boundary
                space_idx = overlap_text.find(" ")
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx + 1:]

                chunk = f"...{overlap_text}\n\n{chunk}"

            result.append(chunk)

        return result

    def _classify_chunk(self, content: str) -> str:
        """
        Classify the semantic type of a chunk.
        
        Returns:
            One of: code_example, api_reference, conceptual, tutorial,
                   configuration, troubleshooting, overview
        """
        content_lower = content.lower()

        # Check for code blocks
        if "```" in content or content.count("    ") > 5:
            return "code_example"

        # Check for API patterns
        api_patterns = ["parameter", "returns", "arguments", "method", "function(", "class "]
        if any(pattern in content_lower for pattern in api_patterns):
            return "api_reference"

        # Check for configuration patterns
        config_patterns = ["config", "setting", "option", "environment", ".env", "yaml", "json"]
        if any(pattern in content_lower for pattern in config_patterns):
            return "configuration"

        # Check for tutorial patterns
        tutorial_patterns = ["step", "first,", "then,", "next,", "finally,", "1.", "2.", "3."]
        if any(pattern in content_lower for pattern in tutorial_patterns):
            return "tutorial"

        # Check for troubleshooting patterns
        trouble_patterns = ["error", "issue", "problem", "fix", "solution", "debug"]
        if any(pattern in content_lower for pattern in trouble_patterns):
            return "troubleshooting"

        # Check for overview patterns
        if content.startswith("#") or "introduction" in content_lower or "overview" in content_lower:
            return "overview"

        return "conceptual"


def chunk_document(
    content: str,
    heading_context: str = "",
    target_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """
    Convenience function to chunk a document.
    
    Args:
        content: Document content
        heading_context: Optional heading context
        target_size: Optional target chunk size in tokens
        overlap: Optional overlap size in tokens
        
    Returns:
        List of Chunk objects
    """
    config = None
    if target_size or overlap:
        config = ChunkerConfig(
            target_size=target_size or 800,
            overlap=overlap or 100,
        )

    chunker = SmartChunker(config=config)
    return chunker.chunk(content, heading_context)
