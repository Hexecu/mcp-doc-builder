"""
Embedding client for generating vector representations of text.
Uses Gemini text-embedding-004 via LiteLLM.
"""

import asyncio
import logging
from typing import Sequence

from doc_builder.config import Settings, get_settings
from doc_builder.llm import get_llm_client

logger = logging.getLogger(__name__)


class Embedder:
    """
    Client for generating text embeddings.
    
    Uses Gemini text-embedding-004 model which produces 768-dimensional vectors.
    Supports batching for efficiency.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        batch_size: int = 20,
    ):
        """
        Initialize the embedder.
        
        Args:
            settings: Optional settings override
            batch_size: Maximum texts per embedding batch
        """
        self.settings = settings or get_settings()
        self.batch_size = batch_size
        self.llm = get_llm_client()

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self.settings.embedding_dimensions

    @property
    def model(self) -> str:
        """Get the embedding model name."""
        return self.settings.embedding_model

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector (768 dimensions)
        """
        return await self.llm.embed_single(text)

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Process in batches
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]

            try:
                embeddings = await self.llm.embed(list(batch))
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"Batch embedding failed: {e}")
                # Fallback: embed one at a time
                for text in batch:
                    try:
                        embedding = await self.embed(text)
                        all_embeddings.append(embedding)
                    except Exception as inner_e:
                        logger.error(f"Single embedding failed: {inner_e}")
                        # Use zero vector as fallback
                        all_embeddings.append([0.0] * self.dimensions)

        return all_embeddings

    async def embed_with_retry(
        self,
        text: str,
        max_retries: int = 3,
    ) -> list[float]:
        """
        Generate embedding with retry logic.
        
        Args:
            text: Text to embed
            max_retries: Maximum retry attempts
            
        Returns:
            Embedding vector
        """
        last_error = None

        for attempt in range(max_retries):
            try:
                return await self.embed(text)
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(f"Embedding attempt {attempt + 1} failed, waiting {wait_time}s: {e}")
                await asyncio.sleep(wait_time)

        # Return zero vector if all retries failed
        logger.error(f"All embedding attempts failed: {last_error}")
        return [0.0] * self.dimensions

    def truncate_text(self, text: str, max_tokens: int = 2048) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text
        """
        try:
            import tiktoken
            encoder = tiktoken.get_encoding("cl100k_base")
            tokens = encoder.encode(text)

            if len(tokens) <= max_tokens:
                return text

            truncated_tokens = tokens[:max_tokens]
            return encoder.decode(truncated_tokens)

        except ImportError:
            # Fallback: rough character estimate
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[:max_chars]

    async def health_check(self) -> dict:
        """
        Check embedding service health.
        
        Returns:
            Health status dictionary
        """
        try:
            embedding = await self.embed("test")
            return {
                "status": "healthy",
                "model": self.model,
                "dimensions": len(embedding),
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.model,
                "error": str(e),
            }


# Global embedder instance
_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    """Get the global embedder instance."""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
