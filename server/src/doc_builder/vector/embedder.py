"""
Embedding client for generating vector representations of text.
Uses LiteLLM for embeddings with smart retry and parallelization.
"""

import asyncio
import logging
from typing import Sequence

from doc_builder.config import Settings, get_settings
from doc_builder.llm import get_llm_client

logger = logging.getLogger(__name__)

# Maximum tokens for embedding model context window
MAX_EMBEDDING_TOKENS = 8000  # text-embedding-3-small has 8192 limit

# Parallelization limits (conservative to avoid rate limiting)
MAX_PARALLEL_EMBEDDINGS = 3
BATCH_SIZE_DEFAULT = 10


class Embedder:
    """
    Client for generating text embeddings.
    
    Features:
    - Automatic truncation for long texts
    - Smart retry with progressive truncation
    - Parallel processing for batches
    - Zero vector fallback for failures
    """

    def __init__(
        self,
        settings: Settings | None = None,
        batch_size: int = BATCH_SIZE_DEFAULT,
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
        self._encoder = None

    def _get_encoder(self):
        """Lazy load tiktoken encoder."""
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                pass
        return self._encoder

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        return self.settings.embedding_dimensions

    @property
    def model(self) -> str:
        """Get the embedding model name."""
        return self.settings.embedding_model

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        encoder = self._get_encoder()
        if encoder:
            return len(encoder.encode(text))
        # Fallback estimate
        return len(text) // 4

    def truncate_text(self, text: str, max_tokens: int = MAX_EMBEDDING_TOKENS) -> str:
        """
        Truncate text to fit within token limit.
        
        Args:
            text: Text to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated text
        """
        encoder = self._get_encoder()
        
        if encoder:
            tokens = encoder.encode(text)
            if len(tokens) <= max_tokens:
                return text
            truncated_tokens = tokens[:max_tokens]
            return encoder.decode(truncated_tokens)
        else:
            # Fallback: rough character estimate
            max_chars = max_tokens * 4
            if len(text) <= max_chars:
                return text
            return text[:max_chars]

    async def embed_safe(self, text: str) -> list[float]:
        """
        Generate embedding with automatic truncation and retry.
        
        This is the main method that should be used. It handles:
        - Truncation if text is too long
        - Retry with progressive truncation if ContextWindowExceeded
        - Zero vector fallback if all else fails
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        if not text or not text.strip():
            return [0.0] * self.dimensions

        # Pre-truncate to safe limit
        safe_text = self.truncate_text(text, MAX_EMBEDDING_TOKENS - 100)  # Buffer
        
        # Try embedding with progressive fallback
        for attempt, truncate_ratio in enumerate([1.0, 0.5, 0.25]):
            try:
                current_text = safe_text
                if truncate_ratio < 1.0:
                    current_text = self.truncate_text(
                        safe_text, 
                        int(MAX_EMBEDDING_TOKENS * truncate_ratio)
                    )
                    logger.debug(f"Retry embedding with {truncate_ratio*100:.0f}% text")
                
                return await self.llm.embed_single(current_text)
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Check for context window / token limit errors
                if "context" in error_str or "token" in error_str or "length" in error_str:
                    if attempt < 2:
                        logger.warning(f"Text too long, truncating to {int(truncate_ratio*50)}%")
                        continue
                
                # For other errors, retry with backoff
                if attempt < 2:
                    wait_time = 2 ** attempt
                    logger.warning(f"Embedding attempt {attempt + 1} failed, waiting {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                
                logger.error(f"Embedding failed after all attempts: {e}")
                break
        
        # Return zero vector as fallback
        return [0.0] * self.dimensions

    async def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.
        Alias for embed_safe.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        return await self.embed_safe(text)

    async def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts with parallelization.
        
        Args:
            texts: Texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Pre-truncate all texts
        safe_texts = [self.truncate_text(t, MAX_EMBEDDING_TOKENS - 100) for t in texts]
        
        # Process in parallel batches
        all_embeddings: list[list[float]] = []
        
        # Split into mini-batches for parallel processing
        for i in range(0, len(safe_texts), self.batch_size * MAX_PARALLEL_EMBEDDINGS):
            chunk = safe_texts[i:i + self.batch_size * MAX_PARALLEL_EMBEDDINGS]
            
            # Create tasks for parallel batches
            tasks = []
            for j in range(0, len(chunk), self.batch_size):
                batch = chunk[j:j + self.batch_size]
                tasks.append(self._embed_batch_internal(batch))
            
            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect results
            for idx, result in enumerate(results):
                if isinstance(result, BaseException):
                    logger.error(f"Batch embedding failed: {result}")
                    # Return zero vectors for failed batch
                    batch_start = i + idx * self.batch_size
                    batch_end = min(batch_start + self.batch_size, len(safe_texts))
                    batch_len = batch_end - batch_start
                    all_embeddings.extend([[0.0] * self.dimensions] * batch_len)
                else:
                    all_embeddings.extend(result)
        
        # Ensure correct length
        return all_embeddings[:len(texts)]

    async def _embed_batch_internal(self, texts: list[str]) -> list[list[float]]:
        """
        Internal batch embedding with fallback to individual embedding.
        
        Args:
            texts: Batch of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = await self.llm.embed(texts)
            return embeddings
        except Exception as e:
            error_str = str(e).lower()
            
            # If context window exceeded, process individually
            if "context" in error_str or "token" in error_str or "length" in error_str:
                logger.warning(f"Batch too large, falling back to individual embedding")
                return await self._embed_individually(texts)
            
            # For other errors, retry the batch once
            try:
                await asyncio.sleep(2)
                return await self.llm.embed(texts)
            except Exception:
                logger.warning(f"Batch retry failed, falling back to individual embedding")
                return await self._embed_individually(texts)

    async def _embed_individually(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts one at a time with error handling.
        
        Args:
            texts: Texts to embed individually
            
        Returns:
            List of embedding vectors
        """
        results = []
        
        # Process with limited parallelism
        semaphore = asyncio.Semaphore(MAX_PARALLEL_EMBEDDINGS)
        
        async def embed_with_semaphore(text: str) -> list[float]:
            async with semaphore:
                return await self.embed_safe(text)
        
        tasks = [embed_with_semaphore(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        return list(results)

    async def embed_with_retry(
        self,
        text: str,
        max_retries: int = 3,
    ) -> list[float]:
        """
        Generate embedding with explicit retry logic.
        Deprecated: Use embed_safe() instead which has built-in retry.
        
        Args:
            text: Text to embed
            max_retries: Maximum retry attempts
            
        Returns:
            Embedding vector
        """
        return await self.embed_safe(text)

    async def health_check(self) -> dict:
        """
        Check embedding service health.
        
        Returns:
            Health status dictionary
        """
        try:
            embedding = await self.llm.embed_single("health check test")
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
