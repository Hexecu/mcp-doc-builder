"""
LLM client with support for chat completions and embeddings.
Supports both Gemini Direct and LiteLLM Gateway modes.
"""

import json
import logging
from typing import Any, TypeVar

import litellm
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from doc_builder.config import Settings, get_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """
    LLM client for chat completions and embeddings.
    
    Uses LiteLLM as the unified interface for multiple providers.
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._configure_litellm()

    def _configure_litellm(self) -> None:
        """Configure LiteLLM based on settings."""
        # Suppress verbose logging
        litellm.suppress_debug_info = True

        # Configure based on mode
        if self.settings.llm_mode == "gemini_direct":
            if self.settings.gemini_api_key:
                litellm.api_key = self.settings.gemini_api_key
        elif self.settings.llm_mode in ("litellm", "both"):
            if self.settings.litellm_api_key:
                litellm.api_key = self.settings.litellm_api_key

    def _get_completion_kwargs(self, model: str | None = None) -> dict[str, Any]:
        """Get kwargs for completion calls."""
        kwargs: dict[str, Any] = {}

        active_model = model or self.settings.active_model

        if self.settings.llm_mode in ("litellm", "both") and self.settings.litellm_base_url:
            kwargs["base_url"] = self.settings.litellm_base_url
            kwargs["api_key"] = self.settings.litellm_api_key
            kwargs["model"] = active_model
        elif self.settings.llm_mode == "gemini_direct":
            kwargs["api_key"] = self.settings.gemini_api_key
            # Ensure gemini/ prefix for direct mode
            if not active_model.startswith("gemini/"):
                kwargs["model"] = f"gemini/{active_model}"
            else:
                kwargs["model"] = active_model
        else:
            kwargs["model"] = active_model

        return kwargs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        """
        Generate a chat completion.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Optional model override
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            **kwargs: Additional arguments to pass to litellm
            
        Returns:
            Generated text content
        """
        completion_kwargs = self._get_completion_kwargs(model)

        try:
            response = await litellm.acompletion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **completion_kwargs,
                **kwargs,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Completion failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def complete_structured(
        self,
        messages: list[dict[str, str]],
        response_model: type[T],
        model: str | None = None,
        temperature: float = 0.3,
        **kwargs,
    ) -> T:
        """
        Generate a structured completion using JSON mode.
        
        Args:
            messages: List of message dicts
            response_model: Pydantic model for response parsing
            model: Optional model override
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Parsed Pydantic model instance
        """
        completion_kwargs = self._get_completion_kwargs(model)

        # Add JSON schema instruction to system message
        schema = response_model.model_json_schema()
        schema_instruction = (
            f"\n\nRespond with valid JSON matching this schema:\n"
            f"```json\n{json.dumps(schema, indent=2)}\n```"
        )

        # Modify messages to include schema
        modified_messages = messages.copy()
        if modified_messages and modified_messages[0]["role"] == "system":
            modified_messages[0] = {
                "role": "system",
                "content": modified_messages[0]["content"] + schema_instruction,
            }
        else:
            modified_messages.insert(0, {
                "role": "system",
                "content": f"You are a helpful assistant.{schema_instruction}",
            })

        try:
            response = await litellm.acompletion(
                messages=modified_messages,
                temperature=temperature,
                response_format={"type": "json_object"},
                **completion_kwargs,
                **kwargs,
            )

            content = response.choices[0].message.content or "{}"

            # Parse JSON response
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown code block
                import re
                match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
                if match:
                    data = json.loads(match.group(1))
                else:
                    raise

            return response_model.model_validate(data)

        except Exception as e:
            logger.error(f"Structured completion failed: {e}")
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def embed(
        self,
        text: str | list[str],
        model: str | None = None,
    ) -> list[list[float]]:
        """
        Generate embeddings for text.
        
        Args:
            text: Single text or list of texts to embed
            model: Optional model override (defaults to embedding_model)
            
        Returns:
            List of embedding vectors
        """
        embed_model = model or self.settings.embedding_model

        # Ensure proper model format for embedding
        if self.settings.llm_mode in ("litellm", "both") and self.settings.litellm_base_url:
            # Use LiteLLM gateway
            kwargs = {
                "model": embed_model,
                "api_base": self.settings.litellm_base_url,
                "api_key": self.settings.litellm_api_key,
            }
        else:
            # Direct Gemini embedding
            if not embed_model.startswith("text-embedding"):
                embed_model = f"text-embedding-004"
            kwargs = {
                "model": f"gemini/{embed_model}",
                "api_key": self.settings.gemini_api_key,
            }

        # Normalize input
        texts = [text] if isinstance(text, str) else text

        try:
            response = await litellm.aembedding(
                input=texts,
                **kwargs,
            )

            return [item["embedding"] for item in response.data]

        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    async def embed_single(self, text: str, model: str | None = None) -> list[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Optional model override
            
        Returns:
            Embedding vector
        """
        embeddings = await self.embed(text, model)
        return embeddings[0]

    async def health_check(self) -> dict[str, Any]:
        """
        Check LLM connectivity.
        
        Returns:
            Health status dictionary
        """
        try:
            # Test completion
            response = await self.complete(
                messages=[{"role": "user", "content": "Say 'ok'"}],
                max_tokens=10,
            )

            return {
                "status": "healthy",
                "model": self.settings.active_model,
                "mode": self.settings.llm_mode,
                "response": response[:50],
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "model": self.settings.active_model,
                "mode": self.settings.llm_mode,
                "error": str(e),
            }


# Global client instance
_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """Get the global LLM client instance."""
    global _client
    if _client is None:
        _client = LLMClient()
    return _client
