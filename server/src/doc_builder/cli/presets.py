"""
Documentation presets for common frameworks and libraries.
"""

from dataclasses import dataclass


@dataclass
class DocPreset:
    """A documentation preset configuration."""

    key: str
    name: str
    url: str
    max_depth: int
    description: str
    estimated_pages: int  # Rough estimate for time calculation


# Common documentation presets
DOC_PRESETS: dict[str, DocPreset] = {
    "langchain": DocPreset(
        key="langchain",
        name="LangChain Python",
        url="https://python.langchain.com/docs/",
        max_depth=2,
        description="LangChain framework for LLM applications",
        estimated_pages=250,
    ),
    "llamaindex": DocPreset(
        key="llamaindex",
        name="LlamaIndex",
        url="https://docs.llamaindex.ai/en/stable/",
        max_depth=2,
        description="LlamaIndex data framework for LLM apps",
        estimated_pages=200,
    ),
    "openai": DocPreset(
        key="openai",
        name="OpenAI API",
        url="https://platform.openai.com/docs/",
        max_depth=2,
        description="OpenAI API reference and guides",
        estimated_pages=150,
    ),
    "nextjs": DocPreset(
        key="nextjs",
        name="Next.js",
        url="https://nextjs.org/docs",
        max_depth=2,
        description="Next.js React framework",
        estimated_pages=180,
    ),
    "react": DocPreset(
        key="react",
        name="React",
        url="https://react.dev/reference/react",
        max_depth=2,
        description="React library reference",
        estimated_pages=120,
    ),
    "python-tutorial": DocPreset(
        key="python-tutorial",
        name="Python Tutorial",
        url="https://docs.python.org/3/tutorial/",
        max_depth=1,
        description="Official Python tutorial",
        estimated_pages=50,
    ),
    "fastapi": DocPreset(
        key="fastapi",
        name="FastAPI",
        url="https://fastapi.tiangolo.com/",
        max_depth=2,
        description="FastAPI web framework",
        estimated_pages=100,
    ),
    "pydantic": DocPreset(
        key="pydantic",
        name="Pydantic",
        url="https://docs.pydantic.dev/latest/",
        max_depth=2,
        description="Pydantic data validation",
        estimated_pages=80,
    ),
    "anthropic": DocPreset(
        key="anthropic",
        name="Anthropic Claude API",
        url="https://docs.anthropic.com/",
        max_depth=2,
        description="Anthropic Claude API documentation",
        estimated_pages=80,
    ),
    "huggingface": DocPreset(
        key="huggingface",
        name="Hugging Face Transformers",
        url="https://huggingface.co/docs/transformers/",
        max_depth=2,
        description="Hugging Face Transformers library",
        estimated_pages=300,
    ),
}


def get_preset(key: str) -> DocPreset | None:
    """Get a preset by key."""
    return DOC_PRESETS.get(key.lower())


def get_all_presets() -> list[DocPreset]:
    """Get all available presets."""
    return list(DOC_PRESETS.values())


def get_preset_keys() -> list[str]:
    """Get all preset keys."""
    return list(DOC_PRESETS.keys())


def estimate_time_minutes(presets: list[DocPreset]) -> tuple[int, int]:
    """
    Estimate indexing time for a list of presets.
    
    Returns:
        Tuple of (min_minutes, max_minutes)
    """
    total_pages = sum(p.estimated_pages for p in presets)
    
    # Estimate: ~2-4 seconds per page (including LLM calls)
    min_minutes = max(1, total_pages * 2 // 60)
    max_minutes = max(2, total_pages * 4 // 60)
    
    return min_minutes, max_minutes
