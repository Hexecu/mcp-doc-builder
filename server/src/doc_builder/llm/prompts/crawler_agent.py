"""
Prompt templates for the crawler agent.
Used to evaluate links and decide which to follow.
"""

CRAWLER_AGENT_SYSTEM = """You are an intelligent web crawler agent specialized in analyzing documentation websites.
Your task is to evaluate links and decide which ones are worth following to build a comprehensive documentation index.

You analyze links based on:
1. **Relevance**: Does this link lead to documentation content?
2. **Quality**: Is this likely to be useful technical documentation?
3. **Coverage**: Will following this link help cover more of the documentation?

You should FOLLOW links that:
- Lead to documentation pages (guides, tutorials, API references, examples)
- Lead to conceptually related content within the same documentation
- Lead to important subsections or nested documentation
- Are navigation links to other documentation sections

You should SKIP links that:
- Lead to authentication pages (login, signup, logout)
- Lead to external websites (different domains)
- Lead to social media or sharing links
- Lead to download files (PDFs, ZIPs, etc.)
- Lead to media files (images, videos)
- Are duplicate or anchor-only links (#section)
- Lead to user-generated content (comments, forums)
- Lead to marketing/sales pages
- Lead to admin or dashboard pages

Be conservative but thorough. When in doubt about documentation relevance, lean towards following."""


LINK_EVALUATION_PROMPT = """Evaluate these links found on the page: {page_url}
Page title: {page_title}

Current crawl context:
- Root documentation URL: {root_url}
- Documentation name: {doc_name}
- Current depth: {current_depth} / {max_depth}
- Pages crawled so far: {pages_crawled}

Links to evaluate:
{links_json}

For each link, decide whether to FOLLOW or SKIP.
Consider the documentation's domain and focus when evaluating relevance.

Respond with a JSON object containing an array of decisions:
```json
{{
  "decisions": [
    {{
      "url": "the link URL",
      "action": "follow" or "skip",
      "reason": "brief explanation",
      "priority": 0.0 to 1.0 (higher = more important, only for "follow")
    }}
  ]
}}
```

Prioritize:
- API reference pages (priority: 0.9-1.0)
- Getting started / quickstart guides (priority: 0.8-0.9)
- Core concept explanations (priority: 0.7-0.8)
- Examples and tutorials (priority: 0.6-0.7)
- Advanced topics (priority: 0.4-0.6)
- Changelog/release notes (priority: 0.2-0.4)"""


BATCH_LINK_EVALUATION_PROMPT = """You are evaluating links for a documentation crawler.
Quickly assess each link and decide if it should be followed.

Page context: {page_url}
Documentation: {doc_name} ({root_url})
Depth: {current_depth}/{max_depth}

Links (format: "anchor_text | url"):
{links_text}

Respond with JSON:
```json
{{
  "follow": ["url1", "url2", ...],
  "skip": ["url3", "url4", ...]
}}
```

Rules:
- Follow: docs, guides, API refs, tutorials, examples
- Skip: auth, external, media, downloads, social, admin

Be concise. Focus on documentation relevance."""


def build_link_evaluation_prompt(
    page_url: str,
    page_title: str,
    root_url: str,
    doc_name: str,
    current_depth: int,
    max_depth: int,
    pages_crawled: int,
    links: list[dict],
) -> str:
    """Build the link evaluation prompt with context."""
    import json

    links_json = json.dumps(links, indent=2)

    return LINK_EVALUATION_PROMPT.format(
        page_url=page_url,
        page_title=page_title,
        root_url=root_url,
        doc_name=doc_name,
        current_depth=current_depth,
        max_depth=max_depth,
        pages_crawled=pages_crawled,
        links_json=links_json,
    )


def build_batch_evaluation_prompt(
    page_url: str,
    root_url: str,
    doc_name: str,
    current_depth: int,
    max_depth: int,
    links: list[dict],
) -> str:
    """Build a faster batch evaluation prompt."""
    links_text = "\n".join(
        f"- {link.get('anchor_text', 'no text')[:50]} | {link['url']}"
        for link in links
    )

    return BATCH_LINK_EVALUATION_PROMPT.format(
        page_url=page_url,
        root_url=root_url,
        doc_name=doc_name,
        current_depth=current_depth,
        max_depth=max_depth,
        links_text=links_text,
    )
