"""
LLM-powered agent for intelligent link evaluation.
Decides which links to follow during crawling.
"""

import json
import logging
from dataclasses import dataclass

from doc_builder.crawler.parser import ExtractedLink
from doc_builder.llm import get_llm_client
from doc_builder.llm.prompts import (
    CRAWLER_AGENT_SYSTEM,
    build_batch_evaluation_prompt,
    build_link_evaluation_prompt,
)
from doc_builder.llm.schemas import BatchLinkResult, LinkEvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class LinkEvaluation:
    """Result of evaluating a single link."""

    url: str
    should_follow: bool
    priority: float = 0.5
    reason: str = ""


class CrawlerAgent:
    """
    Agent that uses LLM to intelligently decide which links to follow.
    """

    def __init__(
        self,
        root_url: str,
        doc_name: str,
        max_depth: int = 2,
        batch_mode: bool = True,
    ):
        """
        Initialize the crawler agent.
        
        Args:
            root_url: Root documentation URL
            doc_name: Name of the documentation
            max_depth: Maximum crawl depth
            batch_mode: Use faster batch evaluation (recommended)
        """
        self.root_url = root_url
        self.doc_name = doc_name
        self.max_depth = max_depth
        self.batch_mode = batch_mode
        self.llm = get_llm_client()

        # Cache for URL patterns to avoid repeated LLM calls
        self._pattern_cache: dict[str, bool] = {}

    async def evaluate_links(
        self,
        links: list[ExtractedLink],
        page_url: str,
        page_title: str,
        current_depth: int,
        pages_crawled: int,
    ) -> list[LinkEvaluation]:
        """
        Evaluate a list of links and decide which to follow.
        
        Args:
            links: Links to evaluate
            page_url: Current page URL
            page_title: Current page title
            current_depth: Current crawl depth
            pages_crawled: Number of pages crawled so far
            
        Returns:
            List of LinkEvaluation results
        """
        if not links:
            return []

        # Pre-filter obviously bad links
        filtered_links = self._prefilter_links(links)

        if not filtered_links:
            return []

        # Check cache first
        cached_results = []
        uncached_links = []

        for link in filtered_links:
            cached = self._check_cache(link.url)
            if cached is not None:
                cached_results.append(LinkEvaluation(
                    url=link.url,
                    should_follow=cached,
                    priority=0.5 if cached else 0.0,
                    reason="cached decision",
                ))
            else:
                uncached_links.append(link)

        # If all cached, return
        if not uncached_links:
            return cached_results

        # Evaluate uncached links with LLM
        try:
            if self.batch_mode:
                llm_results = await self._batch_evaluate(
                    uncached_links,
                    page_url,
                    current_depth,
                )
            else:
                llm_results = await self._detailed_evaluate(
                    uncached_links,
                    page_url,
                    page_title,
                    current_depth,
                    pages_crawled,
                )

            # Update cache
            for result in llm_results:
                self._update_cache(result.url, result.should_follow)

            return cached_results + llm_results

        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            # Fallback: use heuristics
            return cached_results + self._heuristic_evaluate(uncached_links)

    def _prefilter_links(self, links: list[ExtractedLink]) -> list[ExtractedLink]:
        """Pre-filter links using simple rules."""
        from urllib.parse import urlparse

        root_domain = urlparse(self.root_url).netloc
        filtered = []

        for link in links:
            parsed = urlparse(link.url)

            # Must be same domain (or subdomain)
            if not parsed.netloc.endswith(root_domain.split(".")[-2] + "." + root_domain.split(".")[-1]):
                # Allow exact domain match or subdomain
                if parsed.netloc != root_domain and not parsed.netloc.endswith("." + root_domain):
                    continue

            # Skip obvious non-doc patterns
            path_lower = parsed.path.lower()
            skip_patterns = [
                "/login", "/logout", "/signup", "/register",
                "/account", "/profile", "/settings",
                "/search", "/cart", "/checkout",
                "/api/", "/graphql",
                ".pdf", ".zip", ".tar", ".gz",
                ".png", ".jpg", ".jpeg", ".gif", ".svg",
                ".css", ".js", ".woff",
            ]
            if any(pattern in path_lower for pattern in skip_patterns):
                continue

            filtered.append(link)

        return filtered

    async def _batch_evaluate(
        self,
        links: list[ExtractedLink],
        page_url: str,
        current_depth: int,
    ) -> list[LinkEvaluation]:
        """Fast batch evaluation of links."""
        # Prepare links data
        links_data = [
            {"url": link.url, "anchor_text": link.anchor_text}
            for link in links
        ]

        prompt = build_batch_evaluation_prompt(
            page_url=page_url,
            root_url=self.root_url,
            doc_name=self.doc_name,
            current_depth=current_depth,
            max_depth=self.max_depth,
            links=links_data,
        )

        try:
            result = await self.llm.complete_structured(
                messages=[
                    {"role": "system", "content": CRAWLER_AGENT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_model=BatchLinkResult,
                model=self.llm.settings.fast_model,
                temperature=0.2,
            )

            # Convert to LinkEvaluation
            evaluations = []
            follow_set = set(result.follow)

            for link in links:
                should_follow = link.url in follow_set
                evaluations.append(LinkEvaluation(
                    url=link.url,
                    should_follow=should_follow,
                    priority=0.6 if should_follow else 0.0,
                    reason="batch evaluation",
                ))

            return evaluations

        except Exception as e:
            logger.warning(f"Batch evaluation failed: {e}, using heuristics")
            return self._heuristic_evaluate(links)

    async def _detailed_evaluate(
        self,
        links: list[ExtractedLink],
        page_url: str,
        page_title: str,
        current_depth: int,
        pages_crawled: int,
    ) -> list[LinkEvaluation]:
        """Detailed evaluation with priorities and reasons."""
        links_data = [
            {
                "url": link.url,
                "anchor_text": link.anchor_text,
                "title": link.title,
                "is_navigation": link.is_navigation,
                "context": link.context,
            }
            for link in links
        ]

        prompt = build_link_evaluation_prompt(
            page_url=page_url,
            page_title=page_title,
            root_url=self.root_url,
            doc_name=self.doc_name,
            current_depth=current_depth,
            max_depth=self.max_depth,
            pages_crawled=pages_crawled,
            links=links_data,
        )

        try:
            result = await self.llm.complete_structured(
                messages=[
                    {"role": "system", "content": CRAWLER_AGENT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                response_model=LinkEvaluationResult,
                temperature=0.3,
            )

            # Convert to LinkEvaluation
            url_to_decision = {d.url: d for d in result.decisions}
            evaluations = []

            for link in links:
                decision = url_to_decision.get(link.url)
                if decision:
                    evaluations.append(LinkEvaluation(
                        url=link.url,
                        should_follow=decision.action == "follow",
                        priority=decision.priority,
                        reason=decision.reason,
                    ))
                else:
                    # Default to not following if not in response
                    evaluations.append(LinkEvaluation(
                        url=link.url,
                        should_follow=False,
                        priority=0.0,
                        reason="not evaluated",
                    ))

            return evaluations

        except Exception as e:
            logger.warning(f"Detailed evaluation failed: {e}, using heuristics")
            return self._heuristic_evaluate(links)

    def _heuristic_evaluate(self, links: list[ExtractedLink]) -> list[LinkEvaluation]:
        """Fallback heuristic evaluation when LLM fails."""
        evaluations = []

        # Positive patterns
        doc_patterns = [
            "docs", "documentation", "guide", "tutorial",
            "api", "reference", "getting-started", "quickstart",
            "learn", "examples", "cookbook", "manual",
        ]

        for link in links:
            url_lower = link.url.lower()
            anchor_lower = link.anchor_text.lower()

            # Check for documentation patterns
            is_doc_like = any(
                pattern in url_lower or pattern in anchor_lower
                for pattern in doc_patterns
            )

            # Higher priority for non-navigation links with doc patterns
            priority = 0.7 if is_doc_like and not link.is_navigation else 0.4

            evaluations.append(LinkEvaluation(
                url=link.url,
                should_follow=is_doc_like or not link.is_navigation,
                priority=priority,
                reason="heuristic evaluation",
            ))

        return evaluations

    def _check_cache(self, url: str) -> bool | None:
        """Check if URL pattern is cached."""
        # Extract path pattern
        from urllib.parse import urlparse
        parsed = urlparse(url)
        
        # Create pattern by removing specific IDs/numbers
        import re
        pattern = re.sub(r"/\d+", "/#", parsed.path)
        pattern = re.sub(r"/[a-f0-9]{8,}", "/#", pattern)

        return self._pattern_cache.get(pattern)

    def _update_cache(self, url: str, should_follow: bool) -> None:
        """Update pattern cache."""
        from urllib.parse import urlparse
        import re

        parsed = urlparse(url)
        pattern = re.sub(r"/\d+", "/#", parsed.path)
        pattern = re.sub(r"/[a-f0-9]{8,}", "/#", pattern)

        self._pattern_cache[pattern] = should_follow


def create_crawler_agent(
    root_url: str,
    doc_name: str,
    max_depth: int = 2,
) -> CrawlerAgent:
    """
    Create a crawler agent.
    
    Args:
        root_url: Root documentation URL
        doc_name: Name of the documentation
        max_depth: Maximum crawl depth
        
    Returns:
        Configured CrawlerAgent
    """
    return CrawlerAgent(
        root_url=root_url,
        doc_name=doc_name,
        max_depth=max_depth,
    )
