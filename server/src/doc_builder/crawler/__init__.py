"""
Web crawler module with intelligent link evaluation.
"""

from doc_builder.crawler.agent import CrawlerAgent, LinkEvaluation, create_crawler_agent
from doc_builder.crawler.parser import (
    ExtractedLink,
    ExtractedMetatag,
    HTMLParser,
    ParsedPage,
    parse_html,
)
from doc_builder.crawler.spider import (
    CrawlResult,
    CrawlStats,
    CrawlTask,
    Spider,
    crawl_documentation,
)

__all__ = [
    # Parser
    "HTMLParser",
    "ParsedPage",
    "ExtractedLink",
    "ExtractedMetatag",
    "parse_html",
    # Agent
    "CrawlerAgent",
    "LinkEvaluation",
    "create_crawler_agent",
    # Spider
    "Spider",
    "CrawlTask",
    "CrawlResult",
    "CrawlStats",
    "crawl_documentation",
]
