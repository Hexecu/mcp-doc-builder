"""
Async web spider with rate limiting and intelligent link following.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import AsyncIterator, Callable
from urllib.parse import urlparse

import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential

from doc_builder.config import Settings, get_settings
from doc_builder.crawler.agent import CrawlerAgent, LinkEvaluation, create_crawler_agent
from doc_builder.crawler.parser import HTMLParser, ParsedPage
from doc_builder.utils import extract_domain, generate_id, normalize_url

logger = logging.getLogger(__name__)


@dataclass
class CrawlTask:
    """A task in the crawl queue."""

    url: str
    depth: int
    priority: float = 0.5
    
    def __lt__(self, other: "CrawlTask") -> bool:
        """Enable comparison for PriorityQueue."""
        return self.priority > other.priority  # Higher priority first
    parent_url: str | None = None


@dataclass
class CrawlResult:
    """Result of crawling a single page."""

    url: str
    success: bool
    page: ParsedPage | None = None
    error: str | None = None
    depth: int = 0
    links_found: int = 0
    links_to_follow: int = 0


@dataclass
class CrawlStats:
    """Statistics for a crawl session."""

    pages_crawled: int = 0
    pages_failed: int = 0
    pages_skipped: int = 0
    total_links_found: int = 0
    total_links_followed: int = 0
    start_time: float = field(default_factory=time.time)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def pages_per_second(self) -> float:
        if self.elapsed_time > 0:
            return self.pages_crawled / self.elapsed_time
        return 0.0


class RateLimiter:
    """Rate limiter for per-domain request throttling."""

    def __init__(self, min_delay: float = 1.0):
        """
        Initialize rate limiter.
        
        Args:
            min_delay: Minimum seconds between requests to same domain
        """
        self.min_delay = min_delay
        self._last_request: dict[str, float] = defaultdict(float)
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def acquire(self, domain: str) -> None:
        """Wait until we can make a request to the domain."""
        async with self._locks[domain]:
            now = time.time()
            last = self._last_request[domain]
            wait_time = self.min_delay - (now - last)

            if wait_time > 0:
                await asyncio.sleep(wait_time)

            self._last_request[domain] = time.time()


class Spider:
    """
    Async web spider with intelligent link evaluation.
    """

    def __init__(
        self,
        root_url: str,
        doc_name: str,
        max_depth: int = 2,
        max_pages: int = 500,
        max_concurrent: int = 5,
        rate_limit: float = 1.0,
        timeout: int = 30,
        settings: Settings | None = None,
    ):
        """
        Initialize the spider.
        
        Args:
            root_url: Starting URL to crawl
            doc_name: Name of the documentation
            max_depth: Maximum link depth from root
            max_pages: Maximum pages to crawl
            max_concurrent: Maximum concurrent requests
            rate_limit: Seconds between requests per domain
            timeout: Request timeout in seconds
            settings: Optional settings override
        """
        self.root_url = normalize_url(root_url)
        self.doc_name = doc_name
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.max_concurrent = max_concurrent
        self.timeout = timeout

        self.settings = settings or get_settings()
        self.rate_limiter = RateLimiter(rate_limit)
        self.parser = HTMLParser(self.root_url)

        # Crawl state
        self._visited: set[str] = set()
        self._queue: asyncio.PriorityQueue[tuple[float, CrawlTask]] = asyncio.PriorityQueue()
        self._stats = CrawlStats()
        self._running = False

        # Agent for link evaluation
        self._agent: CrawlerAgent | None = None

        # Callbacks
        self._on_page_callback: Callable[[CrawlResult], None] | None = None

    @property
    def stats(self) -> CrawlStats:
        """Get current crawl statistics."""
        return self._stats

    def on_page(self, callback: Callable[[CrawlResult], None]) -> None:
        """
        Register callback for when a page is crawled.
        
        Args:
            callback: Function to call with CrawlResult
        """
        self._on_page_callback = callback

    async def crawl(self) -> None:
        """
        Start crawling. Results are delivered via on_page callback.
        """
        self._running = True
        self._stats = CrawlStats()
        self._visited.clear()

        # Initialize agent
        self._agent = create_crawler_agent(
            root_url=self.root_url,
            doc_name=self.doc_name,
            max_depth=self.max_depth,
        )

        # Add root URL to queue
        await self._queue.put((0.0, CrawlTask(url=self.root_url, depth=0, priority=1.0)))

        # Create session
        connector = aiohttp.TCPConnector(limit=self.max_concurrent, limit_per_host=2)
        timeout = aiohttp.ClientTimeout(total=self.timeout)

        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Create worker tasks
            workers = [
                asyncio.create_task(self._worker(session, i))
                for i in range(self.max_concurrent)
            ]

            try:
                # Wait for all tasks in queue to be processed
                await self._queue.join()
                
            except asyncio.CancelledError:
                logger.info("Crawl cancelled")
            finally:
                self._running = False
                # Cancel workers
                for worker in workers:
                    worker.cancel()
                # Wait for workers to finish
                await asyncio.gather(*workers, return_exceptions=True)

        logger.info(
            f"Crawl completed: {self._stats.pages_crawled} pages, "
            f"{self._stats.pages_failed} failed, "
            f"{self._stats.elapsed_time:.1f}s"
        )

    async def crawl_all(self) -> list[CrawlResult]:
        """
        Crawl all pages and return results as a list.
        
        Returns:
            List of all CrawlResults
        """
        results: list[CrawlResult] = []

        def collect_result(result: CrawlResult):
            results.append(result)

        self.on_page(collect_result)
        await self.crawl()

        return results

    async def _worker(self, session: aiohttp.ClientSession, worker_id: int) -> None:
        """Worker coroutine that processes queue items."""
        while self._running:
            try:
                # Get next task (with timeout to allow checking _running)
                try:
                    _, task = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Skip if already visited or over limit
                if task.url in self._visited:
                    self._queue.task_done()
                    continue

                if self._stats.pages_crawled >= self.max_pages:
                    self._queue.task_done()
                    break

                # Mark as visited
                self._visited.add(task.url)

                # Crawl the page
                result = await self._crawl_page(session, task)

                # Update stats
                if result.success:
                    self._stats.pages_crawled += 1
                    self._stats.total_links_found += result.links_found
                    self._stats.total_links_followed += result.links_to_follow
                else:
                    self._stats.pages_failed += 1

                # Trigger callback
                if self._on_page_callback:
                    self._on_page_callback(result)

                # Queue new links if we have depth remaining
                if result.success and result.page and task.depth < self.max_depth:
                    await self._queue_links(result.page, task)

                self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=1, max=5),
    )
    async def _fetch_page(self, session: aiohttp.ClientSession, url: str) -> str:
        """Fetch a page with retry logic."""
        domain = extract_domain(url)
        await self.rate_limiter.acquire(domain)

        headers = {
            "User-Agent": self.settings.crawler_user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
        }

        async with session.get(url, headers=headers, allow_redirects=True) as response:
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                raise ValueError(f"Not an HTML page: {content_type}")

            return await response.text()

    async def _crawl_page(
        self,
        session: aiohttp.ClientSession,
        task: CrawlTask,
    ) -> CrawlResult:
        """Crawl a single page."""
        try:
            logger.debug(f"Crawling: {task.url} (depth={task.depth})")

            # Fetch HTML
            html = await self._fetch_page(session, task.url)

            # Parse content
            page = self.parser.parse(html, task.url)

            return CrawlResult(
                url=task.url,
                success=True,
                page=page,
                depth=task.depth,
                links_found=len(page.links),
            )

        except Exception as e:
            logger.warning(f"Failed to crawl {task.url}: {e}")
            return CrawlResult(
                url=task.url,
                success=False,
                error=str(e),
                depth=task.depth,
            )

    async def _queue_links(self, page: ParsedPage, parent_task: CrawlTask) -> None:
        """Evaluate and queue links from a page."""
        if not page.links or not self._agent:
            return

        # Filter already visited
        new_links = [
            link for link in page.links
            if link.url not in self._visited
        ]

        if not new_links:
            return

        # Evaluate links with agent
        evaluations = await self._agent.evaluate_links(
            links=new_links,
            page_url=page.url,
            page_title=page.title,
            current_depth=parent_task.depth,
            pages_crawled=self._stats.pages_crawled,
        )

        # Queue links that should be followed
        links_to_follow = 0
        for eval in evaluations:
            if eval.should_follow and eval.url not in self._visited:
                priority = 1.0 - eval.priority  # Lower priority value = higher priority
                await self._queue.put((
                    priority,
                    CrawlTask(
                        url=eval.url,
                        depth=parent_task.depth + 1,
                        priority=eval.priority,
                        parent_url=page.url,
                    ),
                ))
                links_to_follow += 1

        logger.debug(
            f"Page {page.url}: {len(new_links)} new links, "
            f"{links_to_follow} to follow"
        )

    def stop(self) -> None:
        """Stop the crawl."""
        self._running = False


async def crawl_documentation(
    root_url: str,
    doc_name: str,
    max_depth: int = 2,
    max_pages: int = 500,
    on_page: Callable[[CrawlResult], None] | None = None,
) -> CrawlStats:
    """
    Convenience function to crawl a documentation site.
    
    Args:
        root_url: Starting URL
        doc_name: Documentation name
        max_depth: Maximum depth
        max_pages: Maximum pages
        on_page: Callback for each page
        
    Returns:
        Final CrawlStats
    """
    settings = get_settings()

    spider = Spider(
        root_url=root_url,
        doc_name=doc_name,
        max_depth=max_depth,
        max_pages=max_pages,
        max_concurrent=settings.crawler_max_concurrent,
        rate_limit=settings.crawler_rate_limit,
        timeout=settings.crawler_timeout,
    )

    if on_page:
        spider.on_page(on_page)

    await spider.crawl()

    return spider.stats
