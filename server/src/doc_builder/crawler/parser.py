"""
HTML parser for extracting content and links from web pages.
Uses BeautifulSoup and trafilatura for robust content extraction.
Includes Playwright fallback for JavaScript-heavy pages.

Features:
- Trafilatura as primary extractor
- Aggressive manual fallback for problematic pages
- Playwright for JavaScript-rendered content
- Retry logic for failed extractions
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from doc_builder.utils import clean_text, is_valid_doc_url, normalize_url

logger = logging.getLogger(__name__)

# Minimum content length to consider extraction successful
MIN_CONTENT_LENGTH = 100


@dataclass
class ExtractedLink:
    """A link extracted from a page."""

    url: str
    anchor_text: str = ""
    title: str = ""
    is_navigation: bool = False
    context: str = ""  # Surrounding text for context


@dataclass
class ExtractedMetatag:
    """A metatag extracted from a page."""

    key: str
    value: str


@dataclass
class ParsedPage:
    """Result of parsing an HTML page."""

    url: str
    title: str = ""
    description: str = ""
    content: str = ""
    content_html: str = ""
    links: list[ExtractedLink] = field(default_factory=list)
    metatags: list[ExtractedMetatag] = field(default_factory=list)
    headings: list[dict] = field(default_factory=list)
    language: str = "en"
    word_count: int = 0
    extraction_method: str = "trafilatura"  # Track how content was extracted


class HTMLParser:
    """
    Parser for extracting structured content from HTML pages.
    
    Extraction priority:
    1. Trafilatura (best for articles/docs)
    2. Manual extraction (fallback)
    3. Aggressive extraction (get all text)
    4. Playwright (JavaScript rendering) - async only
    """

    # Tags that typically contain navigation
    NAV_TAGS = {"nav", "header", "footer", "aside"}
    NAV_CLASSES = {"nav", "navigation", "menu", "sidebar", "footer", "header", "toc", "breadcrumb"}

    # Tags to skip when extracting content
    SKIP_TAGS = {"script", "style", "noscript", "iframe", "svg", "canvas"}

    # Tags that typically contain documentation content
    CONTENT_TAGS = ["p", "li", "td", "th", "pre", "code", "blockquote", "dd", "dt"]

    def __init__(self, base_url: str | None = None):
        """
        Initialize parser.
        
        Args:
            base_url: Base URL for resolving relative links
        """
        self.base_url = base_url
        self._playwright_available: bool | None = None
        self._browser = None

    def parse(self, html: str, url: str) -> ParsedPage:
        """
        Parse HTML content and extract structured data.
        
        Args:
            html: Raw HTML string
            url: Page URL (for resolving relative links)
            
        Returns:
            ParsedPage with extracted content
        """
        soup = BeautifulSoup(html, "lxml")

        # Extract metadata
        title = self._extract_title(soup)
        description = self._extract_description(soup)
        metatags = self._extract_metatags(soup)
        language = self._extract_language(soup)

        # Extract content with multiple fallbacks
        content, content_html, method = self._extract_content_with_fallbacks(soup)
        headings = self._extract_headings(soup)
        word_count = len(content.split()) if content else 0

        # Extract links
        links = self._extract_links(soup, url)

        return ParsedPage(
            url=url,
            title=title,
            description=description,
            content=content,
            content_html=content_html,
            links=links,
            metatags=metatags,
            headings=headings,
            language=language,
            word_count=word_count,
            extraction_method=method,
        )

    async def parse_async(self, html: str, url: str, use_playwright: bool = True) -> ParsedPage:
        """
        Parse HTML with async support for Playwright fallback.
        
        Args:
            html: Raw HTML string
            url: Page URL
            use_playwright: Whether to try Playwright for JS pages
            
        Returns:
            ParsedPage with extracted content
        """
        # First try synchronous parsing
        result = self.parse(html, url)
        
        # If content is too short and playwright is available, try JS rendering
        if result.word_count < MIN_CONTENT_LENGTH // 4 and use_playwright:
            logger.debug(f"Content too short ({result.word_count} words), trying Playwright")
            playwright_content = await self._playwright_extraction(url)
            
            if playwright_content and len(playwright_content) > len(result.content):
                # Re-parse with JavaScript-rendered content
                soup = BeautifulSoup(playwright_content, "lxml")
                content, content_html, _ = self._extract_content_with_fallbacks(soup)
                
                if len(content) > len(result.content):
                    result.content = content
                    result.content_html = content_html
                    result.word_count = len(content.split())
                    result.extraction_method = "playwright"
                    
                    # Re-extract links from rendered page
                    result.links = self._extract_links(soup, url)
        
        return result

    def _extract_title(self, soup: BeautifulSoup) -> str:
        """Extract page title."""
        # Try <title> tag
        title_tag = soup.find("title")
        if title_tag:
            return clean_text(title_tag.get_text())

        # Try <h1>
        h1 = soup.find("h1")
        if h1:
            return clean_text(h1.get_text())

        # Try og:title
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            return clean_text(og_title["content"])

        return ""

    def _extract_description(self, soup: BeautifulSoup) -> str:
        """Extract page description."""
        # Try meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            return clean_text(meta_desc["content"])

        # Try og:description
        og_desc = soup.find("meta", property="og:description")
        if og_desc and og_desc.get("content"):
            return clean_text(og_desc["content"])

        return ""

    def _extract_metatags(self, soup: BeautifulSoup) -> list[ExtractedMetatag]:
        """Extract all relevant metatags."""
        metatags = []

        for meta in soup.find_all("meta"):
            key = None
            value = meta.get("content", "")

            if not value:
                continue

            # Standard meta tags
            if meta.get("name"):
                key = meta["name"]
            # Open Graph tags
            elif meta.get("property"):
                key = meta["property"]
            # Twitter cards
            elif meta.get("name", "").startswith("twitter:"):
                key = meta["name"]

            if key and value:
                metatags.append(ExtractedMetatag(key=key, value=clean_text(value)))

        return metatags

    def _extract_language(self, soup: BeautifulSoup) -> str:
        """Extract page language."""
        # Try html lang attribute
        html_tag = soup.find("html")
        if html_tag and html_tag.get("lang"):
            return html_tag["lang"][:2].lower()

        # Try meta language
        meta_lang = soup.find("meta", attrs={"http-equiv": "content-language"})
        if meta_lang and meta_lang.get("content"):
            return meta_lang["content"][:2].lower()

        return "en"

    def _extract_content_with_fallbacks(self, soup: BeautifulSoup) -> tuple[str, str, str]:
        """
        Extract content with multiple fallback strategies.
        
        Returns:
            Tuple of (plain text, HTML, extraction method)
        """
        # Method 1: Trafilatura
        content, content_html = self._trafilatura_extraction(soup)
        if content and len(content) >= MIN_CONTENT_LENGTH:
            return content, content_html, "trafilatura"
        
        # Method 2: Manual extraction (targeted areas)
        manual_content, manual_html = self._manual_content_extraction(soup)
        if manual_content and len(manual_content) >= MIN_CONTENT_LENGTH:
            return manual_content, manual_html, "manual"
        
        # Method 3: Aggressive extraction (all text tags)
        aggressive_content = self._aggressive_extraction(soup)
        if aggressive_content and len(aggressive_content) >= MIN_CONTENT_LENGTH // 2:
            return aggressive_content, "", "aggressive"
        
        # Method 4: Last resort - get ALL text
        all_text = self._extract_all_text(soup)
        if all_text:
            return all_text, "", "all_text"
        
        # Return best available (even if short)
        if content:
            return content, content_html, "trafilatura_partial"
        if manual_content:
            return manual_content, manual_html, "manual_partial"
        if aggressive_content:
            return aggressive_content, "", "aggressive_partial"
        
        return "", "", "none"

    def _trafilatura_extraction(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Try trafilatura for content extraction."""
        try:
            import trafilatura

            html_str = str(soup)
            extracted = trafilatura.extract(
                html_str,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                include_links=False,
                favor_precision=False,  # Changed to favor recall
                favor_recall=True,
            )
            if extracted:
                # Also get HTML version
                extracted_html = trafilatura.extract(
                    html_str,
                    include_comments=False,
                    include_tables=True,
                    output_format="html",
                )
                return clean_text(extracted), extracted_html or ""
        except Exception as e:
            logger.debug(f"Trafilatura extraction failed: {e}")

        return "", ""

    def _manual_content_extraction(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Manual fallback for content extraction."""
        # Make a copy to avoid modifying original
        soup_copy = BeautifulSoup(str(soup), "lxml")
        
        # Remove unwanted tags
        for tag in soup_copy.find_all(self.SKIP_TAGS):
            tag.decompose()

        # Try to find main content area
        main_content = None
        for selector in [
            "main", "article", '[role="main"]', 
            ".content", "#content", ".main",
            ".documentation", ".docs-content", ".markdown-body",
            ".post-content", ".entry-content", ".prose",
            "[class*='content']", "[class*='article']", "[class*='doc']",
        ]:
            try:
                main_content = soup_copy.select_one(selector)
                if main_content:
                    break
            except Exception:
                continue

        if not main_content:
            main_content = soup_copy.body or soup_copy

        # Remove navigation elements
        for nav in main_content.find_all(self.NAV_TAGS):
            nav.decompose()

        # Remove elements with navigation classes
        for element in main_content.find_all(class_=lambda x: x and any(
            nav_class in str(x).lower() for nav_class in self.NAV_CLASSES
        )):
            if isinstance(element, Tag):
                element.decompose()

        text = clean_text(main_content.get_text(separator="\n"))
        html = str(main_content)

        return text, html

    def _aggressive_extraction(self, soup: BeautifulSoup) -> str:
        """
        Aggressive extraction - get text from all content tags.
        Used when other methods fail.
        """
        text_parts = []
        
        # Make a copy to avoid modifying original
        soup_copy = BeautifulSoup(str(soup), "lxml")
        
        # Remove script and style
        for tag in soup_copy.find_all(["script", "style", "noscript"]):
            tag.decompose()
        
        # Get text from content-bearing tags
        for tag in soup_copy.find_all(self.CONTENT_TAGS):
            text = tag.get_text(strip=True)
            # Filter out very short or navigation-like text
            if text and len(text) > 20 and not self._looks_like_navigation(text):
                text_parts.append(text)
        
        # Also get heading text
        for level in range(1, 7):
            for heading in soup_copy.find_all(f"h{level}"):
                text = heading.get_text(strip=True)
                if text and len(text) > 5:
                    text_parts.append(text)
        
        return "\n\n".join(text_parts)

    def _extract_all_text(self, soup: BeautifulSoup) -> str:
        """Last resort - extract all visible text."""
        # Make a copy
        soup_copy = BeautifulSoup(str(soup), "lxml")
        
        # Remove script and style
        for tag in soup_copy.find_all(["script", "style", "noscript", "head"]):
            tag.decompose()
        
        # Get all text
        text = soup_copy.get_text(separator="\n", strip=True)
        
        # Clean up multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return clean_text(text)

    def _looks_like_navigation(self, text: str) -> bool:
        """Check if text looks like navigation."""
        text_lower = text.lower()
        nav_indicators = [
            "skip to", "jump to", "go to", "back to",
            "previous", "next", "home", "menu",
            "sign in", "sign up", "log in", "log out",
            "copyright", "all rights reserved",
        ]
        return any(indicator in text_lower for indicator in nav_indicators)

    async def _playwright_extraction(self, url: str) -> str:
        """
        Extract content using Playwright for JavaScript-rendered pages.
        
        Args:
            url: URL to fetch and render
            
        Returns:
            Rendered HTML or empty string
        """
        if not self._check_playwright_available():
            return ""
        
        try:
            from playwright.async_api import async_playwright
            
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=True,
                    args=["--no-sandbox", "--disable-dev-shm-usage"]
                )
                
                try:
                    context = await browser.new_context(
                        user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                    )
                    page = await context.new_page()
                    
                    # Navigate with timeout
                    await page.goto(url, wait_until="networkidle", timeout=30000)
                    
                    # Wait a bit for any lazy-loaded content
                    await asyncio.sleep(1)
                    
                    # Get rendered HTML
                    content = await page.content()
                    
                    await context.close()
                    return content
                    
                finally:
                    await browser.close()
                    
        except Exception as e:
            logger.debug(f"Playwright extraction failed for {url}: {e}")
            return ""

    def _check_playwright_available(self) -> bool:
        """Check if Playwright is available."""
        if self._playwright_available is not None:
            return self._playwright_available
        
        try:
            import playwright
            self._playwright_available = True
        except ImportError:
            self._playwright_available = False
            logger.debug("Playwright not installed. Install with: pip install playwright && playwright install chromium")
        
        return self._playwright_available

    def _extract_headings(self, soup: BeautifulSoup) -> list[dict]:
        """Extract heading hierarchy."""
        headings = []

        for level in range(1, 7):
            for heading in soup.find_all(f"h{level}"):
                text = clean_text(heading.get_text())
                if text:
                    headings.append({
                        "level": level,
                        "text": text,
                        "id": heading.get("id", ""),
                    })

        return headings

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> list[ExtractedLink]:
        """Extract all links from the page."""
        links = []
        seen_urls = set()

        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]

            # Skip empty, javascript, and mailto links
            if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                continue

            # Normalize URL
            try:
                full_url = normalize_url(href, base_url)
            except Exception:
                continue

            # Skip if already seen
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            # Skip non-doc URLs
            if not is_valid_doc_url(full_url):
                continue

            # Get anchor text and context
            anchor_text = clean_text(anchor.get_text())
            title = anchor.get("title", "")

            # Check if this is in a navigation area
            is_navigation = self._is_in_navigation(anchor)

            # Get surrounding context
            context = self._get_link_context(anchor)

            links.append(ExtractedLink(
                url=full_url,
                anchor_text=anchor_text[:200],  # Limit length
                title=title[:200] if title else "",
                is_navigation=is_navigation,
                context=context[:300] if context else "",
            ))

        return links

    def _is_in_navigation(self, element: Tag) -> bool:
        """Check if element is inside a navigation area."""
        for parent in element.parents:
            if parent.name in self.NAV_TAGS:
                return True
            parent_classes = parent.get("class", [])
            if isinstance(parent_classes, list):
                class_str = " ".join(parent_classes).lower()
            else:
                class_str = str(parent_classes).lower()
            if any(nav_class in class_str for nav_class in self.NAV_CLASSES):
                return True
        return False

    def _get_link_context(self, anchor: Tag, max_length: int = 200) -> str:
        """Get text surrounding a link for context."""
        parent = anchor.parent
        if parent:
            text = clean_text(parent.get_text())
            if len(text) <= max_length:
                return text
            # Try to find the anchor text position and extract context
            anchor_text = anchor.get_text()
            pos = text.find(anchor_text)
            if pos >= 0:
                start = max(0, pos - 50)
                end = min(len(text), pos + len(anchor_text) + 50)
                return text[start:end]
        return ""


def parse_html(html: str, url: str) -> ParsedPage:
    """
    Convenience function to parse HTML.
    
    Args:
        html: Raw HTML string
        url: Page URL
        
    Returns:
        ParsedPage with extracted content
    """
    parser = HTMLParser()
    return parser.parse(html, url)


async def parse_html_async(html: str, url: str, use_playwright: bool = True) -> ParsedPage:
    """
    Async convenience function to parse HTML with Playwright fallback.
    
    Args:
        html: Raw HTML string
        url: Page URL
        use_playwright: Whether to try Playwright for JS pages
        
    Returns:
        ParsedPage with extracted content
    """
    parser = HTMLParser()
    return await parser.parse_async(html, url, use_playwright)
