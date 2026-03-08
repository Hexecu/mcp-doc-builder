"""
HTML parser for extracting content and links from web pages.
Uses BeautifulSoup and trafilatura for robust content extraction.
"""

import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup, Tag

from doc_builder.utils import clean_text, is_valid_doc_url, normalize_url

logger = logging.getLogger(__name__)


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


class HTMLParser:
    """
    Parser for extracting structured content from HTML pages.
    """

    # Tags that typically contain navigation
    NAV_TAGS = {"nav", "header", "footer", "aside"}
    NAV_CLASSES = {"nav", "navigation", "menu", "sidebar", "footer", "header", "toc", "breadcrumb"}

    # Tags to skip when extracting content
    SKIP_TAGS = {"script", "style", "noscript", "iframe", "svg", "canvas"}

    def __init__(self, base_url: str | None = None):
        """
        Initialize parser.
        
        Args:
            base_url: Base URL for resolving relative links
        """
        self.base_url = base_url

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

        # Extract content
        content, content_html = self._extract_content(soup)
        headings = self._extract_headings(soup)
        word_count = len(content.split())

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
        )

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

    def _extract_content(self, soup: BeautifulSoup) -> tuple[str, str]:
        """
        Extract main content from the page.
        
        Returns:
            Tuple of (plain text content, HTML content)
        """
        # Try trafilatura for content extraction
        try:
            import trafilatura

            html_str = str(soup)
            extracted = trafilatura.extract(
                html_str,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                include_links=False,
                favor_precision=True,
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
            logger.debug(f"Trafilatura extraction failed, falling back: {e}")

        # Fallback: manual extraction
        return self._manual_content_extraction(soup)

    def _manual_content_extraction(self, soup: BeautifulSoup) -> tuple[str, str]:
        """Manual fallback for content extraction."""
        # Remove unwanted tags
        for tag in soup.find_all(self.SKIP_TAGS):
            tag.decompose()

        # Try to find main content area
        main_content = None
        for selector in ["main", "article", '[role="main"]', ".content", "#content", ".main"]:
            main_content = soup.select_one(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body or soup

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
