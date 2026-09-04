from bs4 import BeautifulSoup

from src.models import Paper


def parse_feed(content: str, source: str = "unknown") -> list[Paper]:
    """
    Parse an Atom/RSS feed and convert its entries into Paper objects.
    """

    soup = BeautifulSoup(content, "xml")

    papers: list[Paper] = []

    for entry in soup.find_all(["entry", "item"]):
        title = _get_text(entry, "title")
        abstract = _get_text(entry, "summary") or _get_text(entry, "description")
        published = (
            _get_text(entry, "published")
            or _get_text(entry, "pubDate")
            or _get_text(entry, "updated")
        )

        url = _extract_url(entry)
        authors = _extract_authors(entry)

        if not title:
            continue

        paper = Paper(
            title = title,
            abstract = abstract,
            authors = authors,
            url = url,
            published = published,
            source = source,
        )

        papers.append(paper)

    return papers


def _get_text(parent, tag_name: str) -> str:
    """Extract and clean the text contained inside a tag."""

    element = parent.find(tag_name)

    if element is None:
        return ""

    return element.get_text(" ", strip = True)


def _extract_url(entry) -> str:
    """Extract a paper URL from an Atom/RSS entry."""

    # atom feeds generally use <link href=""/> so we can find by link component
    link = entry.find("link")

    if link is not None:
        href = link.get("href")

        if href:
            return href.strip()

        text = link.get_text(strip = True)

        if text:
            return text

    # some RSS feeds use guid
    guid = entry.find("guid")

    if guid is not None:
        return guid.get_text(strip = True)

    return ""


def _extract_authors(entry) -> list[str]:
    """Extract author names from an Atom/RSS entry."""

    authors: list[str] = []

    # Atom commonly uses <author><name> tags
    for author in entry.find_all("author"):
        name = author.find("name")

        if name is not None:
            author_name = name.get_text(" ", strip = True)

            if author_name:
                authors.append(author_name)

    # some feeds use <creator>
    if not authors:
        creator = entry.find("creator")

        if creator is not None:
            creator_name = creator.get_text(" ", strip = True)

            if creator_name:
                authors.append(creator_name)

    return authors