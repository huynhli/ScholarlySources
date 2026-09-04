import asyncio
import json
from pathlib import Path

import httpx

from src.models import Paper
from src.parser import parse_feed


OUTPUT_FILE = Path("data/papers.json")

# arXiv API endpoint.
# We will eventually make sources configurable.
ARXIV_URLS = [
    "https://export.arxiv.org/api/query?search_query=all:machine%20learning&start=0&max_results=10",
    "https://export.arxiv.org/api/query?search_query=all:computer%20vision&start=0&max_results=10",
    "https://export.arxiv.org/api/query?search_query=all:natural%20language%20processing&start=0&max_results=10",
]


async def fetch(client: httpx.AsyncClient, url: str) -> str:
    """
    Fetch a URL asynchronously and return its response body.
    """

    response = await client.get(url)

    response.raise_for_status()

    return response.text


async def crawl_source(client: httpx.AsyncClient, url: str) -> list[Paper]:
    """
    Fetch and parse one paper source.
    """

    try:
        print(f"Fetching: {url}")

        content = await fetch(client, url)

        papers = parse_feed(
            content,
            source="arxiv",
        )

        print(f"Found {len(papers)} papers")

        return papers

    except httpx.HTTPError as error:
        print(f"HTTP error while fetching {url}: {error}")
        return []

    except Exception as error:
        print(f"Error while processing {url}: {error}")
        return []


async def crawl(urls: list[str]) -> list[Paper]:
    """
    Crawl multiple URLs concurrently.
    """

    timeout = httpx.Timeout(30.0)

    headers = {"User-Agent": "paper-search/0.1"}

    async with httpx.AsyncClient(timeout = timeout, headers = headers, follow_redirects = True) as client:
        tasks = [crawl_source(client, url) for url in urls]
        results = await asyncio.gather(*tasks)

    papers: list[Paper] = []

    for result in results:
        papers.extend(result)

    return papers


def save_papers(papers: list[Paper], path: Path) -> None:
    """
    Save papers to a JSON file.
    """

    path.parent.mkdir(parents = True, exist_ok = True)

    data = [paper.to_dict() for paper in papers]

    with path.open("w", encoding = "utf-8") as file:
        json.dump(data, file, indent = 2, ensure_ascii = False)


async def main() -> None:
    """
    Run the crawler.
    """

    papers = await crawl(ARXIV_URLS)

    save_papers(papers, OUTPUT_FILE)

    print()
    print(f"Saved {len(papers)} papers to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())