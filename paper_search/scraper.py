import requests
import feedparser
import json


SEARCH_QUERY = "large language models"
MAX_RESULTS = 100

ARXIV_URL = "https://export.arxiv.org/api/query"


def fetch_papers():
    params = {
        "search_query": f"all:{SEARCH_QUERY}",
        "start": 0,
        "max_results": MAX_RESULTS,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    response = requests.get(ARXIV_URL, params=params)

    response.raise_for_status()

    feed = feedparser.parse(response.text)

    papers = []

    for entry in feed.entries:
        paper = {
            "title": entry.title.replace("\n", " ").strip(),
            "abstract": entry.summary.replace("\n", " ").strip(),
            "authors": [author.name for author in entry.authors],
            "published": entry.published,
            "url": entry.link,
        }

        papers.append(paper)

    return papers


def save_papers(papers):
    with open("papers.json", "w", encoding="utf-8") as file:
        json.dump(papers, file, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    papers = fetch_papers()

    print(f"Found {len(papers)} papers.")

    save_papers(papers)

    print("Saved papers to papers.json")