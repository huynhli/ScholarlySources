import json
from pathlib import Path

import torch

from src.embeddings import EmbeddingModel
from src.models import Paper


PAPERS_FILE = Path("data/papers.json")

TOP_K = 5


def load_papers(path: Path) -> list[Paper]:
    """
    Load papers from JSON.
    """

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return [
        Paper.from_dict(item)
        for item in data
    ]


def paper_to_text(paper: Paper) -> str:
    """
    Create the text representation that will be embedded.

    We combine the title and abstract because both contain
    useful information for semantic search.
    """

    return f"{paper.title}. {paper.abstract}"


def search(
    query: str,
    papers: list[Paper],
    model: EmbeddingModel,
    top_k: int = TOP_K,
) -> list[tuple[Paper, float]]:
    """
    Find the papers most semantically similar to the query.
    """

    paper_texts = [
        paper_to_text(paper)
        for paper in papers
    ]

    paper_embeddings = model.encode(paper_texts)

    query_embedding = model.encode([query])[0]

    similarities = torch.matmul(
        paper_embeddings,
        query_embedding,
    )

    top_results = torch.topk(
        similarities,
        k=min(top_k, len(papers)),
    )

    results: list[tuple[Paper, float]] = []

    for index, score in zip(
        top_results.indices,
        top_results.values,
    ):
        paper = papers[index.item()]
        similarity = score.item()

        results.append(
            (paper, similarity)
        )

    return results


def print_results(
    results: list[tuple[Paper, float]],
) -> None:
    """
    Display search results in a readable format.
    """

    if not results:
        print("No results found.")
        return

    print()
    print("Search results")
    print("=" * 60)

    for rank, (paper, score) in enumerate(results, start=1):
        print()
        print(f"{rank}. {paper.title}")
        print(f"   Similarity: {score:.4f}")
        print(f"   Authors: {', '.join(paper.authors)}")
        print(f"   Published: {paper.published}")
        print(f"   Source: {paper.source}")
        print(f"   URL: {paper.url}")


def main() -> None:
    """
    Run the interactive search program.
    """

    papers = load_papers(PAPERS_FILE)

    if not papers:
        print("No papers found.")
        return

    print(f"Loaded {len(papers)} papers.")

    model = EmbeddingModel()

    while True:
        print()

        query = input(
            "Enter a search query (or 'q' to quit): "
        ).strip()

        if query.lower() == "q":
            break

        if not query:
            continue

        results = search(
            query=query,
            papers=papers,
            model=model,
        )

        print_results(results)


if __name__ == "__main__":
    main()