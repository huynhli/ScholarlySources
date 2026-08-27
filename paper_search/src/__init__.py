import json

import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


def load_papers():
    with open("papers.json", "r", encoding="utf-8") as file:
        return json.load(file)


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


def search_papers(query, papers, model):
    abstracts = [paper["abstract"] for paper in papers]

    query_embedding = model.encode(query)
    paper_embeddings = model.encode(abstracts)

    results = []

    for paper, embedding in zip(papers, paper_embeddings):
        score = cosine_similarity(
            query_embedding,
            embedding
        )

        results.append({
            "paper": paper,
            "score": float(score)
        })

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results


def main():
    papers = load_papers()

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    query = input("\nSearch for papers: ")

    results = search_papers(
        query,
        papers,
        model
    )

    print("\nResults:\n")

    for result in results[:10]:
        paper = result["paper"]

        print(f"{result['score']:.3f}  {paper['title']}")
        print(f"      {paper['url']}")
        print()


if __name__ == "__main__":
    main()