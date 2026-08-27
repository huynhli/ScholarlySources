import torch
from pathlib import Path

from src.embeddings import EmbeddingModel
from src.models import Paper


EMBEDDINGS_FILE = Path("data/paper_embeddings.pt")


def create_embeddings(papers: list[Paper], model: EmbeddingModel) -> torch.Tensor:
    """
    Generate embeddings for all papers.
    """

    texts = [f"{paper.title}. {paper.abstract}" for paper in papers]

    return model.encode(texts)


def save_embeddings(embeddings: torch.Tensor, path: Path = EMBEDDINGS_FILE) -> None:
    """
    Save paper embeddings to disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(embeddings.cpu(), path)

    print(f"Saved embeddings to {path}")


def load_embeddings(path: Path = EMBEDDINGS_FILE) -> torch.Tensor:
    """
    Load paper embeddings from disk.
    """

    if not path.exists():
        raise FileNotFoundError(f"Embedding file not found: {path}")
    

    return torch.load(path, weights_only=True)
