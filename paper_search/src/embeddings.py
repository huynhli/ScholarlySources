import torch
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:
    """
    Converts text into numerical embeddings.
    The model is loaded once and reused for all embedding operations.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading embedding model: {model_name}")
        print(f"Using device: {self.device}")

        self.model = SentenceTransformer(model_name, device = self.device,)

    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Convert a list of texts into normalized embedding vectors.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings