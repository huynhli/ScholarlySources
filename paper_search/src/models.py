from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Paper:
    title: str
    abstract: str
    authors: list[str]
    url: str
    published: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Convert Paper object into a JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Paper":
        """Create Paper object from a dictionary."""
        return cls(
            title = data.get("title", ""),
            abstract = data.get("abstract", ""),
            authors = data.get("authors", []),
            url = data.get("url", ""),
            published = data.get("published", ""),
            source = data.get("source", ""),
        )