"""
Schema for JSONL output records produced by the preprocessing pipeline.
Each Wikipedia article is serialised as one JSON object per line.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class WikiArticle:
    """A single cleaned Wikipedia article ready for model training."""

    id: str
    url: str
    title: str
    text: str
    categories: list[str] = field(default_factory=list)
    # Optional metadata – populated when available
    word_count: Optional[int] = None
    char_count: Optional[int] = None

    def __post_init__(self) -> None:
        if self.word_count is None:
            self.word_count = len(self.text.split())
        if self.char_count is None:
            self.char_count = len(self.text)

    def to_jsonl(self) -> str:
        """Return a single-line JSON string (no trailing newline)."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "WikiArticle":
        """Reconstruct a WikiArticle from a parsed JSON dict."""
        return cls(
            id=data["id"],
            url=data["url"],
            title=data["title"],
            text=data["text"],
            categories=data.get("categories", []),
        )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def write_jsonl(articles: Iterator[WikiArticle], output_path: str | Path) -> int:
    """Write an iterable of WikiArticle objects to a JSONL file.

    Returns the number of records written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for article in articles:
            fh.write(article.to_jsonl() + "\n")
            count += 1
    return count


def read_jsonl(input_path: str | Path) -> Iterator[WikiArticle]:
    """Yield WikiArticle objects from a JSONL file."""
    input_path = Path(input_path)
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield WikiArticle.from_dict(json.loads(line))
