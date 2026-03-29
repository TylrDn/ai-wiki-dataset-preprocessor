"""
Sample data loader for the preprocessed Wikipedia JSONL dataset.

This module provides a HuggingFace-compatible ``load_dataset`` wrapper as well
as a lightweight pure-Python loader so users without the ``datasets`` library
can still iterate over the data.

Example
-------
    # HuggingFace datasets (recommended)
    from src.data_loader import load_wiki_dataset
    ds = load_wiki_dataset("data/processed/wiki.jsonl")
    print(ds[0])

    # Lightweight fallback
    from src.data_loader import iter_articles
    for article in iter_articles("data/processed/wiki.jsonl"):
        print(article["title"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


# ---------------------------------------------------------------------------
# Lightweight loader (no external dependencies)
# ---------------------------------------------------------------------------


def iter_articles(jsonl_path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield article dicts from a JSONL file one by one.

    This is a zero-dependency alternative to :func:`load_wiki_dataset`.

    Parameters
    ----------
    jsonl_path:
        Path to the JSONL file produced by the preprocessing pipeline.

    Yields
    ------
    dict
        One article per iteration with at least the keys
        ``id``, ``url``, ``title``, and ``text``.
    """
    path = Path(jsonl_path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ---------------------------------------------------------------------------
# HuggingFace datasets loader
# ---------------------------------------------------------------------------


def load_wiki_dataset(jsonl_path: str | Path, split: str = "train"):
    """Load the processed Wikipedia JSONL as a HuggingFace ``Dataset``.

    Requires the ``datasets`` package (``pip install datasets``).

    Parameters
    ----------
    jsonl_path:
        Path to the JSONL file produced by the preprocessing pipeline.
    split:
        Dataset split label (default ``"train"``).

    Returns
    -------
    datasets.Dataset
        A HuggingFace Dataset with columns matching :class:`~src.schema.WikiArticle`.
    """
    try:
        from datasets import Dataset  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for load_wiki_dataset. "
            "Install it with: pip install datasets"
        ) from exc

    records = list(iter_articles(jsonl_path))
    return Dataset.from_list(records)


# ---------------------------------------------------------------------------
# Quick self-test / demo
# ---------------------------------------------------------------------------


def _demo(jsonl_path: str) -> None:
    """Print a summary of the first few articles in a JSONL file."""
    print(f"Loading articles from: {jsonl_path}\n")
    for i, article in enumerate(iter_articles(jsonl_path)):
        print(f"[{i}] {article['title']} — {article.get('word_count', '?')} words")
        if i >= 4:
            print("  ...")
            break


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.data_loader <path/to/wiki.jsonl>")
        sys.exit(1)
    _demo(sys.argv[1])
