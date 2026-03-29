"""
Preprocessing pipeline for Wikipedia XML dumps.

Workflow
--------
1. Extract plain text from a Wikipedia XML dump using *wikiextractor*.
2. Clean / normalise the extracted text.
3. Serialise each article as a :class:`~src.schema.WikiArticle` and write
   the results to a JSONL file.

Usage (CLI)
-----------
    python -m src.preprocess \\
        --input  data/raw/enwiki-latest-pages-articles.xml.bz2 \\
        --output data/processed/wiki.jsonl \\
        [--min-words 50]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator

from src.schema import WikiArticle, write_jsonl

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text cleaning helpers
# ---------------------------------------------------------------------------


_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
_SECTION_HEADER_RE = re.compile(r"^={2,}\s*.+?\s*={2,}$", re.MULTILINE)
_EDIT_LINK_RE = re.compile(r"\[edit\]")
_WHITESPACE_RE = re.compile(r"[ \t]+")


def clean_text(raw: str) -> str:
    """Apply heuristic cleaning to wikiextractor-produced plain text.

    Steps
    ~~~~~
    * Strip leading/trailing whitespace.
    * Remove ``[edit]`` anchors left by wikiextractor.
    * Collapse excessive blank lines (> 2 in a row → 1 blank line).
    * Normalise intra-line whitespace (tabs / multiple spaces → single space).
    """
    text = raw.strip()
    text = _EDIT_LINK_RE.sub("", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text


# ---------------------------------------------------------------------------
# wikiextractor integration
# ---------------------------------------------------------------------------


def run_wikiextractor(dump_path: Path, output_dir: Path) -> None:
    """Run wikiextractor on *dump_path* and write output to *output_dir*.

    Requires ``wikiextractor`` to be installed (see requirements.txt).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "wikiextractor",
        "--json",
        "--output",
        str(output_dir),
        "--quiet",
        str(dump_path),
    ]
    logger.info("Running wikiextractor: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"wikiextractor failed (exit {result.returncode}):\n{result.stderr}"
        )


def iter_wikiextractor_output(output_dir: Path) -> Iterator[dict]:
    """Yield raw JSON dicts produced by wikiextractor's ``--json`` flag."""
    for json_file in sorted(output_dir.rglob("wiki_*")):
        with json_file.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Skipping malformed line in %s", json_file)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def process_dump(
    input_path: Path,
    output_path: Path,
    min_words: int = 50,
) -> int:
    """Full pipeline: dump → wikiextractor → clean → JSONL.

    Parameters
    ----------
    input_path:
        Path to the raw Wikipedia dump (XML or bz2-compressed XML).
    output_path:
        Destination JSONL file.
    min_words:
        Articles with fewer words after cleaning are discarded.

    Returns
    -------
    int
        Number of articles written.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        extracted_dir = Path(tmp_dir) / "extracted"
        run_wikiextractor(input_path, extracted_dir)
        articles = _build_articles(extracted_dir, min_words)
        count = write_jsonl(articles, output_path)

    logger.info("Wrote %d articles to %s", count, output_path)
    return count


def _build_articles(
    extracted_dir: Path,
    min_words: int,
) -> Iterator[WikiArticle]:
    """Yield cleaned WikiArticle objects from wikiextractor output."""
    for raw in iter_wikiextractor_output(extracted_dir):
        text = clean_text(raw.get("text", ""))
        if len(text.split()) < min_words:
            continue
        yield WikiArticle(
            id=str(raw.get("id", "")),
            url=raw.get("url", ""),
            title=raw.get("title", ""),
            text=text,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Process a Wikipedia XML dump into a JSONL dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the Wikipedia dump file (e.g. enwiki-*.xml.bz2).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSONL file (e.g. data/processed/wiki.jsonl).",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=50,
        help="Minimum word count for an article to be included (default: 50).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    count = process_dump(
        input_path=args.input,
        output_path=args.output,
        min_words=args.min_words,
    )
    print(f"Done – wrote {count} articles to {args.output}")


if __name__ == "__main__":
    main()
