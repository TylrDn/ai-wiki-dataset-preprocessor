"""
Tests for the preprocessing pipeline.

These tests are intentionally self-contained and do **not** require a real
Wikipedia dump.  They exercise:

* Text cleaning helpers (src/preprocess.py)
* WikiArticle schema serialisation / round-trip (src/schema.py)
* Data loader iteration (src/data_loader.py)
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.schema import WikiArticle, read_jsonl, write_jsonl
from src.preprocess import clean_text, _build_articles, iter_wikiextractor_output
from src.data_loader import iter_articles


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------


def test_clean_text_strips_whitespace():
    assert clean_text("  hello world  ") == "hello world"


def test_clean_text_removes_edit_links():
    text = "Some section[edit]\nMore text"
    result = clean_text(text)
    assert "[edit]" not in result
    assert "Some section" in result


def test_clean_text_collapses_blank_lines():
    text = "Para one\n\n\n\nPara two"
    result = clean_text(text)
    assert "\n\n\n" not in result


def test_clean_text_collapses_tabs():
    text = "word1\t\tword2"
    result = clean_text(text)
    assert "\t" not in result


# ---------------------------------------------------------------------------
# WikiArticle schema
# ---------------------------------------------------------------------------


def test_wiki_article_word_count():
    article = WikiArticle(id="1", url="http://example.com", title="Test", text="one two three")
    assert article.word_count == 3


def test_wiki_article_char_count():
    article = WikiArticle(id="1", url="http://example.com", title="Test", text="hello")
    assert article.char_count == 5


def test_wiki_article_to_jsonl_is_valid_json():
    article = WikiArticle(id="42", url="http://x.com", title="A", text="Some text here.")
    line = article.to_jsonl()
    parsed = json.loads(line)
    assert parsed["id"] == "42"
    assert parsed["title"] == "A"


def test_wiki_article_round_trip(tmp_path):
    articles = [
        WikiArticle(id="1", url="http://a.com", title="Alpha", text="Alpha text " * 10),
        WikiArticle(id="2", url="http://b.com", title="Beta", text="Beta text " * 10),
    ]
    out = tmp_path / "test.jsonl"
    n = write_jsonl(iter(articles), out)
    assert n == 2

    loaded = list(read_jsonl(out))
    assert len(loaded) == 2
    assert loaded[0].title == "Alpha"
    assert loaded[1].title == "Beta"


def test_write_jsonl_creates_parent_dirs(tmp_path):
    out = tmp_path / "sub" / "dir" / "out.jsonl"
    articles = [WikiArticle(id="1", url="u", title="T", text="text here " * 5)]
    write_jsonl(iter(articles), out)
    assert out.exists()


# ---------------------------------------------------------------------------
# Data loader
# ---------------------------------------------------------------------------


def test_iter_articles(tmp_path):
    records = [
        {"id": "1", "url": "u1", "title": "T1", "text": "hello", "word_count": 1, "char_count": 5},
        {"id": "2", "url": "u2", "title": "T2", "text": "world", "word_count": 1, "char_count": 5},
    ]
    jsonl_path = tmp_path / "sample.jsonl"
    with jsonl_path.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    result = list(iter_articles(jsonl_path))
    assert len(result) == 2
    assert result[0]["title"] == "T1"
    assert result[1]["title"] == "T2"


def test_iter_articles_skips_empty_lines(tmp_path):
    jsonl_path = tmp_path / "sparse.jsonl"
    with jsonl_path.open("w") as fh:
        fh.write('{"id":"1","url":"u","title":"T","text":"x"}\n')
        fh.write("\n")
        fh.write('{"id":"2","url":"u","title":"T2","text":"y"}\n')
    result = list(iter_articles(jsonl_path))
    assert len(result) == 2


# ---------------------------------------------------------------------------
# iter_wikiextractor_output
# ---------------------------------------------------------------------------


def test_iter_wikiextractor_output(tmp_path):
    """Simulate the flat file layout produced by wikiextractor --json."""
    wiki_dir = tmp_path / "AA"
    wiki_dir.mkdir()
    wiki_file = wiki_dir / "wiki_00"
    records = [
        {"id": "10", "url": "http://en.wikipedia.org/wiki/Foo", "title": "Foo",
         "text": "Foo is a thing.\n"},
        {"id": "11", "url": "http://en.wikipedia.org/wiki/Bar", "title": "Bar",
         "text": "Bar is another thing.\n"},
    ]
    with wiki_file.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    result = list(iter_wikiextractor_output(tmp_path))
    assert len(result) == 2
    assert result[0]["title"] == "Foo"


# ---------------------------------------------------------------------------
# _build_articles (min_words filter)
# ---------------------------------------------------------------------------


def test_build_articles_min_words_filter(tmp_path):
    wiki_dir = tmp_path / "AA"
    wiki_dir.mkdir()
    wiki_file = wiki_dir / "wiki_00"
    records = [
        {"id": "1", "url": "u1", "title": "Short", "text": "Too short."},
        {"id": "2", "url": "u2", "title": "Long",
         "text": " ".join(["word"] * 60)},
    ]
    with wiki_file.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    articles = list(_build_articles(tmp_path, min_words=50))
    assert len(articles) == 1
    assert articles[0].title == "Long"
