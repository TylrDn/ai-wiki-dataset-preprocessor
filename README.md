# ai-wiki-dataset-preprocessor

Pipeline to process Wikipedia dumps into model-ready JSONL/text format.

---

## Table of contents

1. [Overview](#overview)
2. [Directory structure](#directory-structure)
3. [Quickstart](#quickstart)
4. [Downloading a Wikipedia dump](#downloading-a-wikipedia-dump)
5. [Running the pipeline](#running-the-pipeline)
6. [Loading the processed dataset](#loading-the-processed-dataset)
7. [Development](#development)

---

## Overview

This project converts raw Wikipedia XML dumps into clean, tokeniser-friendly
JSONL files where **each line is one article** encoded as:

```json
{
  "id": "12",
  "url": "https://en.wikipedia.org/wiki/Anarchism",
  "title": "Anarchism",
  "text": "Anarchism is a political philosophy ...",
  "categories": [],
  "word_count": 12843,
  "char_count": 84201
}
```

---

## Directory structure

```
ai-wiki-dataset-preprocessor/
├── data/
│   ├── raw/          # Place downloaded dump files here (gitignored)
│   └── processed/    # Pipeline output lives here (gitignored)
├── notebooks/
│   └── 01_sample_wiki.ipynb   # End-to-end demo with a tiny sample
├── src/
│   ├── __init__.py
│   ├── preprocess.py  # wikiextractor wrapper + text cleaning
│   ├── schema.py      # WikiArticle dataclass + JSONL I/O helpers
│   └── data_loader.py # Lightweight & HuggingFace dataset loaders
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/TylrDn/ai-wiki-dataset-preprocessor.git
cd ai-wiki-dataset-preprocessor

# 2. Create and activate a virtual environment (Python 3.9+)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the sample notebook
jupyter notebook notebooks/01_sample_wiki.ipynb
```

---

## Downloading a Wikipedia dump

Full Wikipedia dumps are hosted at
[dumps.wikimedia.org](https://dumps.wikimedia.org/enwiki/latest/).

### English Wikipedia — articles only (recommended start)

```bash
# Latest articles dump (~22 GB compressed)
wget -P data/raw/ \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

### Smaller multi-stream dump (easier to sample)

```bash
# Articles multi-stream index + data (~22 GB total, randomly accessible)
wget -P data/raw/ \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream.xml.bz2
wget -P data/raw/ \
  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream-index.txt.bz2
```

### Other language editions

Replace `enwiki` with the language code, e.g. `dewiki`, `frwiki`, `zhwiki`:

```bash
wget -P data/raw/ \
  https://dumps.wikimedia.org/dewiki/latest/dewiki-latest-pages-articles.xml.bz2
```

> **Tip:** Dumps are released roughly once a month.
> Check [dumps.wikimedia.org/enwiki](https://dumps.wikimedia.org/enwiki/) for
> the exact date of the latest snapshot.

---

## Running the pipeline

```bash
python -m src.preprocess \
    --input  data/raw/enwiki-latest-pages-articles.xml.bz2 \
    --output data/processed/wiki.jsonl \
    --min-words 50
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input`  | *(required)* | Path to the Wikipedia dump file |
| `--output` | *(required)* | Destination JSONL file |
| `--min-words` | `50` | Skip articles shorter than this |
| `--log-level` | `INFO` | Logging verbosity |

Processing a full English Wikipedia dump takes **~2–4 hours** on a modern
laptop.  Output is roughly **20–25 GB** of uncompressed JSONL.

---

## Loading the processed dataset

### Pure Python (no extra dependencies)

```python
from src.data_loader import iter_articles

for article in iter_articles("data/processed/wiki.jsonl"):
    print(article["title"], article["word_count"])
```

### HuggingFace `datasets`

```python
from src.data_loader import load_wiki_dataset

ds = load_wiki_dataset("data/processed/wiki.jsonl")
print(ds)
# Dataset({features: ['id', 'url', 'title', 'text', 'categories',
#                      'word_count', 'char_count'], num_rows: ...})
```

---

## Development

```bash
# Run tests
pytest tests/ -v

# Lint
flake8 src/ tests/
```

The CI workflow (`.github/workflows/ci.yml`) runs linting and the test suite
on every push and pull request.
