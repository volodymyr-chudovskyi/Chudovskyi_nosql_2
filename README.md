# HW2 — Semantic Search over arXiv

Build an end-to-end semantic search pipeline for scientific papers: ingest the
arXiv dataset, embed with **SPECTER2**, index in **Pinecone**, then compare
pure-semantic, filtered, chunked, and hybrid (BM25 + vector via **RRF**)
retrieval.

Coursework for *NoSQL and Vector Databases* (GoIT), Topic 7 → Завдання 2.

## Why these choices

| Decision | Why |
|----------|-----|
| **SPECTER2** model | Trained on the citation graph of scientific papers — much better paper-to-paper similarity than a general-purpose `all-MiniLM`. Output is 768-d. |
| **Cosine** as primary metric | SPECTER2 embeddings encode topic in the *direction* of the vector; magnitude is uninformative, so cosine is more stable than dot product across abstracts of different lengths. `04_search.py` shows the side-by-side. |
| **Pinecone serverless** | Spec requirement. Free tier is enough for 8K vectors. Index is created with `cosine`. |
| **Chunk size 256 tokens + 20% overlap** | Abstracts are mostly short, so chunking only matters for the longest ones. 256/50 is the typical RAG sweet spot — small enough that each chunk is one tight idea, big enough to retain context. `05_chunking.py` compares this against a no-overlap and a sentence-based variant. |
| **RRF for hybrid** | Score-free fusion: no need to normalize BM25 scores against cosine. With `k=60` (Cormack et al.), a paper ranked top by either method bubbles up; a paper ranked decently by both wins. |

## Pipeline

```
01_prepare_data.py   →  data/arxiv_subset.parquet
02_embed.py          →  embeddings/{embeddings.npy, ids.npy}
03_load_to_pinecone  →  Pinecone index (~8K vectors)
04_search.py         →  semantic / filtered / metric comparison
05_chunking.py       →  fixed vs overlap vs sentence-based
06_hybrid_search.py  →  BM25 + vector via RRF
```

## Setup

```bash
# 1. Python env
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Pinecone API key
cp .env.example .env
# edit .env, set PINECONE_API_KEY=...

# 3. arXiv data
# Download arxiv-metadata-oai-snapshot.json from
# https://www.kaggle.com/datasets/Cornell-University/arxiv
# Place it under data/
```

## Run

```bash
python scripts/01_prepare_data.py        # ~30s after first download
python scripts/02_embed.py               # 2-5 min on CPU, <1 min on GPU
python scripts/03_load_to_pinecone.py    # ~1 min (creates index + upserts)
python scripts/04_search.py              # demo: semantic / filter / metrics
python scripts/05_chunking.py            # chunking strategy comparison
python scripts/06_hybrid_search.py       # BM25 vs vector vs RRF
```

## Project layout

```
hw2-semantic-search/
├── .env.example
├── .gitignore
├── requirements.txt
├── README.md
├── data/                  (gitignored: contains parquet + arXiv dump)
├── embeddings/            (gitignored: .npy)
└── scripts/
    ├── 01_prepare_data.py
    ├── 02_embed.py
    ├── 03_load_to_pinecone.py
    ├── 04_search.py
    ├── 05_chunking.py
    └── 06_hybrid_search.py
```

## What each script demonstrates (key answers for the homework writeup)

- **Vector vs full-text** — `04_search.py` returns relevant papers for queries whose terms don't appear literally; BM25 in `06_hybrid_search.py` fails on the same queries when phrasing differs.
- **Chunking trade-off** — `05_chunking.py` shows that for short abstracts the strategy matters less, but overlap helps when an abstract straddles two themes.
- **Why RRF wins** — `06_hybrid_search.py` shows examples where BM25 ranks an exact-match paper #1 but misses synonym matches, and the vector index finds the synonym matches but misses an obvious keyword hit — RRF lands both in top-5.

## Notes / known limitations

- `02_embed.py` downloads SPECTER2 weights (~440 MB) on first run.
- Pinecone serverless free tier is region-locked; if `us-east-1` is unavailable on your free account, change `PINECONE_REGION` in `.env`.
- `05_chunking.py` uses a smaller 1500-paper subset to keep the in-memory comparison fast; final answers will be representative but not exhaustive.
