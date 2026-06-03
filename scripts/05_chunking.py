"""
05_chunking.py
Compare three chunking strategies on the same long abstracts:

  A. Fixed-size (256 tokens).
  B. Fixed-size with 20% overlap (sliding window).
  C. Sentence-based with hard cap.

For each strategy we (re-)embed the chunks, run the same set of queries, and
report which strategy puts the most relevant paper in the top-3 most often.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "arxiv_subset.parquet"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 3

QUERIES = [
    ("transformer attention bottleneck and long-context inference", "cs.CL"),
    ("contrastive self-supervised representation learning for images", "cs.CV"),
    ("differential privacy guarantees in federated learning", "cs.LG"),
]

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def fixed_chunks(text: str, size: int = 256, overlap: int = 0) -> list[str]:
    toks = text.split()
    if not toks:
        return []
    step = max(1, size - overlap)
    return [" ".join(toks[i : i + size]) for i in range(0, len(toks), step)]


def sentence_chunks(text: str, max_tokens: int = 256) -> list[str]:
    out: list[str] = []
    buf: list[str] = []
    n = 0
    for s in SENT_SPLIT.split(text):
        toks = s.split()
        if n + len(toks) > max_tokens and buf:
            out.append(" ".join(buf))
            buf, n = [], 0
        buf.extend(toks)
        n += len(toks)
    if buf:
        out.append(" ".join(buf))
    return out


def embed_chunks(model: SentenceTransformer, df: pd.DataFrame, chunker, label: str) -> tuple[np.ndarray, list[int]]:
    chunks: list[str] = []
    parent: list[int] = []  # parent paper row index
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"chunking [{label}]"):
        text = f"{row['title']} {row['abstract']}"
        for c in chunker(text):
            chunks.append(c)
            parent.append(i)
    vecs = model.encode(chunks, batch_size=32, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    return vecs, parent


def top_paper_idx(qv: np.ndarray, chunk_vecs: np.ndarray, parent: list[int], k: int) -> list[int]:
    # cosine via L2-normalize
    cv = chunk_vecs / (np.linalg.norm(chunk_vecs, axis=1, keepdims=True) + 1e-12)
    qn = qv / (np.linalg.norm(qv) + 1e-12)
    scores = cv @ qn
    # collapse to per-paper score = max of its chunk scores
    by_paper: dict[int, float] = {}
    for s, p in zip(scores, parent):
        if s > by_paper.get(p, -1):
            by_paper[p] = float(s)
    ranked = sorted(by_paper.items(), key=lambda kv: -kv[1])
    return [p for p, _ in ranked[:k]]


def main() -> None:
    df = pd.read_parquet(DATA).reset_index(drop=True).head(1500)  # smaller subset for chunking benchmark
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer(MODEL_NAME, device=device)

    strategies = {
        "fixed_256": (lambda t: fixed_chunks(t, 256, 0)),
        "fixed_256_overlap": (lambda t: fixed_chunks(t, 256, 50)),
        "sentence_256": (lambda t: sentence_chunks(t, 256)),
    }

    results: dict[str, list[list[str]]] = defaultdict(list)

    for name, fn in strategies.items():
        cv, parent = embed_chunks(model, df, fn, name)
        for q, expected_cat in QUERIES:
            qv = model.encode([q], convert_to_numpy=True)[0]
            top_rows = top_paper_idx(qv, cv, parent, TOP_K)
            titles = [df.iloc[r]["title"][:80] for r in top_rows]
            cats = [df.iloc[r]["primary_category"] for r in top_rows]
            hits = sum(1 for c in cats if c == expected_cat)
            results[name].append([f"hits={hits}/{TOP_K}"] + titles)

    print("\n=== Chunking comparison ===")
    for name, rows in results.items():
        print(f"\n>> {name}")
        for (q, _), info in zip(QUERIES, rows):
            print(f"  Q: {q[:70]}")
            for x in info:
                print(f"    - {x}")


if __name__ == "__main__":
    main()
