"""
06_hybrid_search.py
Hybrid retrieval: BM25 (lexical) + dense vector (semantic), fused with
Reciprocal Rank Fusion (RRF). For each query we print the top-5 from
BM25-only, vector-only, and the RRF combination — and show cases where the
hybrid moves a relevant paper that neither method alone ranked high enough.

RRF formula:
    score(d) = Σ_r  1 / (k + rank_r(d))    [k = 60 in the original paper]
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data" / "arxiv_subset.parquet"
API_KEY = os.environ["PINECONE_API_KEY"]
INDEX = os.environ.get("PINECONE_INDEX_NAME", "arxiv-search")
MODEL_NAME = "allenai/specter2_base"

TOK = re.compile(r"\w+")

QUERIES = [
    "How do attention mechanisms scale with sequence length?",
    "Distillation of large language models into smaller students.",
    "Adversarial robustness of image classifiers under PGD attack.",
]

K_RRF = 60
TOP_N = 5


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in TOK.findall(text)]


def bm25_rank(bm25: BM25Okapi, df: pd.DataFrame, query: str, n: int) -> list[str]:
    scores = bm25.get_scores(tokenize(query))
    top = np.argsort(-scores)[:n]
    return df.iloc[top]["id"].astype(str).tolist()


def vector_rank(index, model: SentenceTransformer, query: str, n: int) -> list[str]:
    qv = model.encode([query], convert_to_numpy=True)[0].tolist()
    res = index.query(vector=qv, top_k=n, include_metadata=False)
    return [m["id"] for m in res["matches"]]


def rrf(ranked_lists: list[list[str]], k: int = K_RRF, top_n: int = TOP_N) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranks in ranked_lists:
        for rank, doc_id in enumerate(ranks, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_n]


def show(label: str, ids: list[str], df_by_id: pd.DataFrame) -> None:
    print(f"\n  {label}")
    for i, paper_id in enumerate(ids, 1):
        try:
            title = df_by_id.loc[paper_id]["title"]
            cat = df_by_id.loc[paper_id]["primary_category"]
            print(f"    {i}. [{cat}] {str(title)[:90]}")
        except KeyError:
            print(f"    {i}. (missing metadata for {paper_id})")


def main() -> None:
    df = pd.read_parquet(DATA)
    df["id"] = df["id"].astype(str)
    df_by_id = df.set_index("id")

    corpus = (df["title"].fillna("") + " " + df["abstract"].fillna("")).map(tokenize).tolist()
    bm25 = BM25Okapi(corpus)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer(MODEL_NAME, device=device)

    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(INDEX)

    for q in QUERIES:
        print("\n" + "=" * 80)
        print(f"Q: {q}")
        b = bm25_rank(bm25, df, q, TOP_N * 4)
        v = vector_rank(index, model, q, TOP_N * 4)
        fused = [doc for doc, _ in rrf([b, v])]
        show("BM25 only", b[:TOP_N], df_by_id)
        show("Vector only", v[:TOP_N], df_by_id)
        show("Hybrid (RRF)", fused, df_by_id)


if __name__ == "__main__":
    main()
