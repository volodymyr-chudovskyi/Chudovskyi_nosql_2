"""
04_search.py
Three flavours of vector search over the Pinecone index, demonstrating:

  1. Pure semantic search (natural-language query → top-k papers).
  2. Filtered search (semantic + metadata predicate: year/category).
  3. Metric comparison: cosine vs dot product vs Euclidean — same query,
     different scoring, side-by-side ranking.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

API_KEY = os.environ["PINECONE_API_KEY"]
INDEX = os.environ.get("PINECONE_INDEX_NAME", "arxiv-search")
MODEL_NAME = "allenai/specter2_base"

QUERIES = [
    "transformer architecture for natural language understanding",
    "graph neural networks applied to molecular property prediction",
    "differential privacy in federated learning",
]


def encode(model: SentenceTransformer, text: str) -> np.ndarray:
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=False)[0]


def pretty(label: str, matches) -> None:
    print(f"\n--- {label} ---")
    for m in matches:
        md = m["metadata"]
        print(f"  {m['score']:.3f}  [{md.get('primary_category','?')} {md.get('year','?')}]  {md.get('title','')[:90]}")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = SentenceTransformer(MODEL_NAME, device=device)
    pc = Pinecone(api_key=API_KEY)
    index = pc.Index(INDEX)

    # ---------- (1) plain semantic search ----------
    print("\n========== (1) Pure semantic search ==========")
    q = QUERIES[0]
    qv = encode(model, q)
    res = index.query(vector=qv.tolist(), top_k=5, include_metadata=True)
    pretty(f"Q: {q!r}", res["matches"])

    # ---------- (2) semantic + metadata filter ----------
    print("\n========== (2) Filter: recent NLP papers only ==========")
    q = QUERIES[0]
    res = index.query(
        vector=qv.tolist(),
        top_k=5,
        include_metadata=True,
        filter={"primary_category": {"$eq": "cs.CL"}, "year": {"$gte": 2022}},
    )
    pretty(f"Q: {q!r} | cs.CL & year>=2022", res["matches"])

    # ---------- (3) metric comparison ----------
    print("\n========== (3) Metric comparison ==========")
    q = QUERIES[1]
    qv = encode(model, q)

    # Pinecone serverless index was created with cosine; for an apples-to-
    # apples metric comparison we pull the top-100 candidates by cosine and
    # rerank them locally with cosine / dot / L2.
    cand = index.query(vector=qv.tolist(), top_k=100, include_values=True, include_metadata=True)
    rows = [(m, np.asarray(m["values"], dtype="float32")) for m in cand["matches"]]

    by_cos = sorted(rows, key=lambda r: -cosine_sim(qv, r[1]))[:5]
    by_dot = sorted(rows, key=lambda r: -float(np.dot(qv, r[1])))[:5]
    by_l2 = sorted(rows, key=lambda r: l2_dist(qv, r[1]))[:5]

    print(f"\n>> Q: {q!r}")
    for label, ranked in [("cosine", by_cos), ("dot product", by_dot), ("euclidean L2", by_l2)]:
        print(f"\n  ~ ranked by {label}")
        for m, v in ranked:
            md = m["metadata"]
            print(f"    {md.get('title','')[:90]}")


if __name__ == "__main__":
    main()
