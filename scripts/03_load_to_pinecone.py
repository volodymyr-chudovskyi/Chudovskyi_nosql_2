"""
03_load_to_pinecone.py
Create (or reuse) a Pinecone serverless index and upload all vectors with
metadata so 04_search.py can run filter+search queries.

Requires PINECONE_API_KEY in .env. Free serverless tier is enough.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATA = ROOT / "data" / "arxiv_subset.parquet"
VECS = ROOT / "embeddings" / "embeddings.npy"
IDS = ROOT / "embeddings" / "ids.npy"

API_KEY = os.environ["PINECONE_API_KEY"]
INDEX = os.environ.get("PINECONE_INDEX_NAME", "arxiv-search")
CLOUD = os.environ.get("PINECONE_CLOUD", "aws")
REGION = os.environ.get("PINECONE_REGION", "us-east-1")

BATCH = 100


def main() -> None:
    vectors = np.load(VECS)
    ids = np.load(IDS, allow_pickle=True)
    df = pd.read_parquet(DATA).set_index("id")
    dim = vectors.shape[1]
    print(f"Loaded {len(vectors):,} vectors, dim={dim}")

    pc = Pinecone(api_key=API_KEY)
    if INDEX not in [i["name"] for i in pc.list_indexes()]:
        print(f"Creating index '{INDEX}' (dim={dim}, cosine)...")
        pc.create_index(
            name=INDEX,
            dimension=dim,
            metric="cosine",  # SPECTER2 → cosine works well; see 04_search comparison
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
    index = pc.Index(INDEX)

    pbar = tqdm(range(0, len(vectors), BATCH), desc="Upsert")
    for start in pbar:
        end = min(start + BATCH, len(vectors))
        items = []
        for i in range(start, end):
            paper_id = str(ids[i])
            row = df.loc[paper_id]
            items.append(
                {
                    "id": paper_id,
                    "values": vectors[i].tolist(),
                    "metadata": {
                        "title": str(row["title"])[:1000],
                        "primary_category": str(row["primary_category"]),
                        "year": int(row["year"]) if row["year"] else 0,
                        "authors": str(row["authors"])[:500],
                    },
                }
            )
        index.upsert(vectors=items)

    stats = index.describe_index_stats()
    print(f"Index stats: {stats}")


if __name__ == "__main__":
    main()
