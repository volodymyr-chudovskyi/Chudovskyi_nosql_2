"""
02_embed.py
Encode arXiv titles + abstracts with the SPECTER2 model (designed for
scientific-paper retrieval). Save vectors to embeddings/embeddings.npy and
the matching ids to embeddings/ids.npy.

Why SPECTER2: pretrained on citation graphs of scientific papers — produces
much better paper-to-paper similarity than a general-purpose sentence
transformer. Output dim = 768.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "arxiv_subset.parquet"
OUT_VECTORS = ROOT / "embeddings" / "embeddings.npy"
OUT_IDS = ROOT / "embeddings" / "ids.npy"

MODEL_NAME = "allenai/specter2_base"
BATCH = 32


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Run 01_prepare_data.py first; missing {IN_PATH}")

    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df):,} rows.")

    # SPECTER2 expects "TITLE [SEP] ABSTRACT".
    texts = (df["title"].fillna("") + " [SEP] " + df["abstract"].fillna("")).tolist()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading {MODEL_NAME} on {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    vectors = model.encode(
        texts,
        batch_size=BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # we normalize at index time if needed
    ).astype("float32")

    OUT_VECTORS.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT_VECTORS, vectors)
    np.save(OUT_IDS, df["id"].to_numpy())
    print(f"Saved embeddings shape={vectors.shape} → {OUT_VECTORS}")


if __name__ == "__main__":
    main()
