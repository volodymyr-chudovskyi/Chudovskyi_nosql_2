"""
01_prepare_data.py
Read the arXiv JSONL dump, keep a subset of papers, clean basic fields,
save as Parquet for downstream scripts.

Input  : data/arxiv-metadata-oai-snapshot.json  (download from
         https://www.kaggle.com/datasets/Cornell-University/arxiv)
Output : data/arxiv_subset.parquet
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
IN_PATH = ROOT / "data" / "arxiv-metadata-oai-snapshot.json"
OUT_PATH = ROOT / "data" / "arxiv_subset.parquet"

N_RECORDS = 8000
ALLOWED_CATS = {"cs.LG", "cs.CL", "cs.AI", "cs.IR", "cs.CV", "stat.ML"}

WS = re.compile(r"\s+")


def clean(text: str) -> str:
    return WS.sub(" ", text).strip()


def iter_rows(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main() -> None:
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Place the arXiv snapshot at {IN_PATH}. Download from "
            "https://www.kaggle.com/datasets/Cornell-University/arxiv"
        )

    # Taking the FIRST N matching records would give an all-2007-2013 subset
    # (the snapshot is ordered oldest-first) and make year-based filters in
    # 04_search.py meaningless. Instead keep every K-th matching record so the
    # subset spans the whole snapshot's timeline.
    KEEP_EVERY = 60  # ~500k matching records / 60 ≈ 8.3k → trimmed to N_RECORDS

    rows: list[dict] = []
    seen = 0
    for rec in tqdm(iter_rows(IN_PATH), desc="Scanning arXiv"):
        cats = (rec.get("categories") or "").split()
        if not any(c in ALLOWED_CATS for c in cats):
            continue
        title = clean(rec.get("title") or "")
        abstract = clean(rec.get("abstract") or "")
        if len(abstract) < 100:
            continue
        seen += 1
        if seen % KEEP_EVERY:
            continue
        update_year = (rec.get("update_date") or "")[:4]
        rows.append(
            {
                "id": rec["id"],
                "title": title,
                "abstract": abstract,
                "authors": clean(rec.get("authors") or ""),
                "primary_category": cats[0] if cats else "",
                "categories": " ".join(cats),
                "year": int(update_year) if update_year.isdigit() else 0,
            }
        )
        if len(rows) >= N_RECORDS:
            break

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows → {OUT_PATH}")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
