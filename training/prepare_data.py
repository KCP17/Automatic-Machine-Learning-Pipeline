"""Regenerate ``data/raw.csv`` from a public dataset.

This is a **manual** data-refresh tool, not part of CI. CI consumes whatever
``data/raw.csv`` the DVC pointer resolves to; it never runs this script.

    python training/prepare_data.py
    dvc add data/raw.csv && dvc push
    git add data/raw.csv.dvc && git commit -m "Refresh dataset" && git push

Source: Rotten Tomatoes movie-review sentiment (Pang & Lee, 2005) — short
single-sentence reviews labelled 0 = negative, 1 = positive, evenly balanced.
A stratified sample of the train split is written so CI trains in minutes on a
CPU runner. Tune the size with SAMPLE_SIZE.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_FILE = REPO_ROOT / "data" / "raw.csv"

SOURCE = os.getenv("SOURCE_DATASET", "cornell-movie-review-data/rotten_tomatoes")
SPLIT = os.getenv("SOURCE_SPLIT", "train")
SAMPLE_SIZE = int(os.getenv("SAMPLE_SIZE", "3000"))
SEED = int(os.getenv("SEED", "42"))

_WHITESPACE = re.compile(r"\s+")


def main() -> None:
    dataset = load_dataset(SOURCE, split=SPLIT)
    df = dataset.to_pandas()[["text", "label"]]

    df["text"] = df["text"].map(lambda s: _WHITESPACE.sub(" ", str(s)).strip())
    df = df[(df["text"].str.len() > 0)].drop_duplicates(subset="text")
    df["label"] = df["label"].astype(int)

    # Even sample across labels so the eval split the pipeline carves out isn't
    # skewed. Falls back to whatever a class has if it's short.
    per_class = SAMPLE_SIZE // df["label"].nunique()
    sample = (
        pd.concat(
            g.sample(min(len(g), per_class), random_state=SEED)
            for _, g in df.groupby("label")
        )
        .sample(frac=1, random_state=SEED)  # shuffle
        .reset_index(drop=True)
    )

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_FILE, index=False)

    counts = sample["label"].value_counts().sort_index().to_dict()
    print(f"Wrote {len(sample)} rows to {OUT_FILE} (label counts: {counts})")


if __name__ == "__main__":
    main()
