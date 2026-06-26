"""
19_poet_motif_specialty.py

Per-poet motif specialty: which motifs each poet's works concentrate in.

For each poet (작가):
  - Number of works in our corpus
  - Mean motif % across their works (per topic + unexplained)
  - "Specialty motifs": motifs where the poet's mean % is notably higher
    than the corpus mean
  - Mean unexplained %

For each motif:
  - Poets whose works concentrate most in that motif

Outputs:
  results/poet_specialty/poet_motif_mean.csv          poet × motif mean %
  results/poet_specialty/poet_specialty_summary.csv   one-row-per-poet summary
  results/poet_specialty/motif_top_poets.csv          per-motif top poets
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILT = ROOT / "data" / "filtered"
DIST = ROOT / "results" / "distribution"
OUT = ROOT / "results" / "poet_specialty"
OUT.mkdir(parents=True, exist_ok=True)

ROBUST_TOPICS = {0, 1, 7, 11, 14, 16}
MIN_WORKS_FOR_INCLUSION = 3   # poet must have ≥ this many works


def main():
    t_total = time.time()
    print("Loading data ...")
    df = pd.read_csv(DIST / "doc_motif_distribution.csv")
    works = pd.read_parquet(FILT / "works_filtered.parquet")
    print(f"  N={len(df)} docs, poets in works={works['작가'].nunique()}")

    # Merge poet info
    poet_lookup = dict(zip(works["work_id"], works["작가"].fillna("(미상)")))
    df["poet"] = df["work_id"].map(poet_lookup)

    motif_cols = [c for c in df.columns if c.startswith("T") and c.endswith("_pct")]
    all_cols = motif_cols + ["unexplained_pct"]

    # Corpus-level mean (reference)
    corpus_mean = df[all_cols].mean()

    # Per-poet aggregation
    print("\n[1] Per-poet mean motif % ...")
    agg = df.groupby("poet").agg(
        n_works=("work_id", "count"),
        **{c: (c, "mean") for c in all_cols}
    ).reset_index()
    agg_filtered = agg[agg["n_works"] >= MIN_WORKS_FOR_INCLUSION].copy()
    print(f"  Total poets: {len(agg)}, with ≥{MIN_WORKS_FOR_INCLUSION} works: {len(agg_filtered)}")

    # Specialty score: poet's mean - corpus mean (per motif)
    print("\n[2] Computing specialty scores ...")
    specialty = agg_filtered[motif_cols].sub(corpus_mean[motif_cols], axis=1)
    # For each poet, top-3 specialty motifs
    summary_rows = []
    for _, r in agg_filtered.iterrows():
        poet = r["poet"]
        nw = int(r["n_works"])
        # Top 3 motifs by raw mean %
        means_only = r[motif_cols].astype(float)
        top3 = means_only.nlargest(3)
        # Top 3 specialty (above corpus mean)
        spec = (means_only - corpus_mean[motif_cols].astype(float)).nlargest(3)
        summary_rows.append({
            "poet": poet,
            "n_works": nw,
            "mean_unexplained_pct": round(float(r["unexplained_pct"]), 2),
            "top1_motif": top3.index[0].replace("_pct", ""),
            "top1_pct": round(float(top3.iloc[0]), 2),
            "top2_motif": top3.index[1].replace("_pct", ""),
            "top2_pct": round(float(top3.iloc[1]), 2),
            "top3_motif": top3.index[2].replace("_pct", ""),
            "top3_pct": round(float(top3.iloc[2]), 2),
            "specialty1_motif": spec.index[0].replace("_pct", ""),
            "specialty1_excess_pct": round(float(spec.iloc[0]), 2),
            "specialty2_motif": spec.index[1].replace("_pct", ""),
            "specialty2_excess_pct": round(float(spec.iloc[1]), 2),
            "specialty3_motif": spec.index[2].replace("_pct", ""),
            "specialty3_excess_pct": round(float(spec.iloc[2]), 2),
        })
    summary = pd.DataFrame(summary_rows).sort_values("n_works", ascending=False)
    summary.to_csv(OUT / "poet_specialty_summary.csv", index=False)
    print(f"  saved poet_specialty_summary.csv ({len(summary)} poets)")

    # Save full poet × motif matrix
    agg_filtered.to_csv(OUT / "poet_motif_mean.csv", index=False)
    print(f"  saved poet_motif_mean.csv")

    # Per-motif top poets
    print("\n[3] Per-motif top poets ...")
    motif_top = []
    for col in motif_cols:
        tid = int(col.split("_")[0][1:])
        top10 = agg_filtered.nlargest(10, col)[["poet", "n_works", col]]
        for rank, (_, r) in enumerate(top10.iterrows(), start=1):
            motif_top.append({
                "topic_id": tid,
                "is_robust": tid in ROBUST_TOPICS,
                "rank": rank,
                "poet": r["poet"],
                "n_works": int(r["n_works"]),
                "mean_motif_pct": round(float(r[col]), 2),
                "excess_over_corpus_mean": round(float(r[col]) - corpus_mean[col], 2),
            })
    pd.DataFrame(motif_top).to_csv(OUT / "motif_top_poets.csv", index=False)
    print(f"  saved motif_top_poets.csv")

    # Print headline
    print("\n" + "=" * 75)
    print("Poet specialty — top 15 poets by # works in our corpus")
    print("=" * 75)
    head = summary.head(15)
    for _, r in head.iterrows():
        print(f"  {r['poet']:>10s} (n={int(r['n_works']):>3d})  "
              f"top: {r['top1_motif']}({r['top1_pct']:.1f}%) "
              f"{r['top2_motif']}({r['top2_pct']:.1f}%) "
              f"{r['top3_motif']}({r['top3_pct']:.1f}%) "
              f"| unexp {r['mean_unexplained_pct']:.1f}%")

    print(f"\n  Total elapsed: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
