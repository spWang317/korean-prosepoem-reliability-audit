"""
08_soft_cluster_overlap.py

Direct quantification of the "motifs clear but inter-permeating" hypothesis:
Korean prose poem topics may appear clear (high membership reliability,
high κ) yet inter-penetrate (boundaries between topics are soft).

Four diagnostics, computed on the locked baseline (25 topics, seed=100):

  1. Topic centroid distance matrix
     - Centroid = mean SBERT embedding of docs in each baseline topic.
     - Topic-topic cosine similarity matrix.
     - Closest topic pairs = "permeation" candidates.

  2. Per-doc soft membership distribution
     - For each doc, cosine similarity to all K topic centroids.
     - Margin = top1 - top2 sim (small margin → ambiguous).
     - Entropy of softmax(sim) (high entropy → uncertain).

  3. Outlier (T = -1) analysis
     - For each outlier doc, distance to nearest topic centroid.
     - "Inter-topic" outliers = docs whose top1 and top2 sim are close.
     - Distribution: which topics outliers gravitate to.

  4. Robust subset permeation
     - For the audit-robust topics, are their docs also close to other
       topics' centroids? Or are they truly distinct?

References
  - Soft clustering vs hard clustering distinction [McLachlan & Peel 2000]
  - Cluster ambiguity quantification via entropy [Shannon 1948]
  - Outlier as "between cluster" interpretation in density-based clustering
    [Campello et al. 2015, HDBSCAN paper]

Outputs:
  results/soft_cluster/centroid_similarity.csv   K×K cosine sim matrix
  results/soft_cluster/per_doc_soft.csv          per-doc top1, top2, margin, entropy
  results/soft_cluster/outlier_analysis.json     outlier distance & gravitation
  results/soft_cluster/robust_subset_check.csv   robust topics' permeation
  results/soft_cluster/summary.json              headline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parent.parent
FILT_DIR = ROOT / "data" / "filtered"
TOK_DIR = ROOT / "data" / "tokenized"
BASE = ROOT / "results" / "baseline"
PSYCH = ROOT / "results" / "psychometric"
SENS = ROOT / "results" / "sensitivity"
OUT = ROOT / "results" / "soft_cluster"
OUT.mkdir(parents=True, exist_ok=True)


def main():
    t_total = time.time()
    print("Loading data...")
    embeddings = np.load(BASE / "embeddings.npy")  # (N, 768)
    per_doc = pd.read_csv(BASE / "topic_per_doc.csv")
    topic_info = pd.read_csv(BASE / "topic_info.csv")
    topic_info = topic_info[topic_info["Topic"] != -1].reset_index(drop=True)
    K = len(topic_info)
    N = embeddings.shape[0]
    assert len(per_doc) == N
    print(f"  N={N}, K={K} non-outlier topics")

    # L2-normalise for cosine
    emb_norm = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )

    # ---------------------------------------------------- 1. Topic centroids
    print("\n[1] Computing topic centroids...")
    topic_ids = topic_info["Topic"].tolist()
    centroids = np.zeros((K, embeddings.shape[1]))
    topic_size = np.zeros(K, dtype=int)
    for k, tid in enumerate(topic_ids):
        mask = (per_doc["topic"] == tid).values
        if mask.sum() > 0:
            centroids[k] = emb_norm[mask].mean(axis=0)
            topic_size[k] = int(mask.sum())
    # re-normalise centroids
    centroids = centroids / np.maximum(
        np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12
    )
    sim_kk = centroids @ centroids.T  # (K, K) cosine sim
    # off-diagonal
    off = sim_kk.copy()
    np.fill_diagonal(off, -np.inf)
    print(f"  Mean off-diagonal sim: {(sim_kk.sum() - K) / (K*K - K):.3f}")
    print(f"  Max off-diagonal sim:  {off.max():.3f}")
    print(f"  Min off-diagonal sim:  {off[np.isfinite(off)].min():.3f}")

    # closest topic pairs
    pairs = []
    for i in range(K):
        for j in range(i+1, K):
            pairs.append((sim_kk[i, j], topic_ids[i], topic_ids[j],
                          topic_info.iloc[i]["TopWords"][:60],
                          topic_info.iloc[j]["TopWords"][:60]))
    pairs.sort(reverse=True)
    print(f"\n  Top-10 closest topic pairs (potential permeation):")
    for s, ti, tj, wi, wj in pairs[:10]:
        print(f"    sim={s:.3f}  T{ti}↔T{tj}")
        print(f"      T{ti}: {wi}")
        print(f"      T{tj}: {wj}")

    sim_df = pd.DataFrame(sim_kk, index=[f"T{t}" for t in topic_ids],
                          columns=[f"T{t}" for t in topic_ids])
    sim_df.to_csv(OUT / "centroid_similarity.csv")

    # ---------------------------------------------------- 2. Per-doc soft membership
    print("\n[2] Per-doc soft membership distribution...")
    doc_topic_sim = emb_norm @ centroids.T  # (N, K)
    # sort each row to get top-1, top-2
    sorted_sims = -np.sort(-doc_topic_sim, axis=1)
    top1 = sorted_sims[:, 0]
    top2 = sorted_sims[:, 1]
    margin = top1 - top2
    # softmax + entropy (temperature τ = 1.0 for diagnostic purposes)
    soft = softmax(doc_topic_sim, axis=1)
    ent = entropy(soft.T)  # entropy along K-dim
    # entropy in nats; normalize by max entropy log(K)
    ent_norm = ent / np.log(K)

    soft_df = pd.DataFrame({
        "work_id": per_doc["work_id"].values,
        "assigned_topic": per_doc["topic"].values,
        "top1_sim": top1,
        "top2_sim": top2,
        "margin": margin,
        "entropy_nats": ent,
        "entropy_normalized": ent_norm,
    })
    soft_df.to_csv(OUT / "per_doc_soft.csv", index=False)

    n_outlier = int((per_doc["topic"] == -1).sum())
    n_inlier = N - n_outlier
    print(f"  N inlier docs:    {n_inlier}")
    print(f"  N outlier docs:   {n_outlier}")
    print(f"  Mean margin:      {margin.mean():.3f}")
    print(f"  Mean entropy (norm): {ent_norm.mean():.3f}  (0=hard, 1=uniform)")
    print(f"  Docs with margin < 0.05: {int((margin < 0.05).sum())} ({100*(margin<0.05).mean():.1f}%)")
    print(f"  Docs with margin < 0.10: {int((margin < 0.10).sum())} ({100*(margin<0.10).mean():.1f}%)")
    print(f"  Docs with entropy_norm > 0.80: {int((ent_norm > 0.8).sum())} ({100*(ent_norm>0.8).mean():.1f}%)")

    # ---------------------------------------------------- 3. Outlier analysis
    print("\n[3] Outlier (T=-1) analysis...")
    outlier_mask = (per_doc["topic"] == -1).values
    outlier_top1 = top1[outlier_mask]
    outlier_top2 = top2[outlier_mask]
    outlier_margin = margin[outlier_mask]
    outlier_ent = ent_norm[outlier_mask]
    inlier_top1 = top1[~outlier_mask]
    inlier_margin = margin[~outlier_mask]
    inlier_ent = ent_norm[~outlier_mask]

    # which topic each outlier is closest to
    outlier_top1_idx = doc_topic_sim[outlier_mask].argmax(axis=1)
    closest_topic_counts = pd.Series([topic_ids[i] for i in outlier_top1_idx]).value_counts()

    outlier_summary = {
        "n_outliers": int(outlier_mask.sum()),
        "outlier_top1_mean":    round(float(outlier_top1.mean()), 4),
        "outlier_top1_median":  round(float(np.median(outlier_top1)), 4),
        "outlier_top1_std":     round(float(outlier_top1.std()), 4),
        "inlier_top1_mean":     round(float(inlier_top1.mean()), 4),
        "outlier_margin_mean":  round(float(outlier_margin.mean()), 4),
        "inlier_margin_mean":   round(float(inlier_margin.mean()), 4),
        "outlier_entropy_norm_mean": round(float(outlier_ent.mean()), 4),
        "inlier_entropy_norm_mean":  round(float(inlier_ent.mean()), 4),
        "inter_topic_outliers_margin_lt_005": int((outlier_margin < 0.05).sum()),
        "inter_topic_outliers_share": round(float((outlier_margin < 0.05).mean()), 4),
        "outlier_gravitation_top5": closest_topic_counts.head(5).to_dict(),
    }
    print(f"  Outlier top-1 sim:   mean={outlier_summary['outlier_top1_mean']}")
    print(f"  Inlier top-1 sim:    mean={outlier_summary['inlier_top1_mean']}")
    print(f"  Outlier margin:      mean={outlier_summary['outlier_margin_mean']}")
    print(f"  Inlier margin:       mean={outlier_summary['inlier_margin_mean']}")
    print(f"  'Inter-topic' outliers (margin<0.05): "
          f"{outlier_summary['inter_topic_outliers_margin_lt_005']} "
          f"({100*outlier_summary['inter_topic_outliers_share']:.1f}%)")
    print(f"  Outliers gravitate most to (top 5):")
    for tid, n in list(closest_topic_counts.head(5).items()):
        kw = topic_info.loc[topic_info["Topic"] == tid, "TopWords"].values
        kw_str = kw[0][:60] if len(kw) else ""
        print(f"    T{tid} ({n}): {kw_str}")

    (OUT / "outlier_analysis.json").write_text(
        json.dumps(outlier_summary, indent=2, ensure_ascii=False))

    # ---------------------------------------------------- 4. Robust subset check
    print("\n[4] Robust subset permeation check...")
    # Load the robust subset (τ=0.5 + α≥0.7, the 5 topics)
    try:
        pmtx = pd.read_csv(SENS / "persistence_plus_psychometric.csv")
    except FileNotFoundError:
        print("  Skipped: persistence_plus_psychometric.csv not found")
        pmtx = None

    robust_rows = []
    if pmtx is not None:
        # Strict (2 topics)
        strict = pmtx[
            (pmtx["R1_persistence"] >= 0.7) &
            (pmtx["R2_persistence"] >= 0.7) &
            (pmtx["R3star_persistence"] >= 0.7) &
            (pmtx["cronbach_alpha"] >= 0.7)
        ]
        # Reasonable (5 topics)
        reasonable = pmtx[
            (pmtx["R1_persistence"] >= 0.5) &
            (pmtx["R2_persistence"] >= 0.5) &
            (pmtx["R3star_persistence"] >= 0.5) &
            (pmtx["cronbach_alpha"] >= 0.7)
        ]
        for label, subset in [("strict_2", strict), ("reasonable_5", reasonable)]:
            for _, row in subset.iterrows():
                tid = int(row["topic_id"])
                if tid not in topic_ids:
                    continue
                k = topic_ids.index(tid)
                # docs in this topic
                docs_in = (per_doc["topic"] == tid).values
                if docs_in.sum() == 0:
                    continue
                # for these docs, distribution of top1 sim to *own* topic vs others
                own_sim = doc_topic_sim[docs_in, k]
                other_sim = doc_topic_sim[docs_in].copy()
                other_sim[:, k] = -np.inf
                next_best_sim = other_sim.max(axis=1)
                own_margin = own_sim - next_best_sim
                robust_rows.append({
                    "label": label,
                    "topic_id": tid,
                    "topwords": row["topwords"][:80],
                    "size": int(docs_in.sum()),
                    "mean_own_sim": round(float(own_sim.mean()), 4),
                    "mean_next_best_sim": round(float(next_best_sim.mean()), 4),
                    "mean_own_margin": round(float(own_margin.mean()), 4),
                    "share_low_margin": round(float((own_margin < 0.05).mean()), 4),
                })
    robust_df = pd.DataFrame(robust_rows)
    if not robust_df.empty:
        robust_df.to_csv(OUT / "robust_subset_check.csv", index=False)
        print("  Robust subset permeation:")
        print("    label          T   own_sim  next_best  margin  low_margin_share  topwords")
        for _, r in robust_df.iterrows():
            print(f"    {r['label']:<12s} T{r['topic_id']:<2}  "
                  f"{r['mean_own_sim']:>6.3f}   {r['mean_next_best_sim']:>6.3f}   "
                  f"{r['mean_own_margin']:>6.3f}   {100*r['share_low_margin']:>4.1f}%   "
                  f"{r['topwords']}")

    # ---------------------------------------------------- 5. Summary
    summary = {
        "N": int(N), "K": int(K),
        "centroid_similarity": {
            "mean_off_diag": round(float((sim_kk.sum() - K) / (K*K - K)), 4),
            "max_off_diag": round(float(off.max()), 4),
            "min_off_diag": round(float(off[np.isfinite(off)].min()), 4),
            "top3_closest_pairs": [
                {"T_a": int(ti), "T_b": int(tj), "sim": round(float(s), 4)}
                for s, ti, tj, _, _ in pairs[:3]
            ],
        },
        "per_doc_soft": {
            "mean_margin": round(float(margin.mean()), 4),
            "mean_entropy_normalized": round(float(ent_norm.mean()), 4),
            "share_margin_lt_005": round(float((margin < 0.05).mean()), 4),
            "share_entropy_gt_080": round(float((ent_norm > 0.8).mean()), 4),
        },
        "outlier_vs_inlier": outlier_summary,
        "robust_subset_count_checked": int(len(robust_df)) if not robust_df.empty else 0,
        "elapsed_s": round(time.time() - t_total, 1),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("Soft cluster overlap — headline")
    print("=" * 70)
    print(f"  Topic centroid mean off-diag sim: {summary['centroid_similarity']['mean_off_diag']}")
    print(f"  Per-doc mean margin (top1-top2):  {summary['per_doc_soft']['mean_margin']}")
    print(f"  Per-doc mean entropy (normalized): {summary['per_doc_soft']['mean_entropy_normalized']}")
    print(f"  Docs with margin < 0.05:           {100*summary['per_doc_soft']['share_margin_lt_005']:.1f}%")
    print(f"  Outliers margin < 0.05 share:      {100*outlier_summary['inter_topic_outliers_share']:.1f}%")
    print(f"  Elapsed: {summary['elapsed_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
