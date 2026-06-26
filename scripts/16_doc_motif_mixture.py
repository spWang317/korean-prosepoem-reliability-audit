"""
16_doc_motif_mixture.py

Doc-level motif mixture (LDA-style soft assignment) computed from
SBERT embedding × topic-centroid similarities. Stored for later use in
qualitative analysis (paper §6).

BERTopic's default is hard assignment (each doc → one topic or outlier).
We compute a soft alternative by direct centroid similarity:

  For each document d and topic k:
    sim(d, k) = cos(emb_d, centroid_k)

  Soft mixture:
    mixture(d, k) = softmax_k(sim(d, k) / τ)     τ = temperature

  Normalised similarity (no softmax):
    nsim(d, k) = sim(d, k) / sum_k sim(d, k)

Two views:
  (A) Full 25-topic mixture for every doc — diagnostic / future use
  (B) 6-robust-motif sub-mixture — sea, flower, rain, meta-poetic,
      music, cat (the audit-robust 4/4 motifs)

Outputs:
  results/mixture/doc_topic_sim_full.npz       (N, K) cosine sim matrix
  results/mixture/doc_motif_mixture_6.csv      per-doc 6-motif softmax
  results/mixture/per_doc_dominant.csv         dominant motif + strength
  results/mixture/representative_docs_per_motif.json  top-10 docs per robust motif
  results/mixture/outlier_motif_distribution.csv     outlier docs → motif mixture
  results/mixture/summary.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path(__file__).resolve().parent.parent
FILT = ROOT / "data" / "filtered"
BASE = ROOT / "results" / "baseline"
OUT = ROOT / "results" / "mixture"
OUT.mkdir(parents=True, exist_ok=True)

# 6 audit-robust motifs
ROBUST_TOPICS = [0, 1, 7, 11, 14, 16]
MOTIF_LABEL = {
    0: "sea",
    1: "flower",
    7: "rain",
    11: "meta_poetic",
    14: "music",
    16: "cat",
}

SOFTMAX_TEMP = 0.1  # temperature for sharper soft assignment


def main():
    t_total = time.time()
    print("Loading data ...")
    embeddings = np.load(BASE / "embeddings.npy")
    per_doc = pd.read_csv(BASE / "topic_per_doc.csv")
    topic_info = pd.read_csv(BASE / "topic_info.csv")
    topic_info = topic_info[topic_info["Topic"] != -1].reset_index(drop=True)
    works = pd.read_parquet(FILT / "works_filtered.parquet")
    N = embeddings.shape[0]
    K = len(topic_info)
    print(f"  N={N} docs, K={K} topics")

    # ---------- Centroids ---------------
    emb_norm = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12
    )
    topic_ids = topic_info["Topic"].tolist()
    centroids = np.zeros((K, embeddings.shape[1]))
    for k, tid in enumerate(topic_ids):
        mask = (per_doc["topic"] == tid).values
        if mask.sum() > 0:
            centroids[k] = emb_norm[mask].mean(axis=0)
    centroids = centroids / np.maximum(
        np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12
    )

    # ---------- Full doc × topic similarity ---------------
    print("\n[1] Computing full doc × topic similarity matrix...")
    sim_mat = emb_norm @ centroids.T  # (N, K)
    np.savez_compressed(
        OUT / "doc_topic_sim_full.npz",
        sim_matrix=sim_mat,
        topic_ids=np.array(topic_ids),
        work_ids=per_doc["work_id"].values,
    )
    print(f"  shape: {sim_mat.shape}")
    print(f"  mean sim: {sim_mat.mean():.3f}, range [{sim_mat.min():.3f}, {sim_mat.max():.3f}]")

    # ---------- 6-motif sub-mixture ---------------
    print("\n[2] Computing 6-robust-motif mixture (softmax temperature τ=%.2f)..." % SOFTMAX_TEMP)
    robust_idx = [topic_ids.index(t) for t in ROBUST_TOPICS]
    sim_6 = sim_mat[:, robust_idx]  # (N, 6)
    # Softmax over 6 motifs with temperature
    mix_6 = softmax(sim_6 / SOFTMAX_TEMP, axis=1)

    mix_df = pd.DataFrame(
        mix_6, columns=[f"p_{MOTIF_LABEL[t]}" for t in ROBUST_TOPICS]
    )
    mix_df.insert(0, "work_id", per_doc["work_id"].values)
    mix_df.insert(1, "assigned_topic", per_doc["topic"].values)
    mix_df["is_outlier"] = mix_df["assigned_topic"] == -1
    # Dominant motif
    dom_idx = mix_6.argmax(axis=1)
    mix_df["dominant_motif"] = [MOTIF_LABEL[ROBUST_TOPICS[i]] for i in dom_idx]
    mix_df["dominant_prob"] = mix_6[np.arange(N), dom_idx]
    # Second-best
    second_idx = mix_6.argsort(axis=1)[:, -2]
    mix_df["second_motif"] = [MOTIF_LABEL[ROBUST_TOPICS[i]] for i in second_idx]
    mix_df["second_prob"] = mix_6[np.arange(N), second_idx]
    # Margin between top1 and top2
    mix_df["motif_margin"] = mix_df["dominant_prob"] - mix_df["second_prob"]
    # Entropy (normalised) — how "mixed" the doc is
    eps = 1e-12
    H = -(mix_6 * np.log(mix_6 + eps)).sum(axis=1)
    Hmax = np.log(6)
    mix_df["motif_entropy_norm"] = H / Hmax

    mix_df.to_csv(OUT / "doc_motif_mixture_6.csv", index=False)
    print(f"  saved doc_motif_mixture_6.csv")

    # Distribution of dominant motif
    dom_counts = mix_df["dominant_motif"].value_counts()
    print(f"  Dominant-motif distribution among 4642 docs:")
    for m, c in dom_counts.items():
        print(f"    {m:<12s}: {c:>5d} ({100*c/N:>4.1f}%)")
    print(f"  Mean motif entropy (0=pure, 1=uniform): {mix_df['motif_entropy_norm'].mean():.3f}")
    print(f"  Docs with motif_margin < 0.05 (mixed): "
          f"{(mix_df['motif_margin'] < 0.05).sum()} ({100*(mix_df['motif_margin']<0.05).mean():.1f}%)")
    print(f"  Docs with motif_margin > 0.50 (pure):  "
          f"{(mix_df['motif_margin'] > 0.50).sum()} ({100*(mix_df['motif_margin']>0.50).mean():.1f}%)")

    # ---------- Per-doc dominant (full 25 topic) ---------------
    print("\n[3] Per-doc dominant topic (all 25) ...")
    sim_norm = softmax(sim_mat / SOFTMAX_TEMP, axis=1)
    dom_full = sim_norm.argmax(axis=1)
    dom_full_tid = [topic_ids[i] for i in dom_full]
    pd_df = pd.DataFrame({
        "work_id": per_doc["work_id"].values,
        "assigned_topic_hard": per_doc["topic"].values,
        "dominant_topic_soft": dom_full_tid,
        "dominant_prob": sim_norm[np.arange(N), dom_full],
        "agreement_hard_soft": [a == b for a, b in zip(per_doc["topic"].values, dom_full_tid)],
    })
    pd_df.to_csv(OUT / "per_doc_dominant.csv", index=False)
    agreement = pd_df["agreement_hard_soft"].mean()
    print(f"  Hard (BERTopic) ↔ Soft (centroid-sim softmax) agreement: {100*agreement:.1f}%")

    # ---------- Representative docs per robust motif ---------------
    print("\n[4] Top-10 representative docs per robust motif (by softmax prob)...")
    works_dict = dict(zip(works["work_id"], works["원문"]))
    repr_per_motif = {}
    for tid, label in MOTIF_LABEL.items():
        col = f"p_{label}"
        top_docs = mix_df.nlargest(10, col)[["work_id", col, "assigned_topic"]]
        repr_per_motif[label] = []
        for _, r in top_docs.iterrows():
            text = works_dict.get(r["work_id"], "")
            repr_per_motif[label].append({
                "work_id": str(r["work_id"]),
                "prob": round(float(r[col]), 4),
                "assigned_topic_hard": int(r["assigned_topic"]),
                "text_preview": text[:200],
            })
    (OUT / "representative_docs_per_motif.json").write_text(
        json.dumps(repr_per_motif, indent=2, ensure_ascii=False))
    print(f"  saved representative_docs_per_motif.json")

    # ---------- Outlier motif distribution ---------------
    print("\n[5] Outlier motif distribution ...")
    outlier_mask = mix_df["is_outlier"]
    outlier_dom = mix_df.loc[outlier_mask, "dominant_motif"].value_counts()
    outlier_summary = {
        "n_outliers": int(outlier_mask.sum()),
        "dominant_motif_distribution": outlier_dom.to_dict(),
        "mean_motif_entropy_outlier": float(mix_df.loc[outlier_mask, "motif_entropy_norm"].mean()),
        "mean_motif_entropy_inlier": float(mix_df.loc[~outlier_mask, "motif_entropy_norm"].mean()),
        "share_mixed_outlier_margin_lt_005": float(
            (mix_df.loc[outlier_mask, "motif_margin"] < 0.05).mean()),
        "share_mixed_inlier_margin_lt_005": float(
            (mix_df.loc[~outlier_mask, "motif_margin"] < 0.05).mean()),
    }
    # save outlier-only mixture file
    mix_df.loc[outlier_mask].to_csv(OUT / "outlier_motif_distribution.csv", index=False)
    print(f"  Outlier dominant-motif distribution:")
    for m, c in outlier_dom.items():
        print(f"    {m:<12s}: {c:>5d}")
    print(f"  Mean entropy — outlier {outlier_summary['mean_motif_entropy_outlier']:.3f}  vs inlier {outlier_summary['mean_motif_entropy_inlier']:.3f}")

    # ---------- Summary ---------------
    summary = {
        "N_docs": N,
        "K_baseline_topics": K,
        "robust_motifs_indexed": [{"tid": t, "label": MOTIF_LABEL[t]} for t in ROBUST_TOPICS],
        "softmax_temperature": SOFTMAX_TEMP,
        "global_motif_stats": {
            "dominant_distribution": dom_counts.to_dict(),
            "mean_motif_entropy": float(mix_df["motif_entropy_norm"].mean()),
            "share_pure_margin_gt_050": float((mix_df["motif_margin"] > 0.50).mean()),
            "share_mixed_margin_lt_005": float((mix_df["motif_margin"] < 0.05).mean()),
        },
        "hard_soft_agreement": float(agreement),
        "outlier_summary": outlier_summary,
        "elapsed_s": round(time.time() - t_total, 1),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("Doc-motif mixture — headline")
    print("=" * 70)
    print(f"  Stored: full sim matrix (4642 × 25), 6-motif softmax mixture,")
    print(f"          per-doc dominant, representative docs, outlier mixture.")
    print(f"  Hard ↔ Soft (top1) agreement: {100*agreement:.1f}%")
    print(f"  Mean 6-motif entropy: {mix_df['motif_entropy_norm'].mean():.3f}")
    print(f"  Elapsed: {summary['elapsed_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
