"""
17_doc_motif_distribution.py

Per-document soft mixture over the 25 anchor topics + an explicit
"unexplained" share, using Non-Negative Least Squares (NNLS).

For each document embedding e_d (R^768, L2-normalised) we solve:

    min_w  || e_d − Σ_k w_k · c_k ||²    subject to   w_k ≥ 0   (∀ k = 1..25)

where c_k is the L2-normalised SBERT centroid of anchor topic k. The
NNLS solution gives non-negative weights w_k expressing each motif's
contribution to reconstructing the document embedding. The residual

    r_d = || e_d − Σ_k w_k · c_k ||

quantifies what the 25-motif union does *not* capture for d.

We then normalise so each document yields a distribution that sums to
1:

    motif_k(d)   = w_k / ( Σ_k w_k + r_d )
    unexplained(d) = r_d / ( Σ_k w_k + r_d )

So a document might be: 72% T1 + 23% T7 + 3% T11 + 2% unexplained, etc.

### References

- Lawson, C. L. & Hanson, R. J. 1974. *Solving Least Squares Problems*.
  Prentice-Hall. (Canonical NNLS algorithm.)
- Bro, R. & De Jong, S. 1997. "A fast non-negativity-constrained least
  squares algorithm." *Journal of Chemometrics* 11(5):393–401.
- Cutler, A. & Breiman, L. 1994. "Archetypal analysis." *Technometrics*
  36(4):338–347. (Conceptual sibling — express each sample as a non-
  negative combination of "archetype" reference points.)
- For the topic-modelling context of soft doc-topic assignment, see
  Blei, Ng & Jordan 2003 (LDA) — though LDA assumes a generative
  topic-word multinomial and is not equivalent to NNLS in embedding
  space.

### Assumptions and limitations (paper-grade)

1. *Linearity in SBERT space.* NNLS reconstructs e_d as a non-negative
   linear combination of centroids. SBERT embeddings carry cosine
   semantics, so Euclidean residual is a proxy for "remaining content,"
   not an exact information-theoretic measure.
2. *Reference set = 25 anchor topics.* The unexplained share is
   relative to the *current* topic-centroid basis. Different K, or
   different baseline configuration, would yield different unexplained
   shares.
3. *No identifiability of mixture proportions across runs.* The
   weights are determined by the anchor centroids; a different anchor
   (e.g. seed=256 baseline) would give different per-document
   distributions.

We report these in the paper limitations.

### Outputs

  results/distribution/doc_motif_distribution.csv      (N rows, 25 motif % + unexplained % + work_id + assigned_topic_hard)
  results/distribution/per_doc_top3.csv                top-3 motifs per doc + their %s
  results/distribution/summary.json                    mean / median / std per motif
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
FILT = ROOT / "data" / "filtered"
BASE = ROOT / "results" / "baseline"
OUT = ROOT / "results" / "distribution"
OUT.mkdir(parents=True, exist_ok=True)

# Six audit-robust motifs for downstream summary
ROBUST_TOPICS = [0, 1, 7, 11, 14, 16]


def parse_topwords(s):
    if not isinstance(s, str) or not s:
        return []
    return [w.split("(")[0].strip() for w in s.split(",") if "(" in w]


def main():
    t_total = time.time()
    print("Loading data ...")
    embeddings = np.load(BASE / "embeddings.npy")
    per_doc = pd.read_csv(BASE / "topic_per_doc.csv")
    topic_info = pd.read_csv(BASE / "topic_info.csv")
    topic_info = topic_info[topic_info["Topic"] != -1].reset_index(drop=True)
    N = embeddings.shape[0]
    K = len(topic_info)
    print(f"  N={N} docs, K={K} anchor topics")

    # ----- L2-normalise embeddings and centroids -----
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

    # Centroid matrix C of shape (768, K) so that NNLS solves Cw ≈ e
    C = centroids.T

    print("\n[1] Solving NNLS per document ...")
    weights = np.zeros((N, K))
    residual_norms = np.zeros(N)
    for i in tqdm(range(N), desc="  NNLS"):
        e = emb_norm[i]
        w, r = nnls(C, e, maxiter=1000)
        weights[i] = w
        residual_norms[i] = r

    print(f"  Mean residual norm: {residual_norms.mean():.4f}")
    print(f"  Mean Σw across docs: {weights.sum(axis=1).mean():.4f}")

    # ----- Normalise to a distribution summing to 1 -----
    print("\n[2] Normalising to (motif% + unexplained%) summing to 100% ...")
    sum_w = weights.sum(axis=1)
    denom = sum_w + residual_norms
    # Avoid divide by zero
    denom_safe = np.maximum(denom, 1e-12)
    motif_share = weights / denom_safe[:, None]
    unexplained_share = residual_norms / denom_safe
    # Sanity: rows should sum to 1
    sanity = motif_share.sum(axis=1) + unexplained_share
    print(f"  Sanity row sum (should be 1.0): mean={sanity.mean():.4f}, std={sanity.std():.6f}")
    print(f"  Mean unexplained%: {100 * unexplained_share.mean():.2f}%")
    print(f"  Mean Σ motif%: {100 * motif_share.sum(axis=1).mean():.2f}%")

    # ----- Build per-doc distribution table -----
    print("\n[3] Building distribution table ...")
    cols = [f"T{int(tid)}_pct" for tid in topic_ids]
    df = pd.DataFrame(motif_share * 100, columns=cols)
    df.insert(0, "work_id", per_doc["work_id"].values)
    df.insert(1, "assigned_topic_hard", per_doc["topic"].values)
    df["unexplained_pct"] = unexplained_share * 100
    # Top-1 motif
    top1_idx = motif_share.argmax(axis=1)
    df["top1_topic"] = [topic_ids[i] for i in top1_idx]
    df["top1_pct"] = motif_share[np.arange(N), top1_idx] * 100
    df["row_sum_pct"] = df[cols + ["unexplained_pct"]].sum(axis=1)
    df.to_csv(OUT / "doc_motif_distribution.csv", index=False)
    print(f"  saved doc_motif_distribution.csv ({len(df)} rows)")

    # ----- Top-3 motifs per doc (compact view) -----
    print("\n[4] Top-3 motifs per doc ...")
    top3_rows = []
    for i in range(N):
        order = np.argsort(-motif_share[i])
        t1, t2, t3 = order[0], order[1], order[2]
        top3_rows.append({
            "work_id": per_doc["work_id"].iloc[i],
            "assigned_topic_hard": int(per_doc["topic"].iloc[i]),
            "top1_topic": int(topic_ids[t1]),
            "top1_pct": round(float(motif_share[i, t1] * 100), 2),
            "top2_topic": int(topic_ids[t2]),
            "top2_pct": round(float(motif_share[i, t2] * 100), 2),
            "top3_topic": int(topic_ids[t3]),
            "top3_pct": round(float(motif_share[i, t3] * 100), 2),
            "unexplained_pct": round(float(unexplained_share[i] * 100), 2),
        })
    pd.DataFrame(top3_rows).to_csv(OUT / "per_doc_top3.csv", index=False)

    # ----- Summary statistics -----
    print("\n[5] Summary statistics ...")
    per_motif_mean = motif_share.mean(axis=0) * 100
    per_motif_median = np.median(motif_share, axis=0) * 100
    per_motif_max = motif_share.max(axis=0) * 100
    summary_rows = []
    for k, tid in enumerate(topic_ids):
        kw = parse_topwords(topic_info.iloc[k]["TopWords"])[:5]
        summary_rows.append({
            "topic_id": int(tid),
            "in_audit_robust_6": int(tid) in ROBUST_TOPICS,
            "topwords": ", ".join(kw),
            "mean_pct_across_docs": round(float(per_motif_mean[k]), 3),
            "median_pct": round(float(per_motif_median[k]), 3),
            "max_pct": round(float(per_motif_max[k]), 3),
            "n_docs_dominant": int((np.argmax(motif_share, axis=1) == k).sum()),
        })
    pm_df = pd.DataFrame(summary_rows).sort_values("mean_pct_across_docs",
                                                    ascending=False)
    pm_df.to_csv(OUT / "per_motif_summary.csv", index=False)

    # global summary
    sum_motif = motif_share.sum(axis=1)
    summary = {
        "N_docs": int(N),
        "K_topics": int(K),
        "global": {
            "mean_unexplained_pct": round(100 * float(unexplained_share.mean()), 3),
            "median_unexplained_pct": round(100 * float(np.median(unexplained_share)), 3),
            "std_unexplained_pct": round(100 * float(unexplained_share.std()), 3),
            "mean_total_motif_pct": round(100 * float(sum_motif.mean()), 3),
            "share_docs_unexplained_gt_30pct": float((unexplained_share > 0.30).mean()),
            "share_docs_unexplained_gt_50pct": float((unexplained_share > 0.50).mean()),
            "share_docs_top1_gt_50pct": float((motif_share.max(axis=1) > 0.50).mean()),
            "share_docs_top1_lt_20pct": float((motif_share.max(axis=1) < 0.20).mean()),
        },
        "audit_robust_6": {
            "topics": ROBUST_TOPICS,
            "total_share_mean_pct": round(
                100 * float(motif_share[:, [topic_ids.index(t) for t in ROBUST_TOPICS]].sum(axis=1).mean()), 3),
            "n_docs_top1_in_robust_6": int(sum(
                topic_ids[top1_idx[i]] in ROBUST_TOPICS for i in range(N))),
        },
        "outlier_subset": None,
        "elapsed_s": round(time.time() - t_total, 1),
    }
    # outlier subset
    out_mask = (per_doc["topic"] == -1).values
    if out_mask.any():
        summary["outlier_subset"] = {
            "n": int(out_mask.sum()),
            "mean_unexplained_pct": round(100 * float(unexplained_share[out_mask].mean()), 3),
            "mean_top1_pct": round(100 * float(motif_share[out_mask].max(axis=1).mean()), 3),
            "share_docs_unexplained_gt_30pct": float((unexplained_share[out_mask] > 0.30).mean()),
        }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    # ----- Top per-motif print -----
    print("\n  Per-motif mean share across corpus (sorted):")
    for _, r in pm_df.iterrows():
        flag = "★" if r["in_audit_robust_6"] else " "
        print(f"    T{int(r['topic_id']):>2} {flag}  mean {r['mean_pct_across_docs']:>5.2f}%  "
              f"max {r['max_pct']:>5.1f}%  n_dom={int(r['n_docs_dominant']):>4d}  | {r['topwords']}")

    print()
    print(f"  Mean unexplained % across {N} docs: {summary['global']['mean_unexplained_pct']}%")
    print(f"  Median unexplained %: {summary['global']['median_unexplained_pct']}%")
    print(f"  Docs with unexplained > 30%: {100*summary['global']['share_docs_unexplained_gt_30pct']:.1f}%")
    print(f"  Docs with unexplained > 50%: {100*summary['global']['share_docs_unexplained_gt_50pct']:.1f}%")
    print(f"  Docs with top1 > 50% (pure-ish): {100*summary['global']['share_docs_top1_gt_50pct']:.1f}%")
    print(f"  Docs with top1 < 20% (very mixed): {100*summary['global']['share_docs_top1_lt_20pct']:.1f}%")
    print(f"\n  Audit-robust 6 share of total motif mass: "
          f"{summary['audit_robust_6']['total_share_mean_pct']}%")

    print("\n" + "=" * 70)
    print("Doc-motif distribution (NNLS) — headline")
    print("=" * 70)
    print(f"  Stored: doc × 25 motif + unexplained %, summing to 100% per doc")
    print(f"  Method: NNLS [Lawson & Hanson 1974]")
    print(f"  Mean unexplained %: {summary['global']['mean_unexplained_pct']}%")
    print(f"  Elapsed: {summary['elapsed_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
