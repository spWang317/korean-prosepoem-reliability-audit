"""
13_visibility_quantitative.py

Quantitative-only analysis that anchors §6 (visibility analysis) of the paper.
NO qualitative judgement here — every output is computed directly from data.
Qualitative motif-mapping with critical literature happens *after* this script,
on top of these tables, in a separate (manual) reading pass.

Five analyses:

  [1] Topic profile table — all 25 baseline topics with
      size, top-10 keywords, R1/R2/R3/R3*/α/κ/c_npmi
      → distinguishes 6 robust vs 19 non-robust at a glance

  [2] Non-robust topic specificity vs robust subset
      For each topic: vocabulary-uniqueness ratio (fraction of top-10 keywords
      that appear in no other topic's top-10). Compares robust 6 vs non-robust 19.

  [3] Outlier (49%) profile
      - Length distribution (chars, sentences) of outliers vs inliers
      - Top-K topic gravitation (already in Finding H, extended with margin
        statistics)
      - For each topic, how many outliers gravitate to it

  [4] Soft-cluster permeation pairs
      Top 20 closest non-self topic pairs from centroid_similarity.csv,
      reporting:
      - which is robust, which is not
      - keyword overlap between pair
      - inferred "shared semantic field" (paper §6 anchor, NOT named here)

  [5] Robust vs non-robust geometry
      - Mean centroid sim to ALL OTHER centroids
      - Mean own_sim (cohesion) vs next-best
      - For each topic, distance to nearest robust topic

Outputs:
  results/visibility/topic_profile.csv         all 25 topics, one row each
  results/visibility/specificity.csv           per-topic keyword uniqueness
  results/visibility/outlier_profile.csv       outliers vs inliers
  results/visibility/outlier_gravitation.csv   topic-level outlier counts
  results/visibility/cluster_pairs.csv         top 20 closest topic pairs
  results/visibility/geometry.csv              robust-vs-nonrobust geometry
  results/visibility/summary.json              headline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILT = ROOT / "data" / "filtered"
TOK = ROOT / "data" / "tokenized"
BASE = ROOT / "results" / "baseline"
PSYCH = ROOT / "results" / "psychometric"
SENS = ROOT / "results" / "sensitivity"
SOFT = ROOT / "results" / "soft_cluster"
COH = ROOT / "results" / "coherence"
OUT = ROOT / "results" / "visibility"
OUT.mkdir(parents=True, exist_ok=True)

# Six audit-robust motifs (4/4 axes passed in pre-registered audit)
ROBUST_TOPICS = {0, 1, 7, 11, 14, 16}


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
    K = len(topic_info)
    persistence = pd.read_csv(SENS / "persistence_plus_psychometric.csv")
    coherence = pd.read_csv(COH / "per_topic_coherence.csv")
    centroid_sim = pd.read_csv(SOFT / "centroid_similarity.csv", index_col=0)
    per_doc_soft = pd.read_csv(SOFT / "per_doc_soft.csv")
    works = pd.read_parquet(FILT / "works_filtered.parquet")
    print(f"  K={K} topics, N={len(per_doc)} docs")

    # ============================================================
    # [1] Topic profile table
    # ============================================================
    print("\n[1] Building topic profile table ...")
    rows = []
    for _, t in topic_info.iterrows():
        tid = int(t["Topic"])
        words = parse_topwords(t["TopWords"])
        p = persistence[persistence["topic_id"] == tid].iloc[0] \
            if (persistence["topic_id"] == tid).any() else None
        c = coherence[coherence["topic_id"] == tid].iloc[0] \
            if (coherence["topic_id"] == tid).any() else None
        row = {
            "topic_id": tid,
            "is_robust": tid in ROBUST_TOPICS,
            "size": int(t["Count"]),
            "topwords": ", ".join(words[:10]),
            "R1": float(p["R1_persistence"]) if p is not None else None,
            "R2": float(p["R2_persistence"]) if p is not None else None,
            "R3_raw": float(p["R3_persistence"]) if p is not None else None,
            "R3_star": float(p["R3star_persistence"]) if p is not None else None,
            "alpha": float(p["cronbach_alpha"]) if p is not None and not pd.isna(p["cronbach_alpha"]) else None,
            "kappa": float(p["mean_pairwise_kappa"]) if p is not None and not pd.isna(p["mean_pairwise_kappa"]) else None,
            "c_v": float(c["c_v"]) if c is not None and not pd.isna(c["c_v"]) else None,
            "c_npmi": float(c["c_npmi"]) if c is not None and not pd.isna(c["c_npmi"]) else None,
            "u_mass": float(c["u_mass"]) if c is not None and not pd.isna(c["u_mass"]) else None,
            "c_uci": float(c["c_uci"]) if c is not None and not pd.isna(c["c_uci"]) else None,
        }
        rows.append(row)
    profile = pd.DataFrame(rows).sort_values("is_robust", ascending=False)
    profile.to_csv(OUT / "topic_profile.csv", index=False)
    print(f"  saved topic_profile.csv ({len(profile)} rows)")

    # ============================================================
    # [2] Specificity: how unique is each topic's keyword set
    # ============================================================
    print("\n[2] Computing keyword specificity per topic ...")
    all_topic_words = {int(t.Topic): parse_topwords(t.TopWords)
                       for _, t in topic_info.iterrows()}
    spec_rows = []
    for tid, words in all_topic_words.items():
        other_words = set()
        for o_tid, o_words in all_topic_words.items():
            if o_tid != tid:
                other_words.update(o_words[:10])
        own = set(words[:10])
        unique = own - other_words
        shared = own & other_words
        spec_rows.append({
            "topic_id": tid,
            "is_robust": tid in ROBUST_TOPICS,
            "n_unique_in_top10": len(unique),
            "n_shared_in_top10": len(shared),
            "unique_ratio": round(len(unique) / max(1, len(own)), 3),
            "unique_terms": ", ".join(sorted(unique)),
            "shared_terms": ", ".join(sorted(shared)),
        })
    spec = pd.DataFrame(spec_rows).sort_values("is_robust", ascending=False)
    spec.to_csv(OUT / "specificity.csv", index=False)
    print(f"  robust 6 mean unique_ratio: "
          f"{spec[spec['is_robust']]['unique_ratio'].mean():.3f}")
    print(f"  non-robust 19 mean unique_ratio: "
          f"{spec[~spec['is_robust']]['unique_ratio'].mean():.3f}")

    # ============================================================
    # [3] Outlier profile
    # ============================================================
    print("\n[3] Building outlier profile ...")
    merged = per_doc.merge(works[["work_id", "원문"]], on="work_id", how="left")
    merged["is_outlier"] = (merged["topic"] == -1)
    merged["char_len"] = merged["원문"].fillna("").str.len()

    # sentences (use \n or 마침표/물음표/느낌표 as proxy)
    def n_sent(s):
        if not isinstance(s, str):
            return 0
        import re
        return max(1, len(re.split(r"[.!?。！？]\s*", s.strip())))
    merged["n_sent"] = merged["원문"].apply(n_sent)
    outlier_profile = merged.groupby("is_outlier").agg(
        n_works=("work_id", "count"),
        mean_char_len=("char_len", "mean"),
        median_char_len=("char_len", "median"),
        mean_n_sent=("n_sent", "mean"),
        median_n_sent=("n_sent", "median"),
    ).round(2)
    outlier_profile.to_csv(OUT / "outlier_profile.csv")
    print("  outlier profile:")
    print(outlier_profile)

    # Outlier gravitation per topic
    grav = per_doc_soft[per_doc_soft["assigned_topic"] == -1].copy()
    # find each outlier's top-1 topic from the per_doc_soft 's top1_sim
    # we don't have the topic id here directly, so recompute:
    emb_norm = embeddings / np.maximum(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
    # rebuild centroids
    topic_ids = topic_info["Topic"].tolist()
    centroids = np.zeros((len(topic_ids), embeddings.shape[1]))
    for k, tid in enumerate(topic_ids):
        mask = (per_doc["topic"] == tid).values
        if mask.sum() > 0:
            centroids[k] = emb_norm[mask].mean(axis=0)
    centroids = centroids / np.maximum(
        np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    outlier_mask = (per_doc["topic"] == -1).values
    sims = emb_norm[outlier_mask] @ centroids.T
    top1 = sims.argmax(axis=1)
    sorted_sims = -np.sort(-sims, axis=1)
    margin = sorted_sims[:, 0] - sorted_sims[:, 1]
    grav_rows = []
    for k, tid in enumerate(topic_ids):
        mask = (top1 == k)
        if not mask.any():
            continue
        m_in = margin[mask]
        grav_rows.append({
            "topic_id": int(tid),
            "is_robust": int(tid) in ROBUST_TOPICS,
            "n_outliers_drawn": int(mask.sum()),
            "mean_top1_sim": float(sorted_sims[mask, 0].mean()),
            "mean_margin": float(m_in.mean()),
            "share_margin_lt_005": float((m_in < 0.05).mean()),
        })
    grav_df = pd.DataFrame(grav_rows).sort_values("n_outliers_drawn",
                                                  ascending=False)
    grav_df.to_csv(OUT / "outlier_gravitation.csv", index=False)
    print(f"  outlier gravitation: top-5 topics drawing outliers")
    for _, r in grav_df.head(5).iterrows():
        kw_row = topic_info[topic_info["Topic"] == r["topic_id"]].iloc[0]
        kw = parse_topwords(kw_row["TopWords"])[:5]
        print(f"    T{int(r['topic_id'])}({'★' if r['is_robust'] else ' '})  "
              f"n={int(r['n_outliers_drawn'])}  margin={r['mean_margin']:.3f}  "
              f"share<0.05={r['share_margin_lt_005']*100:.0f}%  | {', '.join(kw)}")

    # ============================================================
    # [4] Top closest topic pairs (cluster permeation)
    # ============================================================
    print("\n[4] Top closest topic pairs ...")
    pairs = []
    for i, ti in enumerate(topic_ids):
        for j, tj in enumerate(topic_ids):
            if i >= j:
                continue
            sim = float(centroid_sim.iloc[i, j])
            kw_i = set(parse_topwords(topic_info[topic_info["Topic"]==ti]["TopWords"].iloc[0])[:10])
            kw_j = set(parse_topwords(topic_info[topic_info["Topic"]==tj]["TopWords"].iloc[0])[:10])
            overlap = kw_i & kw_j
            pairs.append({
                "T_a": int(ti),
                "T_b": int(tj),
                "T_a_robust": int(ti) in ROBUST_TOPICS,
                "T_b_robust": int(tj) in ROBUST_TOPICS,
                "centroid_sim": round(sim, 4),
                "keyword_overlap_count": len(overlap),
                "keyword_overlap_top10": ", ".join(sorted(overlap)),
                "T_a_kw": ", ".join(sorted(kw_i)),
                "T_b_kw": ", ".join(sorted(kw_j)),
            })
    pairs_df = pd.DataFrame(pairs).sort_values("centroid_sim", ascending=False)
    pairs_df.head(30).to_csv(OUT / "cluster_pairs.csv", index=False)
    print(f"  top-10 closest pairs:")
    for _, p in pairs_df.head(10).iterrows():
        r_a = "★" if p["T_a_robust"] else " "
        r_b = "★" if p["T_b_robust"] else " "
        print(f"    sim={p['centroid_sim']:.3f}  T{int(p['T_a']):>2}{r_a} ↔ "
              f"T{int(p['T_b']):>2}{r_b}  kw_overlap={int(p['keyword_overlap_count'])}")

    # ============================================================
    # [5] Robust vs non-robust geometry
    # ============================================================
    print("\n[5] Geometry: robust vs non-robust ...")
    geom_rows = []
    for k, tid in enumerate(topic_ids):
        # mean cosine to all other centroids
        other_sims = [centroid_sim.iloc[k, j] for j in range(len(topic_ids)) if j != k]
        own_docs_mask = (per_doc["topic"] == tid).values
        if own_docs_mask.sum() > 0:
            own_emb = emb_norm[own_docs_mask]
            own_sims = own_emb @ centroids[k]
            own_sim_mean = float(own_sims.mean())
        else:
            own_sim_mean = float("nan")
        # nearest robust topic distance
        robust_idx = [j for j, t in enumerate(topic_ids) if t in ROBUST_TOPICS and j != k]
        if robust_idx:
            nearest_robust_sim = max(centroid_sim.iloc[k, j] for j in robust_idx)
        else:
            nearest_robust_sim = float("nan")
        geom_rows.append({
            "topic_id": int(tid),
            "is_robust": int(tid) in ROBUST_TOPICS,
            "size": int(topic_info[topic_info["Topic"]==tid]["Count"].iloc[0]),
            "mean_cos_to_other_centroids": round(float(np.mean(other_sims)), 4),
            "mean_own_doc_sim_to_centroid": round(own_sim_mean, 4),
            "max_cos_to_robust_centroid": round(float(nearest_robust_sim), 4),
        })
    geom_df = pd.DataFrame(geom_rows).sort_values("is_robust", ascending=False)
    geom_df.to_csv(OUT / "geometry.csv", index=False)

    r6 = geom_df[geom_df["is_robust"]]
    n19 = geom_df[~geom_df["is_robust"]]
    print(f"  Robust 6:    mean_cos_to_other = {r6['mean_cos_to_other_centroids'].mean():.3f}")
    print(f"  Non-robust19 mean_cos_to_other = {n19['mean_cos_to_other_centroids'].mean():.3f}")
    print(f"  Robust 6     own_doc_sim_mean  = {r6['mean_own_doc_sim_to_centroid'].mean():.3f}")
    print(f"  Non-robust19 own_doc_sim_mean  = {n19['mean_own_doc_sim_to_centroid'].mean():.3f}")

    # ============================================================
    # Summary
    # ============================================================
    n_outliers = int(outlier_mask.sum())
    summary = {
        "K": K, "N_docs": int(len(per_doc)),
        "N_outliers": n_outliers,
        "outlier_pct": round(100*n_outliers/len(per_doc), 2),
        "robust_topics": sorted(ROBUST_TOPICS),
        "specificity": {
            "robust_6_mean_unique_ratio": round(float(spec[spec['is_robust']]['unique_ratio'].mean()), 3),
            "nonrobust_19_mean_unique_ratio": round(float(spec[~spec['is_robust']]['unique_ratio'].mean()), 3),
        },
        "outlier_vs_inlier_length": {
            "outlier_mean_chars": float(outlier_profile.loc[True, "mean_char_len"]),
            "inlier_mean_chars": float(outlier_profile.loc[False, "mean_char_len"]),
            "outlier_mean_sentences": float(outlier_profile.loc[True, "mean_n_sent"]),
            "inlier_mean_sentences": float(outlier_profile.loc[False, "mean_n_sent"]),
        },
        "geometry": {
            "robust_mean_cos_to_other": round(float(r6['mean_cos_to_other_centroids'].mean()), 4),
            "nonrobust_mean_cos_to_other": round(float(n19['mean_cos_to_other_centroids'].mean()), 4),
            "robust_mean_own_doc_sim": round(float(r6['mean_own_doc_sim_to_centroid'].mean()), 4),
            "nonrobust_mean_own_doc_sim": round(float(n19['mean_own_doc_sim_to_centroid'].mean()), 4),
        },
        "elapsed_s": round(time.time()-t_total, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "="*70)
    print("Visibility quantitative — headline")
    print("="*70)
    print(f"  Outlier rate: {summary['outlier_pct']}%")
    print(f"  Specificity (unique top-10 ratio): robust {summary['specificity']['robust_6_mean_unique_ratio']} vs non-robust {summary['specificity']['nonrobust_19_mean_unique_ratio']}")
    print(f"  Outlier length: {summary['outlier_vs_inlier_length']['outlier_mean_chars']:.0f} chars (vs inlier {summary['outlier_vs_inlier_length']['inlier_mean_chars']:.0f})")
    print(f"  Outlier sentences: {summary['outlier_vs_inlier_length']['outlier_mean_sentences']:.1f} (vs inlier {summary['outlier_vs_inlier_length']['inlier_mean_sentences']:.1f})")
    print(f"  Geometry — mean_cos_to_other: robust {summary['geometry']['robust_mean_cos_to_other']} vs non-robust {summary['geometry']['nonrobust_mean_cos_to_other']}")
    print(f"  Geometry — own_doc_sim: robust {summary['geometry']['robust_mean_own_doc_sim']} vs non-robust {summary['geometry']['nonrobust_mean_own_doc_sim']}")
    print(f"  Elapsed: {summary['elapsed_s']}s")
    print("="*70)


if __name__ == "__main__":
    main()
