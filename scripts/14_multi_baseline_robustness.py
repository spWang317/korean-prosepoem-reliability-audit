"""
14_multi_baseline_robustness.py

Cross-baseline robustness check across four non-degenerate UMAP seeds.

Approach:
  1. Run BERTopic at each of the four non-degenerate seeds (100, 256, 777, 2048)
     with identical baseline parameters otherwise.
  2. For each baseline, extract its K non-outlier topics with top-10 keywords.
  3. Compute Jaccard top-10 keyword overlap between all topic pairs across
     baselines.
  4. Hungarian matching: align each baseline's topics to the others.
  5. Identify motifs that appear (Jaccard ≥ 0.5) in:
     - all 4 baselines (cross-baseline robust)
     - exactly 3
     - exactly 2
     - only 1 (baseline-specific)

The seed=100 anchor's six audit-robust motifs (4/4 axes passed):
  T0 sea, T1 flower, T3 animal, T7 rain, T11 meta-poem, T16 cat

Outputs:
  results/multi_baseline/seed_{S}/topic_info.csv     each baseline's topics
  results/multi_baseline/cross_matching.json         pairwise Jaccard + Hungarian
  results/multi_baseline/cross_robust_set.csv        motif × n_baselines_present
  results/multi_baseline/summary.json                headline
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
FILT = ROOT / "data" / "filtered"
TOK = ROOT / "data" / "tokenized"
BASE = ROOT / "results" / "baseline"
OUT = ROOT / "results" / "multi_baseline"
OUT.mkdir(parents=True, exist_ok=True)

NON_DEGEN_SEEDS = [100, 256, 777, 2048]

# Baseline parameters (everything else identical)
BASE_UMAP = dict(n_neighbors=15, n_components=5, min_dist=0.0,
                 metric="cosine")
BASE_HDBSCAN = dict(min_cluster_size=30, min_samples=10,
                    metric="euclidean", cluster_selection_method="eom")
BASE_VEC = dict(token_pattern=r"\S+", min_df=2, max_df=0.95)

# Six audit-robust motifs at the seed=100 anchor
ROBUST_6_BASELINE100 = [0, 1, 7, 11, 14, 16]


def parse_topwords(s):
    if not isinstance(s, str) or not s:
        return []
    return [w.split("(")[0].strip() for w in s.split(",") if "(" in w]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def run_bertopic(embeddings, docs_pos, seed):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    umap_params = dict(BASE_UMAP, random_state=seed)
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=UMAP(**umap_params),
        hdbscan_model=HDBSCAN(**BASE_HDBSCAN, prediction_data=True),
        vectorizer_model=CountVectorizer(**BASE_VEC),
        language="multilingual",
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(docs_pos, embeddings)
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].reset_index(drop=True)
    kw_cols = [", ".join(f"{w}({s:.3f})" for w, s in (topic_model.get_topic(tid) or []))
               for tid in info["Topic"]]
    info["TopWords"] = kw_cols
    return info, np.asarray(topics)


def main():
    t_total = time.time()
    print("Loading data...")
    embeddings = np.load(BASE / "embeddings.npy")
    content = pd.read_parquet(TOK / "work_content_tokens.parquet")
    works = pd.read_parquet(FILT / "works_filtered.parquet")
    m = works.merge(content, on="work_id", how="inner")
    docs_pos = m["content_tokens"].fillna("").tolist()
    print(f"  N={len(docs_pos)} docs")

    # ============================================================
    # 1. Run BERTopic at each baseline seed
    # ============================================================
    baselines = {}
    for seed in NON_DEGEN_SEEDS:
        print(f"\n=== Baseline seed={seed} ===")
        seed_dir = OUT / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        cached = seed_dir / "topic_info.csv"
        if cached.exists():
            print(f"  reusing cached: {cached}")
            info = pd.read_csv(cached)
        else:
            t0 = time.time()
            info, topics_arr = run_bertopic(embeddings, docs_pos, seed)
            info.to_csv(cached, index=False)
            # Also save topic_per_doc
            pd.DataFrame({"work_id": m["work_id"].tolist(),
                         "topic": topics_arr}).to_csv(
                seed_dir / "topic_per_doc.csv", index=False)
            outlier_pct = float((topics_arr == -1).mean()) * 100
            print(f"  K={len(info)}, outlier={outlier_pct:.1f}%, "
                  f"elapsed {time.time()-t0:.1f}s")
        topwords = [parse_topwords(s) for s in info["TopWords"]]
        baselines[seed] = {
            "info": info,
            "K": len(info),
            "topwords": topwords,
        }

    print(f"\n  K per baseline: " +
          ", ".join(f"seed={s}: K={baselines[s]['K']}" for s in NON_DEGEN_SEEDS))

    # ============================================================
    # 2. Cross-baseline Hungarian matching (Jaccard top-10)
    # ============================================================
    print("\n=== Cross-baseline Jaccard matching ===")
    from scipy.optimize import linear_sum_assignment

    pairs_summary = []
    cross_matched_lists = {}  # for each baseline seed, list of (topic_idx, matched_in_other_baseline)

    for i, s1 in enumerate(NON_DEGEN_SEEDS):
        for j, s2 in enumerate(NON_DEGEN_SEEDS):
            if i >= j:
                continue
            kw1 = baselines[s1]["topwords"]
            kw2 = baselines[s2]["topwords"]
            K1, K2 = len(kw1), len(kw2)
            J = np.zeros((K1, K2))
            for a in range(K1):
                for b in range(K2):
                    J[a, b] = jaccard(kw1[a], kw2[b])
            row_ind, col_ind = linear_sum_assignment(-J)
            matched_j = J[row_ind, col_ind]
            n_matched_05 = int((matched_j >= 0.5).sum())
            mean_j = float(matched_j[matched_j >= 0.5].mean()) if n_matched_05 else 0.0
            pairs_summary.append({
                "seed_a": s1, "seed_b": s2,
                "K_a": K1, "K_b": K2,
                "n_matched_jaccard_ge_0.5": n_matched_05,
                "match_rate": round(n_matched_05 / min(K1, K2), 3),
                "mean_jaccard_at_matched": round(mean_j, 3),
            })
            print(f"  seed {s1} vs seed {s2}: "
                  f"matched {n_matched_05}/{min(K1,K2)} (rate {n_matched_05/min(K1,K2):.2f}), "
                  f"mean J(matched)={mean_j:.2f}")

    (OUT / "cross_matching.json").write_text(
        json.dumps(pairs_summary, indent=2, ensure_ascii=False))

    # ============================================================
    # 3. Cross-baseline robust set: motifs present in ≥k baselines
    # ============================================================
    print("\n=== Cross-baseline robust motifs ===")
    # Build a topic-instance pool: every (seed, topic_idx) is a node.
    # For each topic-instance, count how many *other* baselines have a Jaccard≥0.5
    # match in their best-Hungarian-assigned partner.
    # Simpler: pick seed=100 as anchor (since it's our paper baseline) and
    # measure how many other baselines have a matching topic for each of its 25.

    anchor_seed = 100
    anchor_kw = baselines[anchor_seed]["topwords"]
    anchor_info = baselines[anchor_seed]["info"]
    K_anchor = len(anchor_kw)
    rows = []
    for a in range(K_anchor):
        anchor_words = anchor_kw[a]
        anchor_tid_local = int(anchor_info.iloc[a]["Topic"])
        # number of other baselines that have a topic with Jaccard ≥ 0.5 to anchor[a]
        present_in_seeds = [anchor_seed]  # by definition
        best_matches = {}
        for s_other in NON_DEGEN_SEEDS:
            if s_other == anchor_seed:
                continue
            other_kw = baselines[s_other]["topwords"]
            best_j = max(jaccard(anchor_words, ok) for ok in other_kw) if other_kw else 0.0
            best_matches[s_other] = round(best_j, 3)
            if best_j >= 0.5:
                present_in_seeds.append(s_other)
        rows.append({
            "anchor_topic_id_in_seed100": anchor_tid_local,
            "in_robust_6_seed100": anchor_tid_local in ROBUST_6_BASELINE100,
            "topwords": ", ".join(anchor_words),
            "n_baselines_present": len(present_in_seeds),
            "present_in": ", ".join(map(str, present_in_seeds)),
            **{f"best_jaccard_vs_seed{s}": best_matches.get(s)
               for s in NON_DEGEN_SEEDS if s != anchor_seed},
        })
    cross_df = pd.DataFrame(rows).sort_values(
        ["n_baselines_present", "in_robust_6_seed100"], ascending=[False, False])
    cross_df.to_csv(OUT / "cross_robust_set.csv", index=False)

    # Distribution
    dist = cross_df["n_baselines_present"].value_counts().sort_index(ascending=False)
    print(f"\n  Cross-baseline presence distribution (anchor seed=100):")
    for n_b, count in dist.items():
        print(f"    in {n_b}/4 baselines: {count} topics")

    # How many of our robust 6 are cross-baseline robust?
    r6_cross = cross_df[cross_df["in_robust_6_seed100"]]
    print(f"\n  Of our 6 paper-robust motifs (seed=100 8-layer audit):")
    for _, r in r6_cross.iterrows():
        kw_short = r["topwords"][:50]
        print(f"    T{int(r['anchor_topic_id_in_seed100'])}: "
              f"present in {int(r['n_baselines_present'])}/4 baselines  | {kw_short}")

    # ============================================================
    # 4. Summary
    # ============================================================
    n_all_4 = int((cross_df["n_baselines_present"] == 4).sum())
    n_at_least_3 = int((cross_df["n_baselines_present"] >= 3).sum())
    n_at_least_2 = int((cross_df["n_baselines_present"] >= 2).sum())
    summary = {
        "non_degenerate_seeds": NON_DEGEN_SEEDS,
        "K_per_baseline": {s: baselines[s]["K"] for s in NON_DEGEN_SEEDS},
        "K_mean": round(float(np.mean([baselines[s]["K"] for s in NON_DEGEN_SEEDS])), 1),
        "K_std": round(float(np.std([baselines[s]["K"] for s in NON_DEGEN_SEEDS])), 1),
        "anchor_seed": anchor_seed,
        "n_anchor_topics": K_anchor,
        "anchor_distribution": dist.to_dict(),
        "n_in_all_4_baselines": n_all_4,
        "n_in_at_least_3_baselines": n_at_least_3,
        "n_in_at_least_2_baselines": n_at_least_2,
        "of_paper_robust_6": {
            "n_in_all_4": int(((cross_df["in_robust_6_seed100"]) &
                               (cross_df["n_baselines_present"] == 4)).sum()),
            "n_in_at_least_3": int(((cross_df["in_robust_6_seed100"]) &
                                     (cross_df["n_baselines_present"] >= 3)).sum()),
            "n_in_at_least_2": int(((cross_df["in_robust_6_seed100"]) &
                                     (cross_df["n_baselines_present"] >= 2)).sum()),
            "individual": r6_cross[["anchor_topic_id_in_seed100",
                                     "n_baselines_present"]].to_dict("records"),
        },
        "elapsed_s": round(time.time() - t_total, 1),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("Multi-baseline robustness — headline")
    print("=" * 70)
    print(f"  K per baseline: {summary['K_per_baseline']}")
    print(f"  K mean ± std:   {summary['K_mean']} ± {summary['K_std']}")
    print(f"\n  Anchor (seed=100): K={K_anchor}")
    print(f"    Topics present in all 4 baselines:      {n_all_4}")
    print(f"    Topics present in at least 3 baselines: {n_at_least_3}")
    print(f"    Topics present in at least 2 baselines: {n_at_least_2}")
    print(f"\n  Of our paper-robust 6 (8-layer audit on seed=100):")
    print(f"    In all 4 baselines:      {summary['of_paper_robust_6']['n_in_all_4']}/6")
    print(f"    In at least 3 baselines: {summary['of_paper_robust_6']['n_in_at_least_3']}/6")
    print(f"    In at least 2 baselines: {summary['of_paper_robust_6']['n_in_at_least_2']}/6")
    print(f"\n  Elapsed: {summary['elapsed_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
