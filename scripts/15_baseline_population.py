"""
15_baseline_population.py

Pre-registered population-style baseline configuration protocol.

Protocol (pre-registered):
  1. Random seed pool: 100 seeds drawn deterministically from {1..10000}
     via numpy default_rng(seed=42).
  2. Run BERTopic at each seed (cached embedding; baseline params).
  3. Non-degenerate filter: K ≥ 15 AND outlier_pct ∈ [30, 70].
     (Justified by Finding A: degenerate seeds give K≤5; baseline at 25 with 49%)
  4. All seeds passing the filter become baseline pool.
  5. Cross-baseline matching: each seed=100 topic's presence across the pool
     (Jaccard ≥ 0.5 top-10).
  6. Report: % of pool in which each seed=100 topic re-appears.

Outputs:
  results/baseline_population/seed_pool.csv      every seed: K, outlier, accepted?
  results/baseline_population/all_topwords.json  per-accepted-seed topic keyword lists
  results/baseline_population/cross_presence.csv each seed=100 topic × % of pool
  results/baseline_population/summary.json
"""
from __future__ import annotations

import json
import time
import warnings
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
FILT = ROOT / "data" / "filtered"
TOK = ROOT / "data" / "tokenized"
BASE = ROOT / "results" / "baseline"
OUT = ROOT / "results" / "baseline_population"
OUT.mkdir(parents=True, exist_ok=True)


# Pre-registered protocol parameters
N_SEED_POOL = 100
SEED_RANGE = (1, 10000)
RNG_SEED = 42            # deterministic for re-runs
K_MIN = 15               # non-degenerate filter
OUTLIER_MIN_PCT = 30.0   # non-degenerate filter
OUTLIER_MAX_PCT = 70.0   # non-degenerate filter
JACCARD_THRESHOLD = 0.5  # matching threshold

# Baseline (identical to seed=100 baseline)
BASE_UMAP = dict(n_neighbors=15, n_components=5, min_dist=0.0,
                 metric="cosine")
BASE_HDBSCAN = dict(min_cluster_size=30, min_samples=10,
                    metric="euclidean", cluster_selection_method="eom")
BASE_VEC = dict(token_pattern=r"\S+", min_df=2, max_df=0.95)


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
    try:
        topics, _ = topic_model.fit_transform(docs_pos, embeddings)
    except Exception:
        return None, None
    topics = np.asarray(topics)
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].reset_index(drop=True)
    K = len(info)
    outlier_pct = float((topics == -1).mean()) * 100
    topwords = []
    for tid in info["Topic"]:
        words = topic_model.get_topic(tid) or []
        topwords.append([w for w, _ in words])
    return {"K": K, "outlier_pct": outlier_pct, "topwords": topwords}, topics


def main():
    t_total = time.time()
    print("Loading data...")
    embeddings = np.load(BASE / "embeddings.npy")
    content = pd.read_parquet(TOK / "work_content_tokens.parquet")
    works = pd.read_parquet(FILT / "works_filtered.parquet")
    m = works.merge(content, on="work_id", how="inner")
    docs_pos = m["content_tokens"].fillna("").tolist()
    print(f"  N={len(docs_pos)} docs")

    # Pre-registered seed sample
    rng = np.random.default_rng(RNG_SEED)
    seed_pool = sorted(rng.choice(np.arange(SEED_RANGE[0], SEED_RANGE[1] + 1),
                                  size=N_SEED_POOL, replace=False).tolist())
    print(f"  Seed pool: {N_SEED_POOL} seeds drawn from {SEED_RANGE} "
          f"(rng_seed={RNG_SEED})")
    # Always include seed=100 (paper baseline) for direct comparison
    if 100 not in seed_pool:
        seed_pool[0] = 100
        seed_pool.sort()
        print("  Added seed=100 to pool (paper baseline)")

    # Run BERTopic at each seed
    print(f"\nRunning BERTopic at each seed (~5s/run × {len(seed_pool)})...")
    pool_records = []
    accepted_seeds = []
    accepted_topwords = {}
    for seed in tqdm(seed_pool, desc="  BERTopic"):
        out, topics = run_bertopic(embeddings, docs_pos, seed)
        if out is None:
            pool_records.append({"seed": int(seed), "K": 0, "outlier_pct": 100.0,
                                "accepted": False, "reason": "exception"})
            continue
        accept = (out["K"] >= K_MIN and
                  OUTLIER_MIN_PCT <= out["outlier_pct"] <= OUTLIER_MAX_PCT)
        pool_records.append({
            "seed": int(seed),
            "K": int(out["K"]),
            "outlier_pct": round(out["outlier_pct"], 2),
            "accepted": bool(accept),
            "reason": "" if accept else (
                f"K<{K_MIN}" if out["K"] < K_MIN
                else f"outlier outside [{OUTLIER_MIN_PCT},{OUTLIER_MAX_PCT}]"),
        })
        if accept:
            accepted_seeds.append(int(seed))
            accepted_topwords[int(seed)] = out["topwords"]

    pool_df = pd.DataFrame(pool_records)
    pool_df.to_csv(OUT / "seed_pool.csv", index=False)
    n_accepted = len(accepted_seeds)
    print(f"\n  Accepted (non-degenerate) seeds: {n_accepted}/{len(seed_pool)} "
          f"({100*n_accepted/len(seed_pool):.0f}%)")

    if n_accepted < 4:
        print(f"  WARNING: only {n_accepted} non-degenerate seeds — "
              "filter may be too strict")

    # Save all topword sets
    json_safe = {str(s): topws for s, topws in accepted_topwords.items()}
    (OUT / "all_topwords.json").write_text(
        json.dumps(json_safe, indent=2, ensure_ascii=False))

    # K and outlier distribution among accepted
    accepted_df = pool_df[pool_df["accepted"]]
    print(f"  K     across accepted: mean={accepted_df['K'].mean():.1f}, "
          f"std={accepted_df['K'].std():.1f}, range=[{accepted_df['K'].min()}, {accepted_df['K'].max()}]")
    print(f"  Outlier% across accepted: mean={accepted_df['outlier_pct'].mean():.1f}, "
          f"std={accepted_df['outlier_pct'].std():.1f}")

    # Cross-baseline presence of seed=100's topics
    print(f"\n=== Cross-baseline presence (anchor seed=100, pool size {n_accepted}) ===")
    if 100 not in accepted_topwords:
        # seed=100 baseline didn't pass filter? warn but proceed anyway from cached
        from pathlib import Path
        topic_info_path = BASE / "topic_info.csv"
        ti = pd.read_csv(topic_info_path)
        ti = ti[ti["Topic"] != -1].reset_index(drop=True)
        anchor_topwords = [parse_topwords(s) for s in ti["TopWords"]]
        print("  using cached seed=100 baseline (not in pool)")
    else:
        anchor_topwords = accepted_topwords[100]
        ti = pd.read_csv(BASE / "topic_info.csv")
        ti = ti[ti["Topic"] != -1].reset_index(drop=True)

    K_anchor = len(anchor_topwords)

    # For each anchor topic, count fraction of pool that has a Jaccard >= 0.5 match
    rows = []
    for a in range(K_anchor):
        anchor_w = anchor_topwords[a]
        if not anchor_w:
            continue
        anchor_tid = int(ti.iloc[a]["Topic"])
        n_present = 0
        n_pool_excl_self = 0
        for s_other in accepted_seeds:
            if s_other == 100:
                continue
            other_topwords = accepted_topwords[s_other]
            n_pool_excl_self += 1
            best_j = max((jaccard(anchor_w, ot) for ot in other_topwords),
                         default=0.0)
            if best_j >= JACCARD_THRESHOLD:
                n_present += 1
        share = n_present / max(1, n_pool_excl_self)
        rows.append({
            "anchor_topic_id": anchor_tid,
            "topwords": ", ".join(anchor_w),
            "n_present_in_pool": n_present,
            "n_pool_excl_self": n_pool_excl_self,
            "presence_share": round(share, 3),
        })
    cross_df = pd.DataFrame(rows).sort_values("presence_share", ascending=False)
    cross_df.to_csv(OUT / "cross_presence.csv", index=False)

    # Of paper-robust 6: T0, T1, T3, T7, T11, T16
    paper_robust_6 = {0, 1, 3, 7, 11, 16}
    cross_df["in_paper_robust_6"] = cross_df["anchor_topic_id"].isin(paper_robust_6)
    print("\n  All 25 seed=100 topics ranked by cross-baseline presence:")
    for _, r in cross_df.head(30).iterrows():
        flag = "★" if r["in_paper_robust_6"] else " "
        kw_short = r["topwords"][:60]
        print(f"    T{int(r['anchor_topic_id']):>2} {flag}  "
              f"presence={r['presence_share']:.2f} ({int(r['n_present_in_pool'])}/{int(r['n_pool_excl_self'])})  | {kw_short}")

    # Summary
    summary = {
        "protocol": {
            "n_seed_pool": N_SEED_POOL,
            "seed_range": SEED_RANGE,
            "rng_seed": RNG_SEED,
            "K_min": K_MIN,
            "outlier_min_pct": OUTLIER_MIN_PCT,
            "outlier_max_pct": OUTLIER_MAX_PCT,
            "jaccard_threshold": JACCARD_THRESHOLD,
        },
        "n_seeds_tried": len(seed_pool),
        "n_accepted_seeds": n_accepted,
        "acceptance_rate": round(n_accepted / len(seed_pool), 3),
        "K_in_accepted": {
            "mean": round(float(accepted_df["K"].mean()), 2),
            "std": round(float(accepted_df["K"].std()), 2),
            "min": int(accepted_df["K"].min()),
            "max": int(accepted_df["K"].max()),
        },
        "outlier_in_accepted": {
            "mean": round(float(accepted_df["outlier_pct"].mean()), 2),
            "std": round(float(accepted_df["outlier_pct"].std()), 2),
        },
        "anchor_seed": 100,
        "cross_presence_distribution": {
            "share_ge_0.95": int((cross_df["presence_share"] >= 0.95).sum()),
            "share_ge_0.80": int((cross_df["presence_share"] >= 0.80).sum()),
            "share_ge_0.50": int((cross_df["presence_share"] >= 0.50).sum()),
            "share_lt_0.20": int((cross_df["presence_share"] < 0.20).sum()),
        },
        "of_paper_robust_6": {
            "n_share_ge_0.95": int(((cross_df["in_paper_robust_6"]) &
                                     (cross_df["presence_share"] >= 0.95)).sum()),
            "n_share_ge_0.80": int(((cross_df["in_paper_robust_6"]) &
                                     (cross_df["presence_share"] >= 0.80)).sum()),
            "n_share_ge_0.50": int(((cross_df["in_paper_robust_6"]) &
                                     (cross_df["presence_share"] >= 0.50)).sum()),
            "details": cross_df[cross_df["in_paper_robust_6"]][
                ["anchor_topic_id", "presence_share"]].to_dict("records"),
        },
        "elapsed_s": round(time.time() - t_total, 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("Baseline population — headline")
    print("=" * 70)
    print(f"  Protocol: {N_SEED_POOL} random seeds; "
          f"non-degenerate filter K≥{K_MIN}, outlier ∈ [{OUTLIER_MIN_PCT},{OUTLIER_MAX_PCT}]")
    print(f"  Accepted: {n_accepted}/{len(seed_pool)} ({100*n_accepted/len(seed_pool):.0f}%)")
    print(f"  K   in accepted: {summary['K_in_accepted']['mean']:.1f} ± {summary['K_in_accepted']['std']:.1f}")
    print(f"  Outlier% in accepted: {summary['outlier_in_accepted']['mean']:.1f} ± {summary['outlier_in_accepted']['std']:.1f}")
    print(f"\n  Presence of paper-robust 6 across {n_accepted-1} other accepted baselines:")
    print(f"    in ≥95%: {summary['of_paper_robust_6']['n_share_ge_0.95']}/6")
    print(f"    in ≥80%: {summary['of_paper_robust_6']['n_share_ge_0.80']}/6")
    print(f"    in ≥50%: {summary['of_paper_robust_6']['n_share_ge_0.50']}/6")
    print(f"\n  Elapsed: {summary['elapsed_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
