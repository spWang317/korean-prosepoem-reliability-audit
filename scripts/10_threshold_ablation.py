"""
10_threshold_ablation.py

Direct test of the "conservative-threshold inflates α" hypothesis:
HDBSCAN with large min_cluster_size is conservative (high outlier rate),
keeping only "confident" docs in clusters → potentially inflated Cronbach α.

We re-run R3 (10 seeds, baseline params) at three min_cluster_size settings,
each measuring per-baseline-topic α exactly as in `06_psychometric_reliability.py`.

  baseline: min_cluster_size=30  (outlier ~49%)  [already run; reload from 06]
  smaller:  min_cluster_size=15  (expected: outlier 25-35%)
  smallest: min_cluster_size=5   (expected: outlier <15%, many small clusters)

Expectation under "conservative inflation" hypothesis:
  - As min_cluster_size shrinks, outlier rate drops, more "marginal" docs
    enter clusters, internal consistency falls → α drops.
  - If observed: hypothesis confirmed; our α 0.72 is partly inflated.
  - If NOT observed (α stable): membership reliability is genuine.

Note: changing min_cluster_size changes the number of clusters and the
identities of clusters, so we re-match each run to the baseline (mcs=30)
topic centroids using doc-set Jaccard. The baseline reference is fixed
at the original baseline (mcs=30, seed=100).

Outputs:
  results/ablation_mcs/runs_per_mcs.json       per-mcs per-seed n_topics, outlier
  results/ablation_mcs/alpha_per_mcs.csv       per-topic α at each mcs
  results/ablation_mcs/summary.json
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
FILT_DIR = ROOT / "data" / "filtered"
TOK_DIR = ROOT / "data" / "tokenized"
BASE = ROOT / "results" / "baseline"
PSYCH = ROOT / "results" / "psychometric"
OUT = ROOT / "results" / "ablation_mcs"
OUT.mkdir(parents=True, exist_ok=True)

BASE_UMAP = dict(n_neighbors=15, n_components=5, min_dist=0.0,
                 metric="cosine", random_state=100)
BASE_VEC = dict(token_pattern=r"\S+", min_df=2, max_df=0.95)
SEEDS = [0, 7, 42, 100, 123, 256, 511, 777, 1024, 2048]

# Test settings — note baseline (mcs=30) results are loaded from 06 output.
MCS_SETTINGS = [15, 5]


def cronbach_alpha(X):
    X = np.asarray(X, dtype=float)
    K, N = X.shape
    if K < 2 or N < 2:
        return float("nan")
    item_var_sum = X.var(axis=1, ddof=1).sum()
    subject_total_var = X.sum(axis=0).var(ddof=1)
    if subject_total_var <= 0:
        return float("nan")
    return float((K / (K - 1)) * (1 - item_var_sum / subject_total_var))


def jaccard_set(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def run_bertopic(embeddings, docs_pos, umap_p, hdbscan_p, vec_p):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=UMAP(**umap_p),
        hdbscan_model=HDBSCAN(**hdbscan_p, prediction_data=True),
        vectorizer_model=CountVectorizer(**vec_p),
        language="multilingual",
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False,
    )
    topics_arr, _ = topic_model.fit_transform(docs_pos, embeddings)
    topics_arr = np.asarray(topics_arr)
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].reset_index(drop=True)
    return topics_arr, info["Topic"].tolist()


def main():
    t_total = time.time()
    print("Loading data...")
    embeddings = np.load(BASE / "embeddings.npy")
    works = pd.read_parquet(FILT_DIR / "works_filtered.parquet")
    content = pd.read_parquet(TOK_DIR / "work_content_tokens.parquet")
    m = works.merge(content, on="work_id", how="inner")
    docs_pos = m["content_tokens"].fillna("").tolist()
    N = embeddings.shape[0]

    # Baseline reference: docs in each baseline topic
    base_per_doc = pd.read_csv(BASE / "topic_per_doc.csv")
    base_topic_info = pd.read_csv(BASE / "topic_info.csv")
    base_topic_info = base_topic_info[base_topic_info["Topic"] != -1].reset_index(drop=True)
    K = len(base_topic_info)
    baseline_doc_sets = []
    for k in range(K):
        tid = int(base_topic_info.iloc[k]["Topic"])
        ids = np.where(base_per_doc["topic"].values == tid)[0]
        baseline_doc_sets.append(set(ids.tolist()))
    print(f"  N={N}, K_baseline={K} topics")

    # --------------------------------------------------- Reload baseline α (mcs=30)
    print("\nReloading baseline (mcs=30) α from 06...")
    baseline_rel = pd.read_csv(PSYCH / "reliability_per_topic.csv")
    print(f"  mcs=30 α mean: {baseline_rel['cronbach_alpha'].dropna().mean():.4f}")

    # --------------------------------------------------- Run ablation
    all_results = {30: {
        "alpha_mean": float(baseline_rel["cronbach_alpha"].dropna().mean()),
        "alpha_median": float(baseline_rel["cronbach_alpha"].dropna().median()),
        "alpha_max": float(baseline_rel["cronbach_alpha"].dropna().max()),
        "alpha_min": float(baseline_rel["cronbach_alpha"].dropna().min()),
        "n_topics_alpha_ge_0.7": int((baseline_rel["cronbach_alpha"] >= 0.7).sum()),
        "n_topics_alpha_ge_0.8": int((baseline_rel["cronbach_alpha"] >= 0.8).sum()),
        "outlier_rate_mean": None,  # placeholder
    }}
    per_topic_at_mcs = {30: baseline_rel["cronbach_alpha"].tolist()}

    for mcs in MCS_SETTINGS:
        print(f"\n=== Ablation: min_cluster_size = {mcs} ===")
        # min_samples default in BERTopic = same as mcs. Use 10 if mcs<10.
        min_samples = max(5, min(mcs, 10))
        hdbscan_p = dict(min_cluster_size=mcs, min_samples=min_samples,
                         metric="euclidean", cluster_selection_method="eom")

        # For each of 10 seeds: run BERTopic, match each cluster to baseline by
        # doc-set Jaccard (Hungarian), build indicator
        indicator = np.zeros((K, len(SEEDS), N), dtype=np.int8)
        run_records = []
        for r_idx, seed in enumerate(tqdm(SEEDS, desc=f"  seeds (mcs={mcs})")):
            umap_p = deepcopy(BASE_UMAP)
            umap_p["random_state"] = seed
            try:
                topics_arr, topic_ids = run_bertopic(
                    embeddings, docs_pos, umap_p, hdbscan_p, BASE_VEC
                )
            except Exception as e:
                print(f"    seed {seed} failed: {e}")
                continue
            outlier_pct = float((topics_arr == -1).mean()) * 100
            n_topics_this = len(topic_ids)
            # Build doc-sets per perturbed cluster
            perturbed_doc_sets = []
            for tid in topic_ids:
                ids = np.where(topics_arr == tid)[0]
                perturbed_doc_sets.append(set(ids.tolist()))
            # Hungarian on doc-set Jaccard
            from scipy.optimize import linear_sum_assignment
            J = np.zeros((K, max(K, n_topics_this)))
            for i in range(K):
                for j in range(n_topics_this):
                    J[i, j] = jaccard_set(baseline_doc_sets[i], perturbed_doc_sets[j])
            # Allow rect matrix (n_topics_this may differ from K)
            if n_topics_this == 0:
                run_records.append({"seed": seed, "n_topics": 0,
                                    "outlier_pct": outlier_pct})
                continue
            row_ind, col_ind = linear_sum_assignment(-J[:, :n_topics_this])
            for i, j in zip(row_ind, col_ind):
                if J[i, j] >= 0.1:   # match threshold for indicator inclusion
                    ids = list(perturbed_doc_sets[j])
                    indicator[i, r_idx, ids] = 1
            run_records.append({
                "seed": seed,
                "n_topics": n_topics_this,
                "outlier_pct": round(outlier_pct, 2),
                "n_matched_with_baseline_dj_ge_0.1": int(sum(J[i, col_ind[idx]] >= 0.1
                                                              for idx, i in enumerate(row_ind))),
            })

        # Compute α per topic
        alphas = []
        for k in range(K):
            alpha = cronbach_alpha(indicator[k]) if (indicator[k].sum(axis=1) > 0).sum() >= 2 else float("nan")
            alphas.append(alpha)
        alphas_arr = np.array(alphas)
        valid = alphas_arr[~np.isnan(alphas_arr)]

        mean_outlier = float(np.mean([r["outlier_pct"] for r in run_records if "outlier_pct" in r]))
        all_results[mcs] = {
            "alpha_mean": float(valid.mean()) if len(valid) else float("nan"),
            "alpha_median": float(np.median(valid)) if len(valid) else float("nan"),
            "alpha_max": float(valid.max()) if len(valid) else float("nan"),
            "alpha_min": float(valid.min()) if len(valid) else float("nan"),
            "n_topics_alpha_ge_0.7": int((valid >= 0.7).sum()),
            "n_topics_alpha_ge_0.8": int((valid >= 0.8).sum()),
            "outlier_rate_mean": round(mean_outlier, 2),
            "mean_n_topics_per_run": round(float(np.mean([r["n_topics"] for r in run_records])), 1),
        }
        per_topic_at_mcs[mcs] = alphas

        print(f"  mcs={mcs}: outlier {mean_outlier:.1f}% (vs baseline ~49%)")
        print(f"  mcs={mcs}: α mean {all_results[mcs]['alpha_mean']:.4f} "
              f"(vs baseline {all_results[30]['alpha_mean']:.4f})")
        (OUT / f"runs_mcs_{mcs}.json").write_text(
            json.dumps(run_records, indent=2, ensure_ascii=False))

    # --------------------------------------------------- Per-topic table
    rows = []
    for k in range(K):
        row = {
            "topic_id": int(base_topic_info.iloc[k]["Topic"]),
            "size": int(base_topic_info.iloc[k]["Count"]),
            "topwords": base_topic_info.iloc[k]["TopWords"][:80],
        }
        for mcs in [30] + MCS_SETTINGS:
            row[f"alpha_mcs{mcs}"] = (round(per_topic_at_mcs[mcs][k], 4)
                                       if not np.isnan(per_topic_at_mcs[mcs][k])
                                       else None)
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / "alpha_per_mcs.csv", index=False)

    # --------------------------------------------------- Headline & verdict
    print("\n" + "=" * 80)
    print("Conservative threshold ablation — headline")
    print("=" * 80)
    print(f"  {'mcs':>6s}  {'outlier%':>10s}  {'mean_topics':>12s}  "
          f"{'α_mean':>8s}  {'α≥0.7':>8s}  {'α≥0.8':>8s}")
    for mcs in [30] + MCS_SETTINGS:
        r = all_results[mcs]
        outlier = r.get("outlier_rate_mean", "?")
        mean_topics = r.get("mean_n_topics_per_run", K)
        print(f"  {mcs:>6d}  {outlier!s:>10s}  {mean_topics!s:>12s}  "
              f"{r['alpha_mean']:>8.4f}  {r['n_topics_alpha_ge_0.7']:>8d}  "
              f"{r['n_topics_alpha_ge_0.8']:>8d}")

    # Verdict
    a30 = all_results[30]["alpha_mean"]
    a15 = all_results[15]["alpha_mean"]
    a5 = all_results[5]["alpha_mean"]
    drop_15 = a30 - a15
    drop_5 = a30 - a5
    if drop_5 >= 0.15 and drop_15 >= 0.05:
        verdict = "CONFIRMED — α drops substantially as outlier rate decreases (conservative threshold inflates α)"
    elif drop_5 < 0.05 and drop_15 < 0.05:
        verdict = "REJECTED — α stable across thresholds (membership reliability is genuine)"
    else:
        verdict = "AMBIGUOUS — partial drop; investigate further"

    summary = {
        "alpha_by_mcs": {str(mcs): all_results[mcs] for mcs in [30, 15, 5]},
        "alpha_drop_mcs30_to_mcs15": round(drop_15, 4),
        "alpha_drop_mcs30_to_mcs5": round(drop_5, 4),
        "verdict": verdict,
        "elapsed_s": round(time.time() - t_total, 1),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\n  α drop (mcs 30→15): {drop_15:+.4f}")
    print(f"  α drop (mcs 30→5):  {drop_5:+.4f}")
    print(f"\n  VERDICT: {verdict}")
    print(f"  Elapsed: {summary['elapsed_s']}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
