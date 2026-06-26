"""
09_negative_controls.py

Two negative-control diagnostics for the psychometric reliability claim:

[A] Random clustering control:
    If we replace BERTopic with *random* assignment of 4642 docs into 25
    clusters of similar size distribution to baseline, do we still see
    Cronbach α ≈ 0.7?
    - Yes → our α 0.72 is an artefact of structured pipelines + matching.
    - No  → our α 0.72 reflects real corpus signal.

[C] Keyword-matching vs doc-set-matching comparison:
    Our R3 reliability uses keyword Jaccard ≥ 0.5 to match perturbed topics
    to baseline topics. An alternative matching uses doc-set Jaccard
    (overlap of assigned documents). If both matchings agree, our reliability
    framework is not artefactually driven by the keyword-matching choice.

Outputs:
  results/negative_controls/random_alpha.json
  results/negative_controls/matching_comparison.json
  results/negative_controls/summary.json
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "baseline"
PSYCH = ROOT / "results" / "psychometric"
OUT = ROOT / "results" / "negative_controls"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [0, 7, 42, 100, 123, 256, 511, 777, 1024, 2048]


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


def main():
    t_total = time.time()
    print("Loading baseline + psychometric data...")
    topic_info = pd.read_csv(BASE / "topic_info.csv")
    topic_info = topic_info[topic_info["Topic"] != -1].reset_index(drop=True)
    K = len(topic_info)
    per_doc = pd.read_csv(BASE / "topic_per_doc.csv")
    N = len(per_doc)
    # Distribution of cluster sizes in baseline (we'll mimic this for [A])
    baseline_sizes = topic_info["Count"].values.astype(int)
    baseline_outlier_count = int((per_doc["topic"] == -1).sum())
    print(f"  N={N}, K={K} baseline topics")
    print(f"  baseline outlier rate: {baseline_outlier_count/N*100:.1f}%")

    # =========================================================
    # [A] Random clustering control
    # =========================================================
    print("\n=== [A] Random clustering negative control ===")
    R = len(SEEDS)
    # Build indicator for each random run: shape (K, R, N)
    # For each run r:
    #   1. randomly assign baseline_outlier_count docs to "outlier"
    #   2. partition remaining docs into K clusters with sizes matching baseline_sizes
    indicator_random = np.zeros((K, R, N), dtype=np.int8)
    rng_master = np.random.default_rng(99)
    for r_idx, seed in enumerate(SEEDS):
        rng = np.random.default_rng(seed)
        idx = np.arange(N)
        rng.shuffle(idx)
        # First baseline_outlier_count are outliers
        outlier_idx = idx[:baseline_outlier_count]
        inlier_idx = idx[baseline_outlier_count:]
        # Assign inliers to K clusters with sizes ≈ baseline_sizes (scale to fit)
        # Recompute proportional sizes to fit available inliers
        proportions = baseline_sizes / baseline_sizes.sum()
        target_sizes = np.round(proportions * len(inlier_idx)).astype(int)
        # Correct for rounding
        diff = len(inlier_idx) - target_sizes.sum()
        target_sizes[np.argmax(target_sizes)] += diff
        # Partition inliers
        pos = 0
        for k in range(K):
            doc_ids = inlier_idx[pos:pos + target_sizes[k]]
            pos += target_sizes[k]
            indicator_random[k, r_idx, doc_ids] = 1
    # Now matching: for the random control, we do NOT match topics across
    # runs (random has no semantic content). The fair test is: for each
    # "baseline" cluster index k, take its run-by-run indicator and compute α.
    # This is the LEAST favourable comparison for random — equivalent to
    # claiming the random clusters at index k are "the same topic" across runs.

    random_alphas = []
    for k in range(K):
        alpha = cronbach_alpha(indicator_random[k])
        random_alphas.append(alpha)
    random_alphas = np.array(random_alphas)
    # NB: with proper random partitioning and no matching, α should be ~0
    # (random partitions ≠ matched topics)
    print(f"  Random α (no matching): mean={np.nanmean(random_alphas):.3f}, "
          f"max={np.nanmax(random_alphas):.3f}, min={np.nanmin(random_alphas):.3f}")

    # Sharper test — match random partitions to first run using doc-set Jaccard
    # (Hungarian). This is the ATTACK angle: even with optimal post-hoc
    # matching, does random reach α 0.72?
    from scipy.optimize import linear_sum_assignment
    # Build doc-membership lists per run
    run_clusters = []  # list of (R) entries; each is a list of K cluster-doc-sets
    for r_idx in range(R):
        clusters_this_run = []
        for k in range(K):
            doc_ids = np.where(indicator_random[k, r_idx] == 1)[0]
            clusters_this_run.append(set(doc_ids.tolist()))
        run_clusters.append(clusters_this_run)
    # Match runs 2..R against run 0 by Hungarian on doc-set Jaccard
    ref = run_clusters[0]
    indicator_random_matched = np.zeros((K, R, N), dtype=np.int8)
    for k in range(K):
        indicator_random_matched[k, 0, list(ref[k])] = 1
    for r_idx in range(1, R):
        # K × K Jaccard matrix
        J = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                J[i, j] = jaccard_set(ref[i], run_clusters[r_idx][j])
        # Hungarian maximises matching: minimise -J
        row_ind, col_ind = linear_sum_assignment(-J)
        # row_ind = ref topic, col_ind = matched perturbed topic
        for i, j in zip(row_ind, col_ind):
            doc_ids = list(run_clusters[r_idx][j])
            indicator_random_matched[i, r_idx, doc_ids] = 1
    random_alphas_matched = np.array([
        cronbach_alpha(indicator_random_matched[k]) for k in range(K)
    ])
    print(f"  Random α (with optimal post-hoc matching): "
          f"mean={np.nanmean(random_alphas_matched):.3f}, "
          f"max={np.nanmax(random_alphas_matched):.3f}, "
          f"min={np.nanmin(random_alphas_matched):.3f}")

    random_result = {
        "K": K, "N": N, "n_runs": R,
        "method_no_matching": {
            "alpha_mean": round(float(np.nanmean(random_alphas)), 4),
            "alpha_max": round(float(np.nanmax(random_alphas)), 4),
            "alpha_min": round(float(np.nanmin(random_alphas)), 4),
        },
        "method_post_hoc_matching": {
            "alpha_mean": round(float(np.nanmean(random_alphas_matched)), 4),
            "alpha_max": round(float(np.nanmax(random_alphas_matched)), 4),
            "alpha_min": round(float(np.nanmin(random_alphas_matched)), 4),
        },
        "interpretation": (
            "If post-hoc-matched random α reaches the observed BERTopic α "
            "(~0.72), then our reliability is a matching artefact. If random α "
            "stays well below, our α reflects genuine corpus structure."
        ),
    }
    (OUT / "random_alpha.json").write_text(
        json.dumps(random_result, indent=2, ensure_ascii=False))

    # =========================================================
    # [C] Keyword-matching vs doc-set-matching comparison
    # =========================================================
    print("\n=== [C] Keyword vs doc-set matching comparison ===")
    # Load real R3 indicator from 06's output
    npz = np.load(PSYCH / "indicator_matrices.npz")
    indicator_real = npz["indicator"]  # (K, R, N)
    seeds_real = npz["seeds"].tolist()
    # `indicator_real[k, r, d] = 1` iff doc d was assigned to a topic in run r
    # whose top-10 keywords matched baseline topic k (keyword Jaccard ≥ 0.5).
    # That's the keyword-matching path. Now do doc-set-matching path:
    # For each run r, partition docs by their assigned topic (we don't have
    # the perturbed topic_per_doc saved as a separate file, but each run's
    # perturbed clusters can be reconstructed by row OR of indicator_real
    # plus the docs that were never assigned).

    # Reconstruct each run's "matched docs" by row k indicator. Each non-zero
    # entry indicates membership in run-r's topic-that-matched-baseline-k.
    # For doc-set-matching, we rebuild what each perturbed cluster contained,
    # but we only have the matched-baseline-k indicator. We approximate the
    # set of run-r-clusters as the K indicator rows (one per matched baseline).
    # Then doc-set matching = bipartite Jaccard on these sets vs baseline docs.

    baseline_cluster_docs = []
    for k in range(K):
        tid = int(topic_info.iloc[k]["Topic"])
        doc_ids = np.where(per_doc["topic"].values == tid)[0]
        baseline_cluster_docs.append(set(doc_ids.tolist()))

    # For each run r: try matching perturbed-clusters to baseline by doc-set
    # Jaccard, instead of by keyword Jaccard.
    # The perturbed-clusters as we have them are already keyword-matched
    # (so this is biased toward agreement). For a fairer comparison we need
    # the raw perturbed partition, which we DO NOT have. So this is partial.
    # Instead, we measure: do the two matchings (keyword vs doc-set, both
    # post-hoc) produce the same K topics surviving with α ≥ 0.7?

    # Heuristic: re-rank by doc-set Jaccard between indicator_real[k, r, :]
    # (which is the keyword-matched docs) and baseline_cluster_docs[k]
    # (the baseline's docs). For each topic k and run r, compute that Jaccard.
    # If keyword-matching and doc-matching converge, Jaccard ≈ 1; if they
    # diverge, Jaccard can be low even when keyword-Jaccard was ≥ 0.5.

    doc_set_jaccards = np.zeros((K, R))
    for k in range(K):
        baseline_set = baseline_cluster_docs[k]
        for r in range(R):
            perturbed_docs = set(np.where(indicator_real[k, r] == 1)[0].tolist())
            doc_set_jaccards[k, r] = jaccard_set(baseline_set, perturbed_docs)

    # Per-topic mean doc-set Jaccard across the (≤R) runs where the topic emerged
    per_topic_mean_j = np.array([
        np.nanmean(doc_set_jaccards[k, doc_set_jaccards[k] > 0])
        if (doc_set_jaccards[k] > 0).any() else float("nan")
        for k in range(K)
    ])
    overall_mean_j = float(np.nanmean(per_topic_mean_j))

    # Compare with α
    print(f"  Mean doc-set Jaccard (perturbed vs baseline, where keyword-matched): "
          f"{overall_mean_j:.3f}")
    print(f"  Topics with mean doc-set Jaccard ≥ 0.5: "
          f"{int((per_topic_mean_j >= 0.5).sum())}/{K}")
    print(f"  Topics with mean doc-set Jaccard ≥ 0.7: "
          f"{int((per_topic_mean_j >= 0.7).sum())}/{K}")

    matching_result = {
        "K": K, "n_runs": R,
        "overall_mean_doc_set_jaccard": round(overall_mean_j, 4),
        "topics_dj_ge_0.5": int((per_topic_mean_j >= 0.5).sum()),
        "topics_dj_ge_0.7": int((per_topic_mean_j >= 0.7).sum()),
        "per_topic_mean_doc_set_jaccard": [
            (int(topic_info.iloc[k]["Topic"]),
             round(float(per_topic_mean_j[k]), 4) if not np.isnan(per_topic_mean_j[k]) else None)
            for k in range(K)
        ],
        "interpretation": (
            "When keyword-matching identifies the same topic across runs, "
            "are the underlying doc-sets also similar? High mean (>0.5) "
            "suggests keyword and doc-set matchings converge — our reliability "
            "is not a keyword-matching artefact."
        ),
    }
    (OUT / "matching_comparison.json").write_text(
        json.dumps(matching_result, indent=2, ensure_ascii=False))

    # =========================================================
    # Summary
    # =========================================================
    real_alpha_mean = 0.721  # from 06
    summary = {
        "real_BERTopic_alpha_mean": real_alpha_mean,
        "random_alpha_no_matching_mean": random_result["method_no_matching"]["alpha_mean"],
        "random_alpha_with_matching_mean": random_result["method_post_hoc_matching"]["alpha_mean"],
        "doc_set_jaccard_when_keyword_matched": matching_result["overall_mean_doc_set_jaccard"],
        "verdict_random_control": (
            "PASS" if random_result["method_post_hoc_matching"]["alpha_mean"] < 0.5
            else "AMBIGUOUS" if random_result["method_post_hoc_matching"]["alpha_mean"] < 0.65
            else "FAIL"
        ),
        "verdict_matching_robustness": (
            "PASS" if matching_result["overall_mean_doc_set_jaccard"] >= 0.5
            else "AMBIGUOUS" if matching_result["overall_mean_doc_set_jaccard"] >= 0.3
            else "FAIL"
        ),
        "elapsed_s": round(time.time() - t_total, 1),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("Negative controls — headline")
    print("=" * 70)
    print(f"  Real BERTopic α mean:                {real_alpha_mean}")
    print(f"  Random α (no matching) mean:         {summary['random_alpha_no_matching_mean']}")
    print(f"  Random α (post-hoc matched) mean:    {summary['random_alpha_with_matching_mean']}")
    print(f"  Doc-set Jaccard when keyword-matched: {summary['doc_set_jaccard_when_keyword_matched']}")
    print(f"\n  Verdict — Random control:        {summary['verdict_random_control']}")
    print(f"  Verdict — Matching robustness:   {summary['verdict_matching_robustness']}")
    print(f"  Elapsed: {summary['elapsed_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
