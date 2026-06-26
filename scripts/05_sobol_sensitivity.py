"""
05_sobol_sensitivity.py

Continuous-parameter variance-based sensitivity analysis [Sobol' 1990, 2001;
Saltelli 2008]. Critique of OAT-only (our R1) addressed: Sobol' captures
*interactions* between parameters, which marginal sweeps cannot.

Sample design: Saltelli's extension of Sobol' sequence (SALib).
  N = 8  → N(2D+2) = 96 BERTopic runs for 5-dim parameter space.

Parameters (continuous bounds, rounded to int where appropriate):
  - umap_n_neighbors          [10, 50]     int
  - umap_n_components         [3, 10]      int
  - umap_min_dist             [0.0, 0.3]   float
  - hdbscan_min_cluster_size  [20, 80]     int
  - hdbscan_min_samples       [5, 15]      int

Output metric (per run): mean topic persistence vs. baseline (Jaccard ≥ 0.5
top-10 keyword match, averaged over baseline's 25 non-outlier topics).

Sobol' indices:
  - S1_i (first-order): fraction of output variance from parameter i alone
  - ST_i (total-order):  fraction of output variance involving parameter i
                         (including interactions)
  - ST_i - S1_i ≈ interaction contribution of parameter i

Outputs:
  results/sobol/samples.npy           Saltelli design
  results/sobol/run_outputs.csv       per-run output metric + per-topic match
  results/sobol/sobol_indices.json    S1, ST, confidence intervals
  results/sobol/summary.json          headline (interaction-dominated? OAT-misleading?)
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
BASELINE_DIR = ROOT / "results" / "baseline"
OUT_DIR = ROOT / "results" / "sobol"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Locked baseline params for non-Sobol' axes
BASE_UMAP_STATIC = dict(metric="cosine", random_state=100)
BASE_HDBSCAN_STATIC = dict(metric="euclidean", cluster_selection_method="eom")
BASE_VEC = dict(token_pattern=r"\S+", min_df=2, max_df=0.95)


# Sobol' problem definition
PROBLEM = {
    "num_vars": 5,
    "names": [
        "umap_n_neighbors",
        "umap_n_components",
        "umap_min_dist",
        "hdbscan_min_cluster_size",
        "hdbscan_min_samples",
    ],
    "bounds": [
        [10, 50],     # n_neighbors (int)
        [3, 10],      # n_components (int)
        [0.0, 0.3],   # min_dist (float)
        [20, 80],     # min_cluster_size (int)
        [5, 15],      # min_samples (int)
    ],
}
INT_PARAMS = {"umap_n_neighbors", "umap_n_components",
              "hdbscan_min_cluster_size", "hdbscan_min_samples"}

SOBOL_N = 8  # Saltelli base sample size → N*(2D+2) = 96 total runs


# ---------------------------------------------------------------------- helpers
def parse_topwords(s):
    if not isinstance(s, str) or not s:
        return []
    return [w.split("(")[0].strip() for w in s.split(",") if "(" in w]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def load_baseline_topics():
    info = pd.read_csv(BASELINE_DIR / "topic_info.csv")
    info = info[info["Topic"] != -1].reset_index(drop=True)
    return [parse_topwords(s) for s in info["TopWords"]]


def topic_persistence_vs_baseline(perturbed_kws, baseline_topics, min_jaccard=0.5):
    """For each baseline topic, 1 if a perturbed topic matches at Jaccard ≥ τ."""
    if not perturbed_kws or not baseline_topics:
        return np.zeros(len(baseline_topics), dtype=int)
    K = len(baseline_topics)
    L = len(perturbed_kws)
    J = np.zeros((K, L))
    for i, ba in enumerate(baseline_topics):
        for j, pe in enumerate(perturbed_kws):
            J[i, j] = jaccard(ba, pe)
    return (J.max(axis=1) >= min_jaccard).astype(int)


def run_bertopic_one(embeddings, docs_pos, umap_params, hdbscan_params):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=UMAP(**umap_params),
        hdbscan_model=HDBSCAN(**hdbscan_params, prediction_data=True),
        vectorizer_model=CountVectorizer(**BASE_VEC),
        language="multilingual",
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False,
    )
    try:
        topics, _ = topic_model.fit_transform(docs_pos, embeddings)
    except Exception as e:
        return [], 0.0, str(e)
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].reset_index(drop=True)
    topic_keywords = []
    for tid in info["Topic"]:
        words = topic_model.get_topic(tid) or []
        topic_keywords.append([w for w, _ in words])
    topics_arr = np.asarray(topics)
    outlier_pct = float((topics_arr == -1).mean()) * 100 if topics_arr.size else 0.0
    return topic_keywords, outlier_pct, None


def sample_to_params(sample_row):
    """Convert one Saltelli sample row into BERTopic params."""
    vals = {}
    for name, v in zip(PROBLEM["names"], sample_row):
        if name in INT_PARAMS:
            vals[name] = int(round(v))
        else:
            vals[name] = float(v)
    umap_params = dict(BASE_UMAP_STATIC)
    umap_params["n_neighbors"] = vals["umap_n_neighbors"]
    umap_params["n_components"] = vals["umap_n_components"]
    umap_params["min_dist"] = vals["umap_min_dist"]
    hdbscan_params = dict(BASE_HDBSCAN_STATIC)
    hdbscan_params["min_cluster_size"] = vals["hdbscan_min_cluster_size"]
    hdbscan_params["min_samples"] = vals["hdbscan_min_samples"]
    # min_samples must be ≤ min_cluster_size for HDBSCAN to be sensible
    if hdbscan_params["min_samples"] > hdbscan_params["min_cluster_size"]:
        hdbscan_params["min_samples"] = hdbscan_params["min_cluster_size"]
    return umap_params, hdbscan_params, vals


# ---------------------------------------------------------------------- main
def main():
    t_total = time.time()
    print("Loading data...")
    embeddings = np.load(BASELINE_DIR / "embeddings.npy")
    works = pd.read_parquet(FILT_DIR / "works_filtered.parquet")
    content = pd.read_parquet(TOK_DIR / "work_content_tokens.parquet")
    m = works.merge(content, on="work_id", how="inner")
    docs_pos = m["content_tokens"].fillna("").tolist()
    baseline_topics = load_baseline_topics()
    K = len(baseline_topics)
    print(f"  Baseline: {K} non-outlier topics; corpus: {embeddings.shape[0]} works")

    # --------------------------------------------------------- 1. Saltelli sample
    from SALib.sample import saltelli
    samples = saltelli.sample(PROBLEM, SOBOL_N, calc_second_order=False)
    n_runs = samples.shape[0]
    print(f"  Saltelli design: {n_runs} runs (N={SOBOL_N}, D={PROBLEM['num_vars']})")
    np.save(OUT_DIR / "samples.npy", samples)

    # --------------------------------------------------------- 2. Run BERTopic at each point
    per_run = []
    per_run_match = np.zeros((n_runs, K), dtype=int)
    Y = np.zeros(n_runs, dtype=float)  # output: mean persistence per run

    for i in tqdm(range(n_runs), desc="Sobol' runs"):
        umap_p, hdbscan_p, vals = sample_to_params(samples[i])
        perturbed_kws, outlier_pct, err = run_bertopic_one(
            embeddings, docs_pos, umap_p, hdbscan_p
        )
        match = topic_persistence_vs_baseline(perturbed_kws, baseline_topics)
        per_run_match[i] = match
        Y[i] = float(match.mean())
        per_run.append({
            "run_idx": i,
            **vals,
            "n_topics": len(perturbed_kws),
            "outlier_pct": round(outlier_pct, 2),
            "matched_baseline": int(match.sum()),
            "mean_persistence": float(match.mean()),
            "error": err,
        })

    per_run_df = pd.DataFrame(per_run)
    per_run_df.to_csv(OUT_DIR / "run_outputs.csv", index=False)

    # --------------------------------------------------------- 3. Sobol' indices
    from SALib.analyze import sobol
    print("\nComputing Sobol' indices...")
    Si = sobol.analyze(PROBLEM, Y, calc_second_order=False,
                       print_to_console=False)
    sobol_out = {
        "param_names": PROBLEM["names"],
        "S1": Si["S1"].tolist(),
        "S1_conf": Si["S1_conf"].tolist(),
        "ST": Si["ST"].tolist(),
        "ST_conf": Si["ST_conf"].tolist(),
        "interaction_share_ST_minus_S1": (Si["ST"] - Si["S1"]).tolist(),
    }
    (OUT_DIR / "sobol_indices.json").write_text(
        json.dumps(sobol_out, indent=2, ensure_ascii=False))

    # --------------------------------------------------------- 4. Summary
    Y_mean = float(Y.mean())
    Y_std = float(Y.std())
    interaction_dominated = bool((Si["ST"] - Si["S1"]).max() > 0.10)
    summary = {
        "n_runs": int(n_runs),
        "Y_mean_persistence": round(Y_mean, 4),
        "Y_std_persistence": round(Y_std, 4),
        "Y_min": round(float(Y.min()), 4),
        "Y_max": round(float(Y.max()), 4),
        "n_zero_persistence_runs": int((Y < 1e-9).sum()),
        "n_high_persistence_runs": int((Y >= 0.5).sum()),
        "highest_S1_param": PROBLEM["names"][int(np.argmax(Si["S1"]))],
        "highest_ST_param": PROBLEM["names"][int(np.argmax(Si["ST"]))],
        "max_interaction_share": round(float((Si["ST"] - Si["S1"]).max()), 4),
        "interaction_dominated_flag": interaction_dominated,
        "elapsed_s": round(time.time() - t_total, 1),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("Sobol' sensitivity — headline")
    print("=" * 70)
    print(f"  N runs:                    {n_runs}")
    print(f"  Mean persistence Y:        {Y_mean:.3f} ± {Y_std:.3f}")
    print(f"  Range [Y_min, Y_max]:      [{Y.min():.3f}, {Y.max():.3f}]")
    print(f"  Runs with Y = 0:           {(Y < 1e-9).sum()}/{n_runs}")
    print(f"  Runs with Y ≥ 0.5:         {(Y >= 0.5).sum()}/{n_runs}")
    print(f"\n  Sobol' first-order (S1) and total (ST) indices:")
    print(f"  {'param':<32s} {'S1':>8s} {'ST':>8s} {'ST-S1':>8s}")
    for name, s1, st in zip(PROBLEM["names"], Si["S1"], Si["ST"]):
        flag = " ←" if (st - s1) > 0.10 else ""
        print(f"  {name:<32s} {s1:>8.3f} {st:>8.3f} {st-s1:>8.3f}{flag}")
    print(f"\n  Interaction-dominated?     {interaction_dominated}")
    print(f"  Elapsed:                   {summary['elapsed_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
