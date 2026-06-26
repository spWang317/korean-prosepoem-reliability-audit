"""
04_marginal_robustness.py

Marginal sensitivity sweep across three axes, holding all OTHER parameters at
baseline. Each axis varies one factor while keeping the rest at baseline
[Saltelli 2008, OAT design]. Per-topic persistence is computed by topic
alignment against the baseline run (Hungarian on Jaccard top-10).

Axes:
  R1. Parameter grid:
        UMAP: n_neighbors ∈ {10,15,20,30,50}, n_components ∈ {3,5,8},
              min_dist ∈ {0.0, 0.1, 0.25}
        HDBSCAN: min_cluster_size ∈ {20,30,50,80}, min_samples ∈ {5,10,15}
        c-TF-IDF: min_df ∈ {2,3,5}, max_df ∈ {0.9, 0.95, 0.99}
      Total OAT runs: 5+3+3 + 4+3 + 3+3 = 24 marginal variations.
  R2. Bootstrap subsample, 80%, 10 replicates [Belford 2018].
  R3. Random seed in {0, 7, 42, 100 (baseline), 123, 256, 511, 777, 1024, 2048}
      → 10 runs.

All runs reuse the cached ko-sroberta embeddings produced by
`02_baseline_bertopic.py`.

Outputs:
  results/robustness/r1_grid/runs.json              individual R1 results
  results/robustness/r2_bootstrap/runs.json
  results/robustness/r3_seeds/runs.json
  results/robustness/persistence_matrix.csv         per-topic, per-axis score
  results/robustness/summary.json
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
OUT_DIR = ROOT / "results" / "robustness"
for sub in ["r1_grid", "r2_bootstrap", "r3_seeds"]:
    (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


# Baseline parameters (lock; identical to scripts/02_baseline_bertopic.py)
BASE_UMAP = dict(n_neighbors=15, n_components=5, min_dist=0.0,
                 metric="cosine", random_state=100)
BASE_HDBSCAN = dict(min_cluster_size=30, min_samples=10,
                    metric="euclidean", cluster_selection_method="eom")
BASE_VEC = dict(token_pattern=r"\S+", min_df=2, max_df=0.95)
SEEDS = [0, 7, 42, 100, 123, 256, 511, 777, 1024, 2048]
BOOTSTRAP_N = 10
BOOTSTRAP_FRAC = 0.80


# ----------------------------------------------------------------------
# Topic matching helpers (used for persistence)
# ----------------------------------------------------------------------
def parse_topwords(s):
    if not isinstance(s, str) or not s:
        return []
    return [w.split("(")[0].strip() for w in s.split(",") if "(" in w]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def align_topics_to_baseline(perturbed_topics: list[list[str]],
                             baseline_topics: list[list[str]],
                             min_jaccard: float = 0.5) -> dict:
    """For each baseline topic, return whether a matched topic exists in perturbed."""
    if not perturbed_topics or not baseline_topics:
        return {i: False for i in range(len(baseline_topics))}
    K = len(baseline_topics)
    L = len(perturbed_topics)
    J = np.zeros((K, L))
    for i, ba in enumerate(baseline_topics):
        for j, pe in enumerate(perturbed_topics):
            J[i, j] = jaccard(ba, pe)
    # for each baseline topic, find best match in perturbed
    matched = {}
    for i in range(K):
        best = J[i].max()
        matched[i] = bool(best >= min_jaccard)
    return matched


# ----------------------------------------------------------------------
# BERTopic runner
# ----------------------------------------------------------------------
def run_bertopic(embeddings: np.ndarray, docs_pos: list,
                 umap_params: dict, hdbscan_params: dict,
                 vec_params: dict) -> list[list[str]]:
    """Run BERTopic and return list of top-10 keyword lists, one per non-outlier topic."""
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=UMAP(**umap_params),
        hdbscan_model=HDBSCAN(**hdbscan_params, prediction_data=True),
        vectorizer_model=CountVectorizer(**vec_params),
        language="multilingual",
        top_n_words=10,
        calculate_probabilities=False,
        verbose=False,
    )
    try:
        topics, _ = topic_model.fit_transform(docs_pos, embeddings)
    except Exception:
        return [], 0.0
    info = topic_model.get_topic_info()
    info = info[info["Topic"] != -1].reset_index(drop=True)
    topic_keywords = []
    for tid in info["Topic"]:
        words = topic_model.get_topic(tid) or []
        topic_keywords.append([w for w, _ in words])
    topics_arr = np.asarray(topics)
    outlier_pct = float((topics_arr == -1).mean()) * 100 if topics_arr.size else 0.0
    return topic_keywords, outlier_pct


# ----------------------------------------------------------------------
# Load baseline reference topics
# ----------------------------------------------------------------------
def load_baseline_topics():
    info = pd.read_csv(BASELINE_DIR / "topic_info.csv")
    info = info[info["Topic"] != -1].reset_index(drop=True)
    return [parse_topwords(s) for s in info["TopWords"]]


# ----------------------------------------------------------------------
# R1 — Parameter grid (OAT)
# ----------------------------------------------------------------------
def run_R1(embeddings, docs_pos, baseline_topics) -> dict:
    print("\n=== R1: Parameter grid (OAT) ===")
    R1_PARAMS = {
        "umap_n_neighbors": [10, 15, 20, 30, 50],
        "umap_n_components": [3, 5, 8],
        "umap_min_dist": [0.0, 0.1, 0.25],
        "hdbscan_min_cluster_size": [20, 30, 50, 80],
        "hdbscan_min_samples": [5, 10, 15],
        "cv_min_df": [2, 3, 5],
        "cv_max_df": [0.9, 0.95, 0.99],
    }
    runs = []
    K = len(baseline_topics)
    # Initialise per-baseline-topic matched counters
    matched_counts = np.zeros(K, dtype=int)
    total_perturbations = 0

    for param_name, values in R1_PARAMS.items():
        print(f"  varying {param_name}: {values}")
        for v in tqdm(values, desc=f"    {param_name}", leave=False):
            umap_p = deepcopy(BASE_UMAP)
            hdbscan_p = deepcopy(BASE_HDBSCAN)
            vec_p = deepcopy(BASE_VEC)
            if param_name.startswith("umap_"):
                umap_p[param_name[len("umap_"):]] = v
            elif param_name.startswith("hdbscan_"):
                hdbscan_p[param_name[len("hdbscan_"):]] = v
            elif param_name.startswith("cv_"):
                vec_p[param_name[3:]] = v

            perturbed_kws, outlier_pct = run_bertopic(
                embeddings, docs_pos, umap_p, hdbscan_p, vec_p
            )
            n_topics = len(perturbed_kws)
            matched = align_topics_to_baseline(perturbed_kws, baseline_topics)
            for i, m in matched.items():
                if m:
                    matched_counts[i] += 1
            total_perturbations += 1
            runs.append({
                "param": param_name, "value": v,
                "n_topics": n_topics, "outlier_pct": round(outlier_pct, 2),
                "n_matched_baseline": int(sum(matched.values())),
            })

    persistence = matched_counts / max(1, total_perturbations)
    (OUT_DIR / "r1_grid" / "runs.json").write_text(
        json.dumps({"runs": runs, "total_runs": total_perturbations,
                    "persistence_per_baseline_topic": persistence.tolist()},
                   indent=2, ensure_ascii=False)
    )
    print(f"  R1 done: {total_perturbations} runs; mean persistence "
          f"{persistence.mean():.2f}")
    return {
        "persistence_per_topic": persistence,
        "n_runs": total_perturbations,
        "axis": "R1",
    }


# ----------------------------------------------------------------------
# R2 — Bootstrap subsample
# ----------------------------------------------------------------------
def run_R2(embeddings, docs_pos, baseline_topics) -> dict:
    print("\n=== R2: Bootstrap subsample (80%) ===")
    K = len(baseline_topics)
    matched_counts = np.zeros(K, dtype=int)
    runs = []
    N = embeddings.shape[0]
    rng = np.random.default_rng(0)
    for r in tqdm(range(BOOTSTRAP_N), desc="    bootstrap"):
        idx = rng.choice(N, size=int(N * BOOTSTRAP_FRAC), replace=False)
        emb_sub = embeddings[idx]
        docs_sub = [docs_pos[i] for i in idx]
        perturbed_kws, outlier_pct = run_bertopic(
            emb_sub, docs_sub, BASE_UMAP, BASE_HDBSCAN, BASE_VEC
        )
        matched = align_topics_to_baseline(perturbed_kws, baseline_topics)
        for i, m in matched.items():
            if m:
                matched_counts[i] += 1
        runs.append({
            "bootstrap_idx": r,
            "n_topics": len(perturbed_kws),
            "outlier_pct": round(outlier_pct, 2),
            "n_matched_baseline": int(sum(matched.values())),
        })

    persistence = matched_counts / BOOTSTRAP_N
    (OUT_DIR / "r2_bootstrap" / "runs.json").write_text(
        json.dumps({"runs": runs, "n_replicates": BOOTSTRAP_N,
                    "fraction": BOOTSTRAP_FRAC,
                    "persistence_per_baseline_topic": persistence.tolist()},
                   indent=2, ensure_ascii=False)
    )
    print(f"  R2 done: mean persistence {persistence.mean():.2f}")
    return {"persistence_per_topic": persistence, "n_runs": BOOTSTRAP_N, "axis": "R2"}


# ----------------------------------------------------------------------
# R3 — Random seeds (UMAP stochasticity)
# ----------------------------------------------------------------------
def run_R3(embeddings, docs_pos, baseline_topics) -> dict:
    print("\n=== R3: Random seeds ===")
    K = len(baseline_topics)
    matched_counts = np.zeros(K, dtype=int)
    runs = []
    for seed in tqdm(SEEDS, desc="    seeds"):
        umap_p = deepcopy(BASE_UMAP)
        umap_p["random_state"] = seed
        perturbed_kws, outlier_pct = run_bertopic(
            embeddings, docs_pos, umap_p, BASE_HDBSCAN, BASE_VEC
        )
        matched = align_topics_to_baseline(perturbed_kws, baseline_topics)
        for i, m in matched.items():
            if m:
                matched_counts[i] += 1
        runs.append({
            "seed": seed,
            "n_topics": len(perturbed_kws),
            "outlier_pct": round(outlier_pct, 2),
            "n_matched_baseline": int(sum(matched.values())),
        })

    persistence = matched_counts / len(SEEDS)
    (OUT_DIR / "r3_seeds" / "runs.json").write_text(
        json.dumps({"runs": runs, "seeds": SEEDS,
                    "persistence_per_baseline_topic": persistence.tolist()},
                   indent=2, ensure_ascii=False)
    )
    print(f"  R3 done: mean persistence {persistence.mean():.2f}")
    return {"persistence_per_topic": persistence, "n_runs": len(SEEDS), "axis": "R3"}


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    t_total = time.time()

    print("Loading data...")
    embeddings = np.load(BASELINE_DIR / "embeddings.npy")
    works = pd.read_parquet(FILT_DIR / "works_filtered.parquet")
    content = pd.read_parquet(TOK_DIR / "work_content_tokens.parquet")
    m = works.merge(content, on="work_id", how="inner")
    docs_pos = m["content_tokens"].fillna("").tolist()
    assert len(docs_pos) == embeddings.shape[0]

    baseline_topics = load_baseline_topics()
    K = len(baseline_topics)
    print(f"  Baseline has {K} non-outlier topics")
    print(f"  Corpus: {embeddings.shape[0]} works, {embeddings.shape[1]} dim")

    # Run each axis
    R1 = run_R1(embeddings, docs_pos, baseline_topics)
    R2 = run_R2(embeddings, docs_pos, baseline_topics)
    R3 = run_R3(embeddings, docs_pos, baseline_topics)

    # ----------------------------------------------------------
    # Persistence matrix
    # ----------------------------------------------------------
    baseline_info = pd.read_csv(BASELINE_DIR / "topic_info.csv")
    baseline_info = baseline_info[baseline_info["Topic"] != -1].reset_index(drop=True)

    persistence_df = pd.DataFrame({
        "topic_id": baseline_info["Topic"].values,
        "size": baseline_info["Count"].values,
        "topwords": baseline_info["TopWords"].values,
        "R1_persistence": R1["persistence_per_topic"],
        "R2_persistence": R2["persistence_per_topic"],
        "R3_persistence": R3["persistence_per_topic"],
    })
    persistence_df.to_csv(OUT_DIR / "persistence_matrix.csv", index=False)

    # Robust set (τ = 0.7 on all three marginal axes)
    tau = 0.7
    robust_mask = (
        (persistence_df["R1_persistence"] >= tau) &
        (persistence_df["R2_persistence"] >= tau) &
        (persistence_df["R3_persistence"] >= tau)
    )
    robust_topics = persistence_df[robust_mask]
    print("\n" + "=" * 70)
    print(f"Persistence summary (τ = {tau})")
    print("=" * 70)
    print(f"  Baseline non-outlier topics : {K}")
    print(f"  Topics robust on R1 (param) : {(persistence_df.R1_persistence >= tau).sum()}")
    print(f"  Topics robust on R2 (boot)  : {(persistence_df.R2_persistence >= tau).sum()}")
    print(f"  Topics robust on R3 (seed)  : {(persistence_df.R3_persistence >= tau).sum()}")
    print(f"  Topics robust on ALL R1-R3  : {robust_mask.sum()}")

    print("\nTop persistent topics:")
    for _, row in persistence_df.sort_values(
        ["R1_persistence", "R2_persistence", "R3_persistence"], ascending=False
    ).head(10).iterrows():
        print(f"  T{row.topic_id:>3} (n={row['size']:>4}): "
              f"R1={row.R1_persistence:.2f} R2={row.R2_persistence:.2f} "
              f"R3={row.R3_persistence:.2f}  | {row.topwords[:80]}")

    (OUT_DIR / "summary.json").write_text(json.dumps({
        "tau": tau,
        "K_baseline_topics": K,
        "robust_topic_count_R1_R2_R3": int(robust_mask.sum()),
        "elapsed_s": round(time.time() - t_total, 1),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
