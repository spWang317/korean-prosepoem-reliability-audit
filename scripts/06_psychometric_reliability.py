"""
06_psychometric_reliability.py

Psychometric reliability of BERTopic topics, beyond Jaccard-top-words matching.

Schroeder & Wood-Doughty (2024/2025, "Reliability of Topic Modeling", arXiv
2410.23186 / NAACL 2025) argue that standard topic-matching reliability
metrics (Jaccard top-words + cosine threshold τ=0.7) *overstate* reliability
because they ignore document-topic assignment variance: keyword sets may
match while document-membership shifts.

Operationalisation here:
  - Re-run R3 (10 random seeds; UMAP stochasticity axis) saving per-doc topic
    assignments per run.
  - For each baseline topic t, build a 10×N indicator matrix:
        I_t[r, d] = 1 iff doc d is assigned to a topic in run r that matches
                       baseline topic t (Jaccard ≥ 0.5 top-10 keywords).
  - Compute Cronbach's α across the 10 runs (rows as raters, docs as items).
  - Also report mean pairwise Cohen's κ.
  - Spearman-Brown projection: predicted reliability if number of runs were 30.

Standard Cronbach's α interpretation [Nunnally 1978]:
  α ≥ 0.9   excellent      α ∈ [0.7, 0.8)  acceptable
  α ∈ [0.8, 0.9)  good     α ∈ [0.6, 0.7)  questionable
                            α < 0.6        unreliable

Outputs:
  results/psychometric/r3_runs_with_assignments.json   per-run topic_per_doc
  results/psychometric/indicator_matrices.npz          K topics × R runs × N docs
  results/psychometric/reliability_per_topic.csv       α, κ, Spearman-Brown
  results/psychometric/summary.json                    headline
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
OUT_DIR = ROOT / "results" / "psychometric"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Baseline params and seeds (identical to 04_marginal_robustness.py R3)
BASE_UMAP = dict(n_neighbors=15, n_components=5, min_dist=0.0,
                 metric="cosine", random_state=100)
BASE_HDBSCAN = dict(min_cluster_size=30, min_samples=10,
                    metric="euclidean", cluster_selection_method="eom")
BASE_VEC = dict(token_pattern=r"\S+", min_df=2, max_df=0.95)
SEEDS = [0, 7, 42, 100, 123, 256, 511, 777, 1024, 2048]
JACCARD_MATCH = 0.5  # for topic-to-baseline matching


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


def best_match_baseline_for_each_perturbed(perturbed_kws, baseline_topics,
                                            min_jaccard=JACCARD_MATCH):
    """For each perturbed topic j, return matched baseline topic id (or -1)."""
    K = len(baseline_topics)
    L = len(perturbed_kws)
    J = np.zeros((K, L))
    for i, ba in enumerate(baseline_topics):
        for j, pe in enumerate(perturbed_kws):
            J[i, j] = jaccard(ba, pe)
    matched = []
    for j in range(L):
        col = J[:, j]
        best = col.argmax()
        if col[best] >= min_jaccard:
            matched.append(int(best))
        else:
            matched.append(-1)
    return matched  # length L


def run_bertopic_save_doc_topic(embeddings, docs_pos, umap_p, hdbscan_p, vec_p):
    """Run BERTopic and return (topic_per_doc, topic_keywords)."""
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
    topic_ids_ordered = info["Topic"].tolist()  # non-outlier topic ids (could include any int)
    topic_keywords = []
    for tid in topic_ids_ordered:
        words = topic_model.get_topic(tid) or []
        topic_keywords.append([w for w, _ in words])
    return topics_arr, topic_ids_ordered, topic_keywords


def cronbach_alpha(indicator_runs_x_docs: np.ndarray) -> float:
    """
    Cronbach's α (= KR-20 for binary data) for inter-rater reliability.

    Input shape (K_raters, N_subjects); raters = rows (runs/seeds),
    subjects = columns (documents). Each cell = binary membership.

    Classical formula:
        α = (K/(K-1)) * (1 - Σ_i σ²(item_i) / σ²(total_per_subject))
    where items are the K raters and subjects are the N docs.
        σ²(item_i)             = variance of rater i's judgments across docs
                                  (= X.var(axis=1) per row)
        σ²(total_per_subject) = variance across docs of the per-doc rater-sum
                                  (= X.sum(axis=0).var())
    """
    X = np.asarray(indicator_runs_x_docs, dtype=float)
    K, N = X.shape
    if K < 2 or N < 2:
        return float("nan")
    item_var_sum = X.var(axis=1, ddof=1).sum()      # Σ variance per rater (item)
    subject_total = X.sum(axis=0)                    # per-doc rater-sum
    subject_total_var = subject_total.var(ddof=1)
    if subject_total_var <= 0:
        return float("nan")
    return float((K / (K - 1)) * (1 - item_var_sum / subject_total_var))


def pairwise_cohens_kappa(indicator_runs_x_docs: np.ndarray) -> float:
    """Mean pairwise Cohen's κ across runs treating each doc as a rating.

    Skips run-pairs where one (or both) raters are all-zero (variance=0),
    which would otherwise produce sklearn divide-by-zero RuntimeWarnings."""
    from sklearn.metrics import cohen_kappa_score
    X = np.asarray(indicator_runs_x_docs, dtype=int)
    K = X.shape[0]
    if K < 2:
        return float("nan")
    kappas = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        for i in range(K):
            if X[i].sum() == 0 or X[i].sum() == X.shape[1]:
                continue  # constant rater → skip
            for j in range(i + 1, K):
                if X[j].sum() == 0 or X[j].sum() == X.shape[1]:
                    continue
                k = cohen_kappa_score(X[i], X[j])
                if not np.isnan(k):
                    kappas.append(float(k))
    return float(np.mean(kappas)) if kappas else float("nan")


def spearman_brown(alpha_K: float, K_current: int, K_target: int) -> float:
    """Predict reliability if number of runs (raters) were K_target."""
    if alpha_K <= 0 or alpha_K >= 1 or K_current <= 0:
        return float("nan")
    ratio = K_target / K_current
    return float((ratio * alpha_K) / (1 + (ratio - 1) * alpha_K))


# ---------------------------------------------------------------------- main
def main():
    t_total = time.time()
    print("Loading data...")
    embeddings = np.load(BASELINE_DIR / "embeddings.npy")
    works = pd.read_parquet(FILT_DIR / "works_filtered.parquet")
    content = pd.read_parquet(TOK_DIR / "work_content_tokens.parquet")
    m = works.merge(content, on="work_id", how="inner")
    docs_pos = m["content_tokens"].fillna("").tolist()
    work_ids = m["work_id"].tolist()
    N = embeddings.shape[0]
    assert len(docs_pos) == N

    baseline_topics = load_baseline_topics()
    K = len(baseline_topics)
    print(f"  Baseline: {K} topics; corpus: {N} works")

    # ------------------------------------------------------- 1. Re-run R3 with assignments
    print("\n=== Re-running R3 (10 seeds) with topic_per_doc saved ===")
    indicator = np.zeros((K, len(SEEDS), N), dtype=np.int8)
    run_records = []
    for r_idx, seed in enumerate(tqdm(SEEDS, desc="    seeds")):
        umap_p = deepcopy(BASE_UMAP)
        umap_p["random_state"] = seed
        try:
            topics_arr, topic_ids, topic_keywords = run_bertopic_save_doc_topic(
                embeddings, docs_pos, umap_p, BASE_HDBSCAN, BASE_VEC
            )
        except Exception as e:
            print(f"  seed {seed} failed: {e}")
            run_records.append({"seed": seed, "n_topics": 0, "error": str(e)})
            continue
        matched_base = best_match_baseline_for_each_perturbed(
            topic_keywords, baseline_topics, JACCARD_MATCH
        )
        # For each doc, find which baseline topic (if any) it belongs to in this run
        for j, base_id in enumerate(matched_base):
            if base_id < 0:
                continue
            tid_in_model = topic_ids[j]
            doc_mask = (topics_arr == tid_in_model)
            indicator[base_id, r_idx, :] = doc_mask.astype(np.int8)
        n_match = sum(1 for x in matched_base if x >= 0)
        run_records.append({
            "seed": seed,
            "n_topics": len(topic_keywords),
            "n_matched_baseline": n_match,
            "outlier_pct": round(float((topics_arr == -1).mean()) * 100, 2),
        })

    # save raw indicators
    np.savez_compressed(
        OUT_DIR / "indicator_matrices.npz",
        indicator=indicator,
        seeds=np.array(SEEDS),
        work_ids=np.array(work_ids),
    )
    (OUT_DIR / "r3_runs_with_assignments.json").write_text(
        json.dumps({"runs": run_records, "seeds": SEEDS,
                    "N_docs": N, "K_baseline_topics": K},
                   indent=2, ensure_ascii=False)
    )

    # ------------------------------------------------------- 2. Reliability per baseline topic
    print("\n=== Computing reliability per baseline topic ===")
    baseline_info = pd.read_csv(BASELINE_DIR / "topic_info.csv")
    baseline_info = baseline_info[baseline_info["Topic"] != -1].reset_index(drop=True)

    rows = []
    for t in range(K):
        X = indicator[t]  # shape (R, N)
        n_runs_active = int((X.sum(axis=1) > 0).sum())
        col_mean = X.mean(axis=0)  # per-doc agreement rate
        prevalence = float(X.mean())
        alpha = cronbach_alpha(X) if n_runs_active >= 2 else float("nan")
        kappa = pairwise_cohens_kappa(X) if n_runs_active >= 2 else float("nan")
        sb_30 = spearman_brown(alpha, len(SEEDS), 30) if not np.isnan(alpha) else float("nan")
        sb_1 = spearman_brown(alpha, len(SEEDS), 1) if not np.isnan(alpha) else float("nan")
        rows.append({
            "topic_id": int(baseline_info.iloc[t]["Topic"]),
            "size": int(baseline_info.iloc[t]["Count"]),
            "topwords": baseline_info.iloc[t]["TopWords"],
            "n_runs_topic_emerged": n_runs_active,
            "prevalence": round(prevalence, 4),
            "cronbach_alpha": round(alpha, 4) if not np.isnan(alpha) else None,
            "mean_pairwise_kappa": round(kappa, 4) if not np.isnan(kappa) else None,
            "spearman_brown_K30": round(sb_30, 4) if not np.isnan(sb_30) else None,
            "spearman_brown_single_run": round(sb_1, 4) if not np.isnan(sb_1) else None,
        })
    rel_df = pd.DataFrame(rows)
    rel_df.to_csv(OUT_DIR / "reliability_per_topic.csv", index=False)

    # ------------------------------------------------------- 3. Summary
    valid = rel_df.dropna(subset=["cronbach_alpha"])
    n_excellent = int((valid["cronbach_alpha"] >= 0.9).sum())
    n_good = int(((valid["cronbach_alpha"] >= 0.8) & (valid["cronbach_alpha"] < 0.9)).sum())
    n_acceptable = int(((valid["cronbach_alpha"] >= 0.7) & (valid["cronbach_alpha"] < 0.8)).sum())
    n_questionable = int(((valid["cronbach_alpha"] >= 0.6) & (valid["cronbach_alpha"] < 0.7)).sum())
    n_unreliable = int((valid["cronbach_alpha"] < 0.6).sum())

    summary = {
        "N_docs": N,
        "K_baseline_topics": K,
        "n_seeds": len(SEEDS),
        "n_topics_with_alpha": len(valid),
        "alpha_mean": round(float(valid["cronbach_alpha"].mean()) if len(valid) else float("nan"), 4),
        "alpha_max": round(float(valid["cronbach_alpha"].max()) if len(valid) else float("nan"), 4),
        "alpha_min": round(float(valid["cronbach_alpha"].min()) if len(valid) else float("nan"), 4),
        "Nunnally_bands": {
            "excellent (α ≥ 0.9)": n_excellent,
            "good (0.8–0.9)": n_good,
            "acceptable (0.7–0.8)": n_acceptable,
            "questionable (0.6–0.7)": n_questionable,
            "unreliable (< 0.6)": n_unreliable,
        },
        "topics_passing_alpha_0.7": int((valid["cronbach_alpha"] >= 0.7).sum()),
        "topics_passing_alpha_0.8": int((valid["cronbach_alpha"] >= 0.8).sum()),
        "elapsed_s": round(time.time() - t_total, 1),
    }
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("Psychometric reliability — headline")
    print("=" * 70)
    print(f"  Baseline topics:           {K}")
    print(f"  Seeds (R3 runs):           {len(SEEDS)}")
    print(f"  Cronbach α — mean:         {summary['alpha_mean']}")
    print(f"  Cronbach α — max:          {summary['alpha_max']}")
    print(f"  Cronbach α — min:          {summary['alpha_min']}")
    print(f"\n  Nunnally 1978 bands:")
    for band, n in summary["Nunnally_bands"].items():
        print(f"    {band:<28s}  {n:>3d} topics")
    print(f"\n  Topics passing α ≥ 0.7:    {summary['topics_passing_alpha_0.7']}/{K}")
    print(f"  Topics passing α ≥ 0.8:    {summary['topics_passing_alpha_0.8']}/{K}")
    print(f"\n  Elapsed:                   {summary['elapsed_s']}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
