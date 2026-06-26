"""
12_coherence_baseline.py

Standard topic coherence baselines for the locked baseline (25 topics,
seed=100) and the 6 robust motifs subset.

This is the *quality* baseline (do the topics make sense as word groups?),
which is orthogonal to the reliability audit (Findings A–J). We compute
the four canonical coherence metrics so that DSH/NLP reviewers can place
our topics within standard literature, while explicitly acknowledging the
Hoyle 2021 critique that automated coherence misaligns with human
judgement on neural topic models.

Metrics (all via Gensim's CoherenceModel [Röder et al. 2015]):
  - c_v       : Röder 2015's composite measure; best correlation with human
                topic-intrusion in their evaluation.
  - c_npmi    : Normalized PMI with sliding window (boundary measure).
  - u_mass    : UMass / Mimno 2011; BERTopic's default intrinsic metric.
  - c_uci    : Lau 2014; PMI with epsilon smoothing.

Reference corpus: the corpus *itself* (intrinsic; standard for topic
modelling baselines), tokenised at the POS-filtered content-lemma level
to match the BERTopic vocabulary.

Output rows:
  - per-topic coherence (4 metrics) for the 25 baseline topics
  - mean coherence over the 25
  - mean coherence over the 6 robust motifs (T0, T1, T3, T7, T11, T16)
  - random-topic baseline (sample 10 random word groups of size 10 from
    the corpus vocabulary; mean coherence) — sanity check

References to cite in §4 reporting:
  - Röder, Both, Hinneburg 2015 (c_v, c_npmi, c_uci formulation)
  - Mimno et al. 2011 (UMass)
  - Lau et al. 2014 (intrusion + c_uci)
  - Hoyle et al. 2021 (NeurIPS) "Is Automated Topic Model Evaluation Broken?"
    — automated coherence ↔ human judgement misalignment critique
  - Doogan & Buntine 2021 (NAACL) "Topic Model or Topic Twaddle?"

Outputs:
  results/coherence/per_topic_coherence.csv
  results/coherence/summary.json
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
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
TOK_DIR = ROOT / "data" / "tokenized"
BASE = ROOT / "results" / "baseline"
OUT = ROOT / "results" / "coherence"
OUT.mkdir(parents=True, exist_ok=True)

# Six audit-robust motifs (4/4 axes passed)
ROBUST_TOPICS = [0, 1, 7, 11, 14, 16]

# Coherence metrics
METRICS = ["c_v", "c_npmi", "u_mass", "c_uci"]


def parse_topwords(s):
    if not isinstance(s, str) or not s:
        return []
    return [w.split("(")[0].strip() for w in s.split(",") if "(" in w]


def main():
    t_total = time.time()
    print("Loading data...")
    content = pd.read_parquet(TOK_DIR / "work_content_tokens.parquet")
    topic_info = pd.read_csv(BASE / "topic_info.csv")
    topic_info = topic_info[topic_info["Topic"] != -1].reset_index(drop=True)
    K = len(topic_info)
    print(f"  N docs = {len(content):,}; K baseline topics = {K}")

    # Tokenize content: each doc → list of tokens
    print("\nTokenising for coherence reference corpus...")
    texts = [doc.split() for doc in content["content_tokens"].fillna("").tolist()]
    texts = [t for t in texts if t]  # drop empty
    print(f"  {len(texts):,} non-empty docs; vocab size estimating...")

    # Build dictionary for gensim
    from gensim.corpora.dictionary import Dictionary
    dictionary = Dictionary(texts)
    print(f"  vocab size (before filter): {len(dictionary):,}")
    # Filter ≥ 2 docs (matches BERTopic min_df=2)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    print(f"  vocab size (after min_df=2, max_df=0.95): {len(dictionary):,}")

    # Get topic word lists
    topic_words = {}
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        words = parse_topwords(row["TopWords"])
        # Keep only words that exist in the filtered dictionary
        words = [w for w in words if w in dictionary.token2id]
        topic_words[tid] = words

    # Compute coherence per metric
    from gensim.models.coherencemodel import CoherenceModel
    rows = []
    metric_results = {m: {} for m in METRICS}

    for metric in METRICS:
        print(f"\n=== Metric: {metric} ===")
        # CoherenceModel needs topics as list of lists
        topic_word_lists = [topic_words[int(row["Topic"])]
                            for _, row in topic_info.iterrows()]
        # Filter out topics with < 2 valid words
        valid_idx = [i for i, w in enumerate(topic_word_lists) if len(w) >= 2]
        valid_topics = [topic_word_lists[i] for i in valid_idx]
        valid_tids = [int(topic_info.iloc[i]["Topic"]) for i in valid_idx]
        if not valid_topics:
            continue
        try:
            cm = CoherenceModel(
                topics=valid_topics,
                texts=texts,
                dictionary=dictionary,
                coherence=metric,
                topn=10,
            )
            per_topic = cm.get_coherence_per_topic()
            for tid, val in zip(valid_tids, per_topic):
                metric_results[metric][tid] = float(val)
            mean_val = float(np.nanmean(per_topic))
            print(f"  mean {metric}: {mean_val:.4f}")
        except Exception as e:
            print(f"  ERROR computing {metric}: {e}")
            continue

    # Build per-topic table
    for _, row in topic_info.iterrows():
        tid = int(row["Topic"])
        d = {
            "topic_id": tid,
            "size": int(row["Count"]),
            "in_robust_subset": tid in ROBUST_TOPICS,
            "topwords": row["TopWords"][:80],
        }
        for m in METRICS:
            d[m] = metric_results[m].get(tid, None)
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "per_topic_coherence.csv", index=False)

    # --------- Random-topic baseline (sanity check) ----------
    print("\n=== Random-topic baseline (sanity check) ===")
    rng = np.random.default_rng(42)
    n_random = 10
    vocab_terms = list(dictionary.token2id.keys())
    random_topics = []
    for _ in range(n_random):
        random_topics.append(list(rng.choice(vocab_terms, size=10, replace=False)))
    random_means = {}
    for metric in METRICS:
        try:
            cm = CoherenceModel(
                topics=random_topics, texts=texts, dictionary=dictionary,
                coherence=metric, topn=10,
            )
            random_means[metric] = float(np.nanmean(cm.get_coherence_per_topic()))
            print(f"  random {metric} mean: {random_means[metric]:.4f}")
        except Exception as e:
            print(f"  random {metric} ERROR: {e}")
            random_means[metric] = None

    # --------- Summary ----------
    summary = {
        "K_baseline_topics": K,
        "N_robust_motifs": len(ROBUST_TOPICS),
        "topwords_used": "BERTopic c-TF-IDF top-10",
        "reference_corpus": "intrinsic (own corpus, POS-filtered content lemmas)",
        "dictionary_size": len(dictionary),
    }
    for m in METRICS:
        vals = [v for v in metric_results[m].values() if v is not None]
        robust_vals = [metric_results[m].get(t) for t in ROBUST_TOPICS
                       if metric_results[m].get(t) is not None]
        summary[m] = {
            "all_25_mean": round(float(np.mean(vals)), 4) if vals else None,
            "all_25_std": round(float(np.std(vals)), 4) if vals else None,
            "all_25_max": round(float(np.max(vals)), 4) if vals else None,
            "all_25_min": round(float(np.min(vals)), 4) if vals else None,
            "robust6_mean": round(float(np.mean(robust_vals)), 4) if robust_vals else None,
            "robust6_max": round(float(np.max(robust_vals)), 4) if robust_vals else None,
            "robust6_min": round(float(np.min(robust_vals)), 4) if robust_vals else None,
            "random_baseline": round(random_means[m], 4) if random_means.get(m) is not None else None,
            "gap_real_vs_random": round(float(np.mean(vals)) - random_means[m], 4)
                                   if vals and random_means.get(m) is not None else None,
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    # Print
    print("\n" + "=" * 80)
    print("Coherence baseline — headline")
    print("=" * 80)
    print(f"  Reference: intrinsic (own corpus, POS-filtered, vocab={len(dictionary):,})")
    print(f"  Topics: 25 baseline (c-TF-IDF top-10)")
    print(f"  Robust subset: {ROBUST_TOPICS}")
    print()
    print(f"  {'metric':<10s}  {'all-25 mean':>12s}  {'robust-6 mean':>14s}  "
          f"{'random':>8s}  {'gap':>8s}")
    for m in METRICS:
        s = summary[m]
        all25 = f"{s['all_25_mean']:.4f}" if s['all_25_mean'] is not None else "—"
        r6 = f"{s['robust6_mean']:.4f}" if s['robust6_mean'] is not None else "—"
        rand = f"{s['random_baseline']:.4f}" if s['random_baseline'] is not None else "—"
        gap = f"{s['gap_real_vs_random']:.4f}" if s['gap_real_vs_random'] is not None else "—"
        print(f"  {m:<10s}  {all25:>12s}  {r6:>14s}  {rand:>8s}  {gap:>8s}")
    print(f"\n  Elapsed: {round(time.time()-t_total, 1)}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
