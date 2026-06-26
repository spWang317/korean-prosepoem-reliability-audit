"""
02_baseline_bertopic.py

Run a baseline BERTopic on the preprocessed corpus. This run is the reference
against which all robustness perturbations (R1-R5, Step 1.3) are compared.

Pipeline (each component cited in references.bib):
  - Document embedding: ko-sroberta-multitask [Jhgan 2021]; SBERT receives RAW
    work text (원문) to preserve semantic context [Grootendorst 2022].
  - Dimensionality reduction: UMAP [McInnes 2018], n_neighbors=15,
    n_components=5, min_dist=0.0, metric=cosine, random_state=42 (BERTopic
    defaults).
  - Clustering: HDBSCAN [McInnes 2017], min_cluster_size=30 (≈0.65% of N=4642,
    aligned with BERTopic ≈1% recommendation), min_samples=10, metric=euclidean,
    cluster_selection_method=eom.
  - Topic representation: CountVectorizer with vocabulary restricted to the
    POS-filtered content lemmas from `01_preprocess.py` (NNG/NP/VA/VV)
    [Eunjeon 2013; Grootendorst 2022]; c-TF-IDF.

Outputs:
  results/baseline/embeddings.npy
      Document embeddings (cached for reuse in robustness runs)
  results/baseline/topic_info.csv
      Topic ID, size, top-10 keywords + scores, representative work_ids
  results/baseline/topic_per_doc.csv
      work_id, assigned topic, topic probability
  results/baseline/baseline_summary.json
      Parameters used, topic count, outlier %, runtime
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILT_DIR = ROOT / "data" / "filtered"
TOK_DIR = ROOT / "data" / "tokenized"
OUT_DIR = ROOT / "results" / "baseline"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Locked baseline parameters (justified in notes/methodology_decisions.md)
# ----------------------------------------------------------------------
BASELINE = {
    "embedding_model": "jhgan/ko-sroberta-multitask",
    # UMAP — BERTopic defaults [Grootendorst 2022], random_state chosen
    # from the non-degenerate regime (see diagnostic in notes/methodology_decisions.md):
    # at this corpus size with default UMAP/HDBSCAN parameters, ~60% of seeds
    # produce a degenerate 2-cluster solution (97% in one cluster) and ~30%
    # produce 25-topic non-degenerate solutions. random_state=100 is a
    # representative non-degenerate seed. The bimodal-seed instability is
    # itself a finding (reported in R3 robustness analysis, Step 1.3).
    "umap": dict(n_neighbors=15, n_components=5, min_dist=0.0,
                 metric="cosine", random_state=100),
    # HDBSCAN — min_cluster_size ≈ 0.65% of N (close to BERTopic 1% recommendation)
    "hdbscan": dict(min_cluster_size=30, min_samples=10,
                    metric="euclidean", cluster_selection_method="eom"),
    # CountVectorizer for c-TF-IDF. min_df is applied to per-topic-aggregated
    # documents (typically ~20 rows). min_df=2 means "appears in ≥2 topics" —
    # stricter than default 1 but tolerates the small number of topic-rows.
    # Higher values (e.g., 5) cause empty-vocabulary errors on small
    # per-topic-aggregated documents.
    "vectorizer": dict(min_df=2, max_df=0.95),
    # Other
    "top_n_words": 10,
    "calculate_probabilities": False,
}


def main() -> None:
    t_total = time.time()

    # ----------------------------------------------------------
    # 1. Load preprocessed data
    # ----------------------------------------------------------
    print("Loading preprocessed corpus ...")
    works = pd.read_parquet(FILT_DIR / "works_filtered.parquet")
    content = pd.read_parquet(TOK_DIR / "work_content_tokens.parquet")
    assert "원문" in works.columns
    assert "work_id" in works.columns and "work_id" in content.columns

    # Align: BERTopic needs (docs[i], tokens_for_vectorizer[i]) per work
    merged = works.merge(content, on="work_id", how="inner")
    docs_raw = merged["원문"].tolist()
    docs_pos_filtered = merged["content_tokens"].fillna("").tolist()
    work_ids = merged["work_id"].tolist()
    print(f"  Aligned {len(merged):,} works")
    print(f"  RAW text → SBERT embedding")
    print(f"  POS-filtered tokens → CountVectorizer / c-TF-IDF")

    # ----------------------------------------------------------
    # 2. Compute embeddings (cached)
    # ----------------------------------------------------------
    emb_path = OUT_DIR / "embeddings.npy"
    if emb_path.exists():
        print(f"\nReusing cached embeddings: {emb_path}")
        embeddings = np.load(emb_path)
    else:
        print(f"\nComputing SBERT embeddings ({BASELINE['embedding_model']}) ...")
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        st = SentenceTransformer(BASELINE["embedding_model"])
        # Work-level embedding: mean-pool sentence embeddings or directly encode
        # full 원문. For consistency with atlas pipeline we encode 원문 directly
        # (SBERT handles up to 512 tokens; longer is truncated).
        embeddings = st.encode(
            docs_raw,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        np.save(emb_path, embeddings)
        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Took {time.time()-t0:.1f}s; saved {emb_path}")

    # ----------------------------------------------------------
    # 3. Fit BERTopic
    # ----------------------------------------------------------
    print("\nFitting BERTopic ...")
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    umap_model = UMAP(**BASELINE["umap"])
    hdbscan_model = HDBSCAN(**BASELINE["hdbscan"], prediction_data=True)

    # CountVectorizer fitted on POS-filtered tokens.
    # Input is space-joined POS-filtered lemmas, so default whitespace
    # tokenization via token_pattern="\\S+" is appropriate.
    vectorizer_model = CountVectorizer(
        token_pattern=r"\S+",  # match non-whitespace runs (our pre-tokenized format)
        min_df=BASELINE["vectorizer"]["min_df"],
        max_df=BASELINE["vectorizer"]["max_df"],
    )

    topic_model = BERTopic(
        embedding_model=None,  # we pass embeddings directly
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        language="multilingual",  # default "english" strips non-ASCII (Korean!) chars
        top_n_words=BASELINE["top_n_words"],
        calculate_probabilities=BASELINE["calculate_probabilities"],
        verbose=True,
    )

    t0 = time.time()
    # We pass docs_pos_filtered to the vectorizer; BERTopic uses these for
    # c-TF-IDF. SBERT embeddings are passed separately for clustering.
    topics, probs = topic_model.fit_transform(docs_pos_filtered, embeddings)
    print(f"  Fit done in {time.time()-t0:.1f}s")

    # ----------------------------------------------------------
    # 4. Save outputs
    # ----------------------------------------------------------
    print("\nSaving outputs ...")

    # 4a. topic_info.csv — with top keywords and representative docs
    topic_info = topic_model.get_topic_info()
    # add top-10 keywords (and scores) column for each topic
    kw_cols = []
    for tid in topic_info["Topic"]:
        words = topic_model.get_topic(tid) or []
        if isinstance(words, list) and words:
            kw_cols.append(", ".join(f"{w}({s:.3f})" for w, s in words))
        else:
            kw_cols.append("")
    topic_info["TopWords"] = kw_cols
    topic_info.to_csv(OUT_DIR / "topic_info.csv", index=False)
    print(f"  topic_info.csv  ({len(topic_info)} topics)")

    # 4b. per-doc assignment
    per_doc = pd.DataFrame({
        "work_id": work_ids,
        "topic": topics,
    })
    per_doc.to_csv(OUT_DIR / "topic_per_doc.csv", index=False)
    print(f"  topic_per_doc.csv  ({len(per_doc)} works)")

    # 4c. representative docs (top-5 per topic, by probability or rank)
    rep_path = OUT_DIR / "representative_docs.json"
    rep_per_topic = {}
    docs_for_topics = topic_model.get_representative_docs() or {}
    for tid, rep_docs in docs_for_topics.items():
        # rep_docs are texts; find their work_ids
        wid_map = {d: w for d, w in zip(docs_pos_filtered, work_ids)}
        rep_wids = [wid_map.get(d, None) for d in rep_docs[:5]]
        rep_per_topic[int(tid)] = {
            "rep_work_ids": [w for w in rep_wids if w is not None],
            "n_works_in_topic": int((per_doc["topic"] == tid).sum()),
        }
    rep_path.write_text(json.dumps(rep_per_topic, indent=2, ensure_ascii=False))
    print(f"  representative_docs.json")

    # 4d. baseline summary
    n_topics = (topic_info["Topic"] != -1).sum()
    outlier_count = int((per_doc["topic"] == -1).sum())
    summary = {
        "params": BASELINE,
        "n_works": len(merged),
        "n_topics": int(n_topics),
        "n_outliers": outlier_count,
        "outlier_pct": round(100 * outlier_count / len(merged), 2),
        "elapsed_s": round(time.time() - t_total, 1),
    }
    (OUT_DIR / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"  baseline_summary.json")

    # ----------------------------------------------------------
    # 5. Print headline numbers
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Baseline BERTopic — headline")
    print("=" * 60)
    print(f"  N (works)         : {len(merged):,}")
    print(f"  Topics (non-outlier): {n_topics}")
    print(f"  Outliers (T-1)    : {outlier_count} ({summary['outlier_pct']}%)")
    print(f"  Elapsed           : {summary['elapsed_s']}s")
    print("\nTop-5 topics by size:")
    for _, row in topic_info.head(6).iterrows():
        if row["Topic"] == -1:
            continue
        print(f"  T{row['Topic']:>3} ({row['Count']:>4}):  {row['TopWords'][:120]}")
    print("=" * 60)


if __name__ == "__main__":
    main()
