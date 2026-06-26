"""
11_keybert_validation.py

Validate whether BERTopic's c-TF-IDF keywords for our 6 robust motifs are
method-independent by comparing against KeyBERT [Grootendorst 2020].

c-TF-IDF (BERTopic):
  - Aggregates each topic's documents into a "class document".
  - Computes class-distinctive term weights (TF on class × IDF across classes).
  - Keywords inherit the cluster-decision pipeline (UMAP+HDBSCAN).

KeyBERT (this script):
  - Computes doc-level SBERT embedding.
  - Computes candidate term embeddings (we use POS-filtered lemmas from our
    preprocessing).
  - Keyword score = cosine sim(term embedding, document embedding), aggregated
    per topic (mean rank-1 list across the topic's docs).
  - Independent of cluster decision; only depends on docs-in-topic.

If KeyBERT and c-TF-IDF largely agree on top-k keywords for the robust topics,
the topic representation is method-independent (supports BERTopic). If they
diverge, c-TF-IDF keywords may be a representation artefact (cautions
BERTopic).

We focus on the six audit-robust motifs (4/4 axes passed):
  T0 바다, T1 꽃, T7 비, T11 메타시, T14 음악, T16 고양이

Outputs:
  results/keybert/per_topic_keybert.json     KeyBERT top-10 per topic
  results/keybert/comparison.csv             c-TF-IDF top-k vs KeyBERT top-k
  results/keybert/summary.json
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
FILT_DIR = ROOT / "data" / "filtered"
TOK_DIR = ROOT / "data" / "tokenized"
BASE = ROOT / "results" / "baseline"
OUT = ROOT / "results" / "keybert"
OUT.mkdir(parents=True, exist_ok=True)


# Six audit-robust motifs (4/4 axes passed)
ROBUST_TOPICS = [0, 1, 7, 11, 14, 16]

TOP_K = 10
N_CANDIDATES_PER_DOC = 5  # KeyBERT: how many candidates per doc


def parse_topwords(s):
    if not isinstance(s, str) or not s:
        return []
    return [w.split("(")[0].strip() for w in s.split(",") if "(" in w]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def main():
    t_total = time.time()
    print("Loading data...")
    embeddings = np.load(BASE / "embeddings.npy")  # (N, 768) doc embeddings
    per_doc = pd.read_csv(BASE / "topic_per_doc.csv")
    topic_info = pd.read_csv(BASE / "topic_info.csv")
    topic_info = topic_info[topic_info["Topic"] != -1].reset_index(drop=True)
    content = pd.read_parquet(TOK_DIR / "work_content_tokens.parquet")
    works = pd.read_parquet(FILT_DIR / "works_filtered.parquet")
    m = works.merge(content, on="work_id", how="inner")
    docs_pos = m["content_tokens"].fillna("").tolist()
    work_ids = m["work_id"].tolist()
    N = embeddings.shape[0]
    assert len(docs_pos) == N
    print(f"  N={N} docs")

    # --------------------------------------------------- Vocabulary
    # Use POS-filtered lemmas across the corpus
    print("\nBuilding vocabulary from POS-filtered tokens...")
    counter = Counter()
    for doc in docs_pos:
        counter.update(doc.split())
    # Keep terms that appear in ≥ 2 docs (matches BERTopic's min_df=2)
    doc_frequency = Counter()
    for doc in docs_pos:
        for term in set(doc.split()):
            doc_frequency[term] += 1
    vocab = [t for t, c in doc_frequency.items() if c >= 2]
    print(f"  {len(vocab):,} unique terms (≥2 docs)")

    # --------------------------------------------------- Term embeddings
    print("\nEmbedding vocabulary with ko-sroberta-multitask...")
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("jhgan/ko-sroberta-multitask")
    BATCH = 256
    term_embs = []
    for i in tqdm(range(0, len(vocab), BATCH), desc="  encoding terms"):
        batch = vocab[i:i+BATCH]
        embs = st.encode(batch, batch_size=BATCH, show_progress_bar=False,
                         convert_to_numpy=True)
        term_embs.append(embs)
    term_embs = np.concatenate(term_embs, axis=0)
    # L2 normalise
    term_embs = term_embs / np.maximum(np.linalg.norm(term_embs, axis=1, keepdims=True), 1e-12)
    # Doc embedding L2 normalise
    doc_embs = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)

    # --------------------------------------------------- KeyBERT per robust topic
    print("\nKeyBERT keyword extraction per robust topic...")
    term_idx = {t: i for i, t in enumerate(vocab)}
    keybert_per_topic = {}
    rows = []

    ctfidf_topwords = {int(r.Topic): parse_topwords(r.TopWords)
                       for _, r in topic_info.iterrows()}

    for tid in ROBUST_TOPICS:
        if tid not in ctfidf_topwords:
            print(f"  T{tid} not in baseline; skip")
            continue
        # Docs in this topic
        doc_mask = (per_doc["topic"] == tid).values
        doc_ids = np.where(doc_mask)[0]
        n_docs = len(doc_ids)
        if n_docs == 0:
            print(f"  T{tid}: no docs; skip")
            continue

        # Per-doc top-N candidates: cosine sim(term, doc) → top N
        # Aggregate via mean rank: for each candidate term across docs, average sim
        # We do this efficiently: avg doc embedding then sim with terms
        topic_doc_embs = doc_embs[doc_ids]  # (n_docs, 768)
        # Mean doc embedding
        topic_centroid = topic_doc_embs.mean(axis=0)
        topic_centroid = topic_centroid / max(np.linalg.norm(topic_centroid), 1e-12)
        # Sim of every term to centroid
        sim_to_centroid = term_embs @ topic_centroid  # (vocab_size,)
        # Top-K
        # Restrict to terms that actually appear in this topic's docs (KeyBERT default)
        topic_terms_used = set()
        for did in doc_ids:
            topic_terms_used.update(docs_pos[did].split())
        topic_terms_used &= set(vocab)
        # Restricted scoring
        scored = [(t, float(sim_to_centroid[term_idx[t]])) for t in topic_terms_used]
        scored.sort(key=lambda x: -x[1])
        keybert_top = scored[:TOP_K]

        ctfidf_top = ctfidf_topwords[tid][:TOP_K]
        keybert_terms = [w for w, _ in keybert_top]

        # Overlap
        ovlp = set(ctfidf_top) & set(keybert_terms)
        jac = jaccard(ctfidf_top, keybert_terms)

        # First-position match: is ctfidf top-1 in keybert top-10?
        ctfidf_top1 = ctfidf_top[0] if ctfidf_top else None
        top1_in_keybert = ctfidf_top1 in keybert_terms if ctfidf_top1 else False

        keybert_per_topic[tid] = {
            "n_docs": int(n_docs),
            "ctfidf_top10": ctfidf_top,
            "keybert_top10": [(w, round(s, 4)) for w, s in keybert_top],
            "overlap_count": len(ovlp),
            "jaccard_top10": round(jac, 4),
            "ctfidf_top1_in_keybert_top10": top1_in_keybert,
        }

        rows.append({
            "topic_id": tid,
            "n_docs": int(n_docs),
            "ctfidf_top10": ", ".join(ctfidf_top),
            "keybert_top10": ", ".join(f"{w}({s:.2f})" for w, s in keybert_top),
            "overlap_count": int(len(ovlp)),
            "jaccard_top10": round(jac, 4),
            "ctfidf_top1_in_keybert_top10": top1_in_keybert,
        })

        print(f"\n  T{tid} (n_docs={n_docs}):")
        print(f"    c-TF-IDF top10: {', '.join(ctfidf_top)}")
        print(f"    KeyBERT  top10: {', '.join(w for w,_ in keybert_top)}")
        print(f"    Overlap: {len(ovlp)}/10  Jaccard: {jac:.3f}  ctfidf_top1 in keybert: {top1_in_keybert}")

    pd.DataFrame(rows).to_csv(OUT / "comparison.csv", index=False)
    (OUT / "per_topic_keybert.json").write_text(
        json.dumps(keybert_per_topic, indent=2, ensure_ascii=False))

    # --------------------------------------------------- Summary
    if rows:
        mean_overlap = float(np.mean([r["overlap_count"] for r in rows]))
        mean_jaccard = float(np.mean([r["jaccard_top10"] for r in rows]))
        n_top1_match = int(sum(r["ctfidf_top1_in_keybert_top10"] for r in rows))

        if mean_jaccard >= 0.5 and n_top1_match == len(rows):
            verdict = "SUPPORT — KeyBERT and c-TF-IDF strongly converge (method-independent keywords)"
        elif mean_jaccard >= 0.3:
            verdict = "PARTIAL SUPPORT — meaningful overlap; some top-words diverge (c-TF-IDF and centroid-based have partial agreement)"
        else:
            verdict = "BETRAY — KeyBERT and c-TF-IDF diverge (c-TF-IDF keywords may be representation artefact)"

        summary = {
            "n_topics_checked": len(rows),
            "mean_top10_overlap_count": round(mean_overlap, 2),
            "mean_top10_jaccard": round(mean_jaccard, 4),
            "ctfidf_top1_in_keybert_top10_count": n_top1_match,
            "verdict": verdict,
            "elapsed_s": round(time.time() - t_total, 1),
        }
    else:
        summary = {"error": "no rows"}

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("KeyBERT validation — headline")
    print("=" * 70)
    print(f"  Topics checked:                       {summary.get('n_topics_checked')}")
    print(f"  Mean top-10 overlap count:            {summary.get('mean_top10_overlap_count')}")
    print(f"  Mean top-10 Jaccard:                  {summary.get('mean_top10_jaccard')}")
    print(f"  c-TF-IDF top-1 ∈ KeyBERT top-10:      {summary.get('ctfidf_top1_in_keybert_top10_count')}/{len(rows)}")
    print(f"\n  VERDICT: {summary.get('verdict')}")
    print(f"  Elapsed: {summary.get('elapsed_s')}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
