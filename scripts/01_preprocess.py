"""
01_preprocess.py

Build the analysis corpus from the raw atlas:
  1. Length filter (P5-P95 of human atlas in characters)
  2. Exact-string deduplication on 원문
  3. KSS sentence segmentation [Ko & Park 2021]
  4. Mecab-ko POS tagging [Eunjeon 2013]
  5. Content-word filter: keep tokens with POS ∈ {NNG, NP, VA, VV}
     (replaces hand-curated stopword list)

Outputs:
  data/filtered/works_filtered.parquet
      work-level metadata after length filter + dedup
  data/filtered/sentences_filtered.parquet
      (work_id, sent_idx, sent_text)
  data/tokenized/lemmas_filtered.parquet
      (work_id, sent_idx, token_idx, surface, pos, lemma)
  data/tokenized/work_content_tokens.parquet
      per-work bag of content lemmas (space-joined) for CountVectorizer
  data/preprocessing_report.json
      filter counts at each stage
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
FILT_DIR = ROOT / "data" / "filtered"
TOK_DIR = ROOT / "data" / "tokenized"
FILT_DIR.mkdir(parents=True, exist_ok=True)
TOK_DIR.mkdir(parents=True, exist_ok=True)

# Content POS tags (Mecab-ko Sejong tagset):
#   NNG = common noun
#   NP  = pronoun
#   VA  = descriptive verb (형용사)
#   VV  = verb
CONTENT_POS = {"NNG", "NP", "VA", "VV"}


def main() -> None:
    t0 = time.time()
    report: dict = {}

    # ----------------------------------------------------------
    # Stage 0. Load raw atlas
    # ----------------------------------------------------------
    print("Stage 0: load raw atlas metadata.csv ...")
    meta = pd.read_csv(RAW_DIR / "metadata.csv")
    print(f"  Loaded {len(meta):,} rows; columns: {list(meta.columns)}")
    assert "원문" in meta.columns, "metadata.csv must have 원문 column"
    assert "work_id" in meta.columns, "metadata.csv must have work_id column"
    n0 = len(meta)
    report["stage_0_raw"] = {"n_works": n0}

    # ----------------------------------------------------------
    # Stage 1. Length filter (P5-P95 of human atlas char-count)
    # ----------------------------------------------------------
    print("\nStage 1: length filter (P5-P95 of char-count) ...")
    meta = meta.dropna(subset=["원문"]).copy()
    meta["n_chars"] = meta["원문"].str.len()
    p5 = int(meta["n_chars"].quantile(0.05))
    p95 = int(meta["n_chars"].quantile(0.95))
    print(f"  Char-count P5 = {p5}, P95 = {p95}")
    mask = meta["n_chars"].between(p5, p95)
    meta_filt = meta[mask].copy()
    print(f"  Kept {len(meta_filt):,} / {len(meta):,} works ({len(meta_filt)/len(meta)*100:.1f}%)")
    report["stage_1_length_filter"] = {
        "p5_chars": p5, "p95_chars": p95,
        "n_in": len(meta), "n_out": len(meta_filt),
    }

    # ----------------------------------------------------------
    # Stage 2. Exact-string dedup on 원문
    # ----------------------------------------------------------
    print("\nStage 2: exact-string dedup on 원문 ...")
    n_pre = len(meta_filt)
    meta_filt = meta_filt.drop_duplicates(subset=["원문"], keep="first")
    n_post = len(meta_filt)
    print(f"  Removed {n_pre - n_post} exact duplicates; kept {n_post:,}")
    report["stage_2_dedup"] = {"n_in": n_pre, "n_out": n_post, "n_dropped": n_pre - n_post}

    # Save work-level filtered metadata
    work_cols = [c for c in meta_filt.columns if c != "n_chars"]
    works_out = FILT_DIR / "works_filtered.parquet"
    meta_filt[work_cols].to_parquet(works_out, index=False)
    print(f"  → wrote {works_out}")

    # ----------------------------------------------------------
    # Stage 3. KSS sentence segmentation
    # ----------------------------------------------------------
    print("\nStage 3: KSS sentence segmentation ...")
    print("  Loading KSS (may take a moment) ...")
    from kss import Kss
    splitter = Kss("split_sentences")

    sent_records = []
    for work_id, text in tqdm(zip(meta_filt["work_id"], meta_filt["원문"]),
                              total=len(meta_filt), desc="  KSS"):
        try:
            sents = splitter(text)
        except Exception as e:
            # extremely rare; fallback to single-sentence
            sents = [text]
        for i, s in enumerate(sents):
            s_clean = s.strip()
            if s_clean:
                sent_records.append({"work_id": work_id, "sent_idx": i,
                                     "sent_text": s_clean})

    sentences = pd.DataFrame(sent_records)
    sent_out = FILT_DIR / "sentences_filtered.parquet"
    sentences.to_parquet(sent_out, index=False)
    print(f"  Segmented to {len(sentences):,} sentences")
    print(f"  → wrote {sent_out}")
    report["stage_3_kss"] = {
        "n_sentences": len(sentences),
        "mean_sents_per_work": float(len(sentences) / len(meta_filt)),
    }

    # ----------------------------------------------------------
    # Stage 4. Mecab-ko POS tagging
    # ----------------------------------------------------------
    print("\nStage 4: Mecab-ko POS tagging ...")
    print("  Loading Mecab ...")
    try:
        from mecab import MeCab
        tagger = MeCab()
    except ImportError:
        from konlpy.tag import Mecab
        tagger = Mecab()

    def pos_with_lemma(text: str):
        """Return list of (surface, pos, lemma) triples.
        Lemma for verbs/adjectives is the dictionary form with -다 suffix."""
        out = []
        try:
            tags = tagger.pos(text)
        except Exception:
            return out
        for surface, pos in tags:
            # Mecab returns compound tags sometimes ("NNG+JKS"); first one wins
            pos_base = pos.split("+")[0]
            if pos_base == "VV" or pos_base == "VA":
                lemma = surface + "다"  # restore dictionary form
            else:
                lemma = surface
            out.append((surface, pos_base, lemma))
        return out

    lemma_records = []
    for _, row in tqdm(sentences.iterrows(), total=len(sentences), desc="  MeCab"):
        wid = row["work_id"]
        sidx = row["sent_idx"]
        for tidx, (surf, pos, lem) in enumerate(pos_with_lemma(row["sent_text"])):
            lemma_records.append({
                "work_id": wid, "sent_idx": sidx, "token_idx": tidx,
                "surface": surf, "pos": pos, "lemma": lem,
            })

    lemmas = pd.DataFrame(lemma_records)
    lemmas_out = TOK_DIR / "lemmas_filtered.parquet"
    lemmas.to_parquet(lemmas_out, index=False)
    print(f"  Tagged {len(lemmas):,} tokens; {lemmas['pos'].nunique()} distinct POS tags")
    print(f"  → wrote {lemmas_out}")
    report["stage_4_mecab"] = {
        "n_tokens": len(lemmas),
        "n_pos_tags": int(lemmas["pos"].nunique()),
    }

    # ----------------------------------------------------------
    # Stage 5. Content-word filter (POS ∈ CONTENT_POS)
    # ----------------------------------------------------------
    print(f"\nStage 5: content-word filter (POS ∈ {sorted(CONTENT_POS)}) ...")
    content = lemmas[lemmas["pos"].isin(CONTENT_POS)].copy()
    print(f"  Kept {len(content):,} / {len(lemmas):,} tokens ({len(content)/len(lemmas)*100:.1f}%)")
    pos_dist = content["pos"].value_counts().to_dict()
    print(f"  POS distribution: {pos_dist}")

    # Per-work bag of content lemmas (space-joined string for CountVectorizer)
    per_work = (content.groupby("work_id")["lemma"]
                       .apply(lambda s: " ".join(s.tolist()))
                       .reset_index()
                       .rename(columns={"lemma": "content_tokens"}))
    per_work_out = TOK_DIR / "work_content_tokens.parquet"
    per_work.to_parquet(per_work_out, index=False)
    print(f"  → wrote {per_work_out}  ({len(per_work):,} works)")
    report["stage_5_content_filter"] = {
        "n_content_tokens": len(content),
        "pos_distribution": pos_dist,
        "n_works_with_content": len(per_work),
    }

    # ----------------------------------------------------------
    # Save report
    # ----------------------------------------------------------
    report["elapsed_s"] = time.time() - t0
    report_path = ROOT / "data" / "preprocessing_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nDone in {report['elapsed_s']:.1f}s. Report: {report_path}")


if __name__ == "__main__":
    main()
