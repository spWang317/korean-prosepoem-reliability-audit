"""
18_motif_representative_works.py

For each of the 25 anchor topics, extract the top-N documents that have
the highest NNLS-derived share of that motif. These representative
works are starting points for the qualitative analysis in paper §6.

For each motif:
  - Top 15 docs by motif % (descending)
  - work_id, motif %, unexplained %, hard assignment
  - Full text (not just preview)

Audit-robust 6 motifs are flagged for priority.

Outputs:
  results/representative/by_motif/T{tid}_top15.json   per-motif JSON
  results/representative/all_motifs_top10.md          one-page markdown overview
  results/representative/summary.csv                  flat table
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILT = ROOT / "data" / "filtered"
BASE = ROOT / "results" / "baseline"
DIST = ROOT / "results" / "distribution"
OUT = ROOT / "results" / "representative"
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "by_motif").mkdir(exist_ok=True)

ROBUST_TOPICS = {0, 1, 7, 11, 14, 16}
TOP_N = 15


def parse_topwords(s):
    if not isinstance(s, str) or not s:
        return []
    return [w.split("(")[0].strip() for w in s.split(",") if "(" in w]


def main():
    t_total = time.time()
    print("Loading data ...")
    df = pd.read_csv(DIST / "doc_motif_distribution.csv")
    works = pd.read_parquet(FILT / "works_filtered.parquet")
    topic_info = pd.read_csv(BASE / "topic_info.csv")
    topic_info = topic_info[topic_info["Topic"] != -1].reset_index(drop=True)

    text_lookup = dict(zip(works["work_id"], works["원문"]))
    # Available columns: 작가, 책 제목, 작품 제목, 출판년도, 등단년도
    title_col = "작품 제목" if "작품 제목" in works.columns else "제목"
    if title_col in works.columns:
        title_lookup = dict(zip(works["work_id"], works[title_col].fillna("")))
    else:
        title_lookup = {}
    if "작가" in works.columns:
        author_lookup = dict(zip(works["work_id"], works["작가"].fillna("")))
    else:
        author_lookup = {}
    year_lookup = (dict(zip(works["work_id"], works["출판년도"].fillna("")))
                   if "출판년도" in works.columns else {})

    print(f"  N={len(df)} docs, K={len(topic_info)} topics")
    print(f"  Available columns in works: {list(works.columns)[:10]}")

    flat_rows = []
    md_lines = ["# 25 Motif별 대표 시 — Top 10 (NNLS dominant share)\n"]
    md_lines.append(f"NNLS doc-motif distribution 기준. 각 motif에서 그 motif % 가장 높은 시 10편.")
    md_lines.append(f"★ = audit-robust 4/4. 옆 % = 그 시에서 motif가 차지하는 비율.\n")
    md_lines.append("---\n")

    for _, t in topic_info.iterrows():
        tid = int(t["Topic"])
        col = f"T{tid}_pct"
        if col not in df.columns:
            continue
        words = parse_topwords(t["TopWords"])[:7]
        topwords_str = ", ".join(words)
        flag = "★" if tid in ROBUST_TOPICS else " "

        # Sort docs by this motif's share
        top_docs = df.nlargest(TOP_N, col)

        # Save per-motif JSON (full text)
        motif_data = {
            "topic_id": tid,
            "is_audit_robust": tid in ROBUST_TOPICS,
            "topwords": topwords_str,
            "top_works": []
        }
        for rank, (_, r) in enumerate(top_docs.iterrows(), start=1):
            wid = str(r["work_id"])
            text = text_lookup.get(wid, text_lookup.get(r["work_id"], ""))
            title = title_lookup.get(wid, title_lookup.get(r["work_id"], ""))
            author = author_lookup.get(wid, author_lookup.get(r["work_id"], ""))
            entry = {
                "rank": rank,
                "work_id": wid,
                "title": str(title) if title else None,
                "author": str(author) if author else None,
                "motif_pct": round(float(r[col]), 2),
                "unexplained_pct": round(float(r["unexplained_pct"]), 2),
                "assigned_topic_hard": int(r["assigned_topic_hard"]),
                "text": text,
            }
            motif_data["top_works"].append(entry)
            flat_rows.append({
                "topic_id": tid,
                "is_robust": tid in ROBUST_TOPICS,
                "topwords": topwords_str,
                "rank": rank,
                "work_id": wid,
                "title": title,
                "author": author,
                "motif_pct": round(float(r[col]), 2),
                "unexplained_pct": round(float(r["unexplained_pct"]), 2),
                "text_preview": text[:300].replace("\n", " ") if text else "",
            })

        (OUT / "by_motif" / f"T{tid:02d}_top15.json").write_text(
            json.dumps(motif_data, indent=2, ensure_ascii=False)
        )

        # Markdown — top 10 only
        md_lines.append(f"## T{tid} {flag}  *{topwords_str}*\n")
        for rank, (_, r) in enumerate(top_docs.head(10).iterrows(), start=1):
            wid = str(r["work_id"])
            text = text_lookup.get(wid, text_lookup.get(r["work_id"], ""))
            title = title_lookup.get(wid, title_lookup.get(r["work_id"], ""))
            author = author_lookup.get(wid, author_lookup.get(r["work_id"], ""))
            year = year_lookup.get(wid, year_lookup.get(r["work_id"], ""))
            meta_parts = []
            if author:
                meta_parts.append(str(author))
            if title:
                meta_parts.append(f"「{title}」")
            if year:
                meta_parts.append(f"({year})")
            meta_str = " ".join(meta_parts) if meta_parts else "(meta 미확인)"
            header = f"**#{rank} {meta_str}** — motif {r[col]:.1f}%, unexp {r['unexplained_pct']:.1f}%, hard=T{int(r['assigned_topic_hard'])}"
            md_lines.append(header)
            text_short = text.replace("\n", " ")[:400]
            md_lines.append(f"> {text_short}…\n" if len(text) > 400 else f"> {text_short}\n")
        md_lines.append("\n---\n")

    pd.DataFrame(flat_rows).to_csv(OUT / "summary.csv", index=False)
    (OUT / "all_motifs_top10.md").write_text("\n".join(md_lines))

    print(f"\n  Saved: by_motif/ (25 JSON files)")
    print(f"  Saved: all_motifs_top10.md ({len(md_lines)} lines)")
    print(f"  Saved: summary.csv ({len(flat_rows)} rows)")
    print(f"  Elapsed: {time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
