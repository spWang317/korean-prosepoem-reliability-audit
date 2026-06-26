"""
07_multi_tau_sensitivity.py

Formal sensitivity reporting across multiple persistence thresholds.

Background: τ = 0.7 is the pre-committed threshold from Greene 2014 (and
used in our methodology_decisions.md). Reporting only one value invites the
"why this τ?" reviewer question. Standard practice [Saltelli 2008 sensitivity
reporting; Schroeder & Wood-Doughty 2025] is to report a sensitivity table
across τ values, with τ = 0.7 explicitly marked as the pre-committed value.

This script:
  - Reads results/robustness/persistence_matrix.csv (from 04_marginal_robustness.py)
  - Reports robust-topic counts at τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
  - Combines this with psychometric reliability (06) for cross-axis view
  - Adds R3 conditional-on-non-degenerate reporting (R3*)

R3 ceiling note: with 10 seeds and ~6 degenerate (UMAP bimodal collapse)
seeds, a single topic can be observed in at most 4 seeds → max R3 raw
persistence = 0.4. Reporting R3* (conditional on non-degenerate seeds) gives
a fair within-regime persistence.

Outputs:
  results/sensitivity/multi_tau_table.csv
  results/sensitivity/multi_tau_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PMTX_PATH = ROOT / "results" / "robustness" / "persistence_matrix.csv"
R3_RUNS = ROOT / "results" / "robustness" / "r3_seeds" / "runs.json"
PSYCH = ROOT / "results" / "psychometric" / "reliability_per_topic.csv"
OUT_DIR = ROOT / "results" / "sensitivity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
PRE_COMMIT_TAU = 0.70  # Greene 2014


def main():
    print("Loading persistence matrix...")
    df = pd.read_csv(PMTX_PATH)
    K = len(df)

    # Identify which seeds are degenerate (n_topics ≤ 5 is our heuristic for collapse)
    with open(R3_RUNS) as f:
        r3 = json.load(f)
    seeds_runs = r3.get("runs", [])
    seed_ntopics = {r["seed"]: r["n_topics"] for r in seeds_runs}
    degenerate_seeds = [s for s, n in seed_ntopics.items() if n <= 5]
    non_degenerate_seeds = [s for s, n in seed_ntopics.items() if n > 5]
    print(f"  R3: {len(seed_ntopics)} seeds, "
          f"{len(degenerate_seeds)} degenerate, "
          f"{len(non_degenerate_seeds)} non-degenerate")
    print(f"    degenerate seeds:     {degenerate_seeds}")
    print(f"    non-degenerate seeds: {non_degenerate_seeds}")

    # R3* (conditional): renormalise raw R3 persistence to the non-degenerate denominator
    n_total = len(seed_ntopics)
    n_nondeg = max(1, len(non_degenerate_seeds))
    # raw R3 = matches/10. matches happen only in non-degenerate runs (degenerate has 0–5 topics ≪ baseline)
    # so R3* = raw_R3 * (n_total / n_nondeg)  (i.e., divide by smaller denominator)
    df["R3star_persistence"] = (df["R3_persistence"] * n_total / n_nondeg).clip(upper=1.0)

    # Multi-τ table
    rows = []
    for tau in THRESHOLDS:
        r1_pass = (df["R1_persistence"] >= tau).sum()
        r2_pass = (df["R2_persistence"] >= tau).sum()
        r3_pass = (df["R3_persistence"] >= tau).sum()
        r3star_pass = (df["R3star_persistence"] >= tau).sum()
        all_pass = (
            (df["R1_persistence"] >= tau) &
            (df["R2_persistence"] >= tau) &
            (df["R3_persistence"] >= tau)
        ).sum()
        all_pass_with_R3star = (
            (df["R1_persistence"] >= tau) &
            (df["R2_persistence"] >= tau) &
            (df["R3star_persistence"] >= tau)
        ).sum()
        rows.append({
            "tau": tau,
            "R1_pass": int(r1_pass),
            "R2_pass": int(r2_pass),
            "R3_raw_pass": int(r3_pass),
            "R3star_pass": int(r3star_pass),
            "all_three_R3_raw": int(all_pass),
            "all_three_R3star": int(all_pass_with_R3star),
            "pre_commit": tau == PRE_COMMIT_TAU,
        })
    table = pd.DataFrame(rows)
    table.to_csv(OUT_DIR / "multi_tau_table.csv", index=False)

    # Optionally add psychometric reliability bands as columns
    psych_summary = None
    if PSYCH.exists():
        psych = pd.read_csv(PSYCH)
        # Merge on topic_id
        merged = df.merge(psych[["topic_id", "cronbach_alpha", "mean_pairwise_kappa"]],
                          on="topic_id", how="left")
        merged.to_csv(OUT_DIR / "persistence_plus_psychometric.csv", index=False)

        # Cross-band reporting
        psych_summary = {
            "topics_with_alpha_ge_0.7": int((merged["cronbach_alpha"] >= 0.7).sum()),
            "topics_with_alpha_ge_0.8": int((merged["cronbach_alpha"] >= 0.8).sum()),
            "topics_with_alpha_ge_0.9": int((merged["cronbach_alpha"] >= 0.9).sum()),
            "topics_with_R1_R2_R3star_at_tau0.7_AND_alpha_0.7": int((
                (merged["R1_persistence"] >= 0.7) &
                (merged["R2_persistence"] >= 0.7) &
                (merged["R3star_persistence"] >= 0.7) &
                (merged["cronbach_alpha"] >= 0.7)
            ).sum()),
            "topics_with_R1_R2_R3star_at_tau0.5_AND_alpha_0.7": int((
                (merged["R1_persistence"] >= 0.5) &
                (merged["R2_persistence"] >= 0.5) &
                (merged["R3star_persistence"] >= 0.5) &
                (merged["cronbach_alpha"] >= 0.7)
            ).sum()),
        }

    summary = {
        "K_baseline_topics": int(K),
        "pre_commit_tau": PRE_COMMIT_TAU,
        "thresholds": THRESHOLDS,
        "R3_degenerate_seeds": degenerate_seeds,
        "R3_non_degenerate_seeds": non_degenerate_seeds,
        "R3_ceiling_raw": round(n_nondeg / n_total, 4),
        "table": rows,
        "psychometric_cross_check": psych_summary,
    }
    (OUT_DIR / "multi_tau_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n" + "=" * 80)
    print("Multi-τ sensitivity table (rows = τ, columns = axes)")
    print("=" * 80)
    print(f"  Pre-committed τ = {PRE_COMMIT_TAU} (Greene 2014)")
    print(f"  R3 ceiling: max raw persistence on R3 = {n_nondeg}/{n_total} = "
          f"{n_nondeg/n_total:.2f} (due to {len(degenerate_seeds)}/{n_total} "
          f"degenerate UMAP seeds)")
    print(f"  R3* renormalises to non-degenerate denominator (n={n_nondeg})")
    print()
    print(f"  {'τ':>5s}  {'R1':>5s}  {'R2':>5s}  {'R3raw':>6s}  {'R3*':>5s}  "
          f"{'all(R3raw)':>10s}  {'all(R3*)':>9s}  {'pre-commit':>10s}")
    for r in rows:
        flag = "  ★" if r["pre_commit"] else ""
        print(f"  {r['tau']:>5.2f}  {r['R1_pass']:>5d}  {r['R2_pass']:>5d}  "
              f"{r['R3_raw_pass']:>6d}  {r['R3star_pass']:>5d}  "
              f"{r['all_three_R3_raw']:>10d}  {r['all_three_R3star']:>9d}{flag}")

    if psych_summary:
        print()
        print(f"  Cross-check with psychometric reliability:")
        for k, v in psych_summary.items():
            print(f"    {k}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()
