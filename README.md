# Prosepoem Stable Topics

Eight-layer reliability audit of Korean conservative-definition prose poetry (4,642 works from four major publishers, 2000–2024). BERTopic-based topic identification with cross-validation across parameter, sample, seed, and membership perturbations.

## Pipeline

```
SBERT (jhgan/ko-sroberta-multitask)
  → UMAP (n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine)
  → HDBSCAN (min_cluster_size=30, min_samples=10)
  → c-TF-IDF
```

Anchor: seed=100, K=25, outlier=48.82%.

## Eight-Layer Audit

Four core axes (audit-robust set = topics passing all four):

| Axis | Perturbation | Reference |
|---|---|---|
| R1 | Parameter sweep (UMAP, HDBSCAN, CountVectorizer, OAT) | Saltelli 2008 |
| R2 | Bootstrap sample stability (80%, 10 runs) | Belford 2018 |
| R3 | UMAP seed cross-presence (32 non-degenerate baselines) | Greene 2014 |
| α  | Cronbach's α membership reliability | Schroeder & Wood-Doughty 2025 |

Four auxiliary diagnostics (cross-validate that audit-robust six are not method-specific):

- Sobol' interaction decomposition (Saltelli 2019) — covers OAT limits of R1
- Three negative controls (random partition α, keyword-matched Jaccard, HDBSCAN threshold ablation)
- KeyBERT alternative extraction (Grootendorst 2020) — verifies motif identity is method-independent
- Coherence baselines (NPMI, c_v, UMass, c_uci)

NNLS-based document–motif distribution (Lawson & Hanson 1974) yields per-document soft mixture and unexplained share. Hard outlier (BERTopic) and soft unexplained (NNLS) comparison diagnoses whether outliers are a method artefact or corpus property.

## Audit-robust six motifs

T0 sea, T1 flora, T7 rain, T11 meta-poetic, T14 music, T16 cat. Corpus mass share: 13.9%. Mean unexplained: 48.8%, converging with hard outlier rate 48.82%.

## Project Structure

```
prosepoem-stable-topics/
├── README.md                         (this file)
├── references.bib                    (BibTeX entries for all cited works)
├── data/
│   ├── raw/                          (NOT shared — see Data Sharing below)
│   ├── filtered/                     (NOT shared — contains 원문 column)
│   ├── tokenized/                    (NOT shared — POS-filtered lemmas)
│   └── shareable/                    (sharable subset; see Data Sharing)
├── scripts/                          (numbered pipeline)
├── figures/                          (figure generation scripts)
├── results/
│   ├── baseline/                     (seed=100 anchor results)
│   ├── audit/                        (R1·R2·R3·α + auxiliary diagnostics)
│   ├── distribution/                 (NNLS doc-motif distribution)
│   ├── poet_specialty/               (per-poet motif aggregation)
│   └── representative/               (top documents per motif)
├── stopwords/
│   └── hand_stopwords_reference.txt  (reference only; not used in pipeline)
├── paper/                            (manuscript, figures plan, English/Korean drafts)
└── references_pdfs/                  (downloaded reference PDFs)
```

## Data Sharing

### Excluded from public release (`data/raw/`, `data/filtered/`, `data/tokenized/`)

Korean poetry full text is under the copyright of poets and publishers. The following are **not** redistributed:

| Path | Reason |
|---|---|
| `data/raw/sentences.parquet` | full sentence text |
| `data/raw/lemmas.parquet` | morpheme sequences (allow partial text reconstruction) |
| `data/raw/sent_embeddings.npz` | sentence embeddings (allow nearest-neighbor reconstruction) |
| `data/raw/metadata.csv` | contains full-text column |
| `data/filtered/works_filtered.parquet` | contains `원문` (raw text) column |
| `data/filtered/sentences_filtered.parquet` | filtered sentence text |
| `data/tokenized/lemmas_filtered.parquet` | POS-filtered morpheme sequences |
| `data/tokenized/work_content_tokens.parquet` | per-work content-word sequences |

These files are required to re-run the pipeline from source. They are kept locally and not committed to the public repository.

### Shareable subset (`data/shareable/`)

| File | Rows | Description |
|---|---|---|
| `works_metadata.parquet` / `.csv` | 4,642 | Work-level metadata, **`원문` column removed**. Columns: `work_id, poet_id, 작가, 책 제목, 출판사, 시리즈, 번호, 출판년도, 작품 제목, 등단년도, 등단매체, 등단매체_정규화, 등단경로, 등단장르, 어절수` |
| `books_dedup.parquet` / `.csv` | 718 | Book-level dedup. One row per (poet × book × pub-year), with `n_works_in_book` count |
| `poets_summary.parquet` / `.csv` | 399 | Poet-level summary: total works, books, publication-year range, debut year |

These files allow reproduction of:
- Corpus filtering counts (length filter, deduplication)
- Per-poet `n_works` and the `n ≥ 16` specialty cutoff
- Per-book / per-publisher distribution statistics

These files do **not** allow reproduction of:
- BERTopic results (require embeddings)
- NNLS document–motif distribution (require embeddings)
- KeyBERT extraction (require text)

For the topic-modeling outputs (BERTopic anchor results, audit results, NNLS distributions, KeyBERT extractions), see `results/` — these are derivatives and can be shared without the underlying text.

### Results

| Path | Content | Sharable |
|---|---|---|
| `results/baseline/` | seed=100 anchor topics, top words, c-TF-IDF | Yes |
| `results/audit/` | R1·R2·R3·α + four auxiliary results | Yes |
| `results/distribution/` | NNLS `doc_motif_distribution.csv` (percentages only) | Yes |
| `results/poet_specialty/` | per-poet motif mean + specialty score | Yes |
| `results/representative/by_motif/T*_top15.json` | top-15 documents per motif | Contains text excerpts; review per fair-use scope before sharing |
| `results/representative/all_motifs_top10.md` | top-10 per motif markdown summary | Contains text excerpts; review before sharing |

For sharing on JCA Dataverse: include `data/shareable/`, `results/baseline/`, `results/audit/`, `results/distribution/`, `results/poet_specialty/`, all `scripts/`, and `references.bib`. Exclude `data/raw/`, `data/filtered/`, `data/tokenized/`, and the text-containing representative files.

## How to Run

Requires the local (non-shared) `data/raw/` and `data/filtered/` files.

1. Preprocessing — produces `data/tokenized/work_content_tokens.parquet`:
   ```
   python scripts/01_preprocess.py
   ```
2. Baseline BERTopic — seed=100:
   ```
   python scripts/02_baseline.py --seed 100
   ```
3. Audit (R1, R2, R3, α) — produces `results/audit/`:
   ```
   python scripts/03_audit.py
   ```
4. Auxiliary diagnostics (Sobol', negative controls, KeyBERT, coherence):
   ```
   python scripts/04_auxiliary.py
   ```
5. NNLS doc–motif distribution:
   ```
   python scripts/05_nnls_distribution.py
   ```
6. Poet specialty:
   ```
   python scripts/06_poet_specialty.py
   ```
7. Figures:
   ```
   python figures/01_baseline_distribution.py
   python figures/02_axis_heatmap.py
   python figures/03_nnls_distribution.py
   python figures/04_group_dispersion.py
   python figures/05_poet_specialty_heatmap.py
   ```

## Environment

- Python 3.10+
- `pip install bertopic sentence-transformers umap-learn hdbscan scikit-learn pandas pyarrow keybert matplotlib SALib`
- Korean NLP: `mecab-ko`, `kss`
- Embeddings: `jhgan/ko-sroberta-multitask` (loaded via Hugging Face)

## References

See `references.bib` for the full bibliography. Method-relevant references are cited inline in the audit table above.
