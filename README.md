# QDER: Query-Specific Document and Entity Representations for Multi-Vector Document Re-Ranking

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data License: CC BY-SA 4.0](https://img.shields.io/badge/Data%20License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![SIGIR 2025](https://img.shields.io/badge/SIGIR-2025-red.svg)](https://doi.org/10.1145/3726302.3730065)

**Paper:** Shubham Chatterjee and Jeff Dalton. *QDER: Query-Specific Document and Entity Representations for Multi-Vector Document Re-Ranking.* SIGIR 2025.  
📄 [ACM DL](https://doi.org/10.1145/3726302.3730065) · 📦 [Data](#data) · 📖 [Wiki](../../wiki)

---

> ⚠️ **Note on Reproducibility**
> 
> An earlier version of the code and data uploaded to this repository contained an error that affected reproducibility. This has since been identified and corrected. The current code and data on this repository are correct and should reproduce the results reported in the paper. We encourage users to use the current code and data and to open a GitHub issue if they encounter any difficulties reproducing the results.

## Overview

QDER is a dual-channel neural re-ranking model that integrates knowledge graph semantics into a multi-vector ranking framework. Rather than computing similarity on aggregated embeddings, QDER maintains individual token and entity representations throughout the ranking process — an approach we call **late aggregation** — transforming them through query-specific attention before combining via learned interaction operations (addition + multiplication) and bilinear scoring. The final score interpolates QDER's output with BM25 using Coordinate Ascent.

```
                    ┌─────────────────────────────────────┐
                    │           Text Channel               │
 Query text  ──────►│  BERT Encoder → Cross-Attention      │──► Add/Multiply ──► Mean Pool ──┐
 Doc text    ──────►│  BERT Encoder ─────────────────────  │                                 │
                    └─────────────────────────────────────┘                                 │
                                                                                             ├──► LayerNorm ──► Bilinear Score ──► λ·BM25 + (1-λ)·QDER
                    ┌─────────────────────────────────────┐                                 │
                    │          Entity Channel              │                                 │
 Query entities ───►│  Wiki2Vec Embeddings → Cross-Attn   │──► Add/Multiply ──► Mean Pool ──┘
 Doc entities  ────►│  Wiki2Vec Embeddings ──────────────  │
                    └─────────────────────────────────────┘
```

The full pipeline runs in two stages:

1. **Entity Ranking** — a MonoBERT classifier identifies query-relevant entities using DBpedia descriptions, producing a ranked entity list per query. The top-K entities are used to filter document candidates.
2. **Document Re-Ranking** — QDER re-ranks the filtered candidates using both text (BERT) and entity (Wikipedia2Vec) channels with query-specific cross-attention and bilinear interaction scoring.

---

## Results

Performance on TREC Robust04 (title queries), 5-fold cross-validation:

| Model | MAP | nDCG@20 | P@20 | MRR |
|-------|-----|---------|------|-----|
| BM25 | 0.2915 | 0.4354 | 0.3839 | 0.6693 |
| CEDR (strongest baseline) | 0.3701 | 0.5475 | 0.4769 | 0.7879 |
| **QDER** | **0.6082** | **0.7694** | **0.7361** | **0.9751** |

QDER achieves a **36% improvement in nDCG@20** over CEDR (the strongest baseline) and a **70% improvement** over the initial BM25+RM3 candidate set. On the most difficult queries (where BM25+RM3 scores nDCG@20 = 0.0), QDER achieves **nDCG@20 = 0.70**. Results are consistent across all five benchmarks evaluated.

---

## Repository Structure

```
SIGIR2025-QDER/
├── scripts/
│   ├── data_preparation/       # Data prep scripts (Steps 1.1–1.3, 2.1)
│   ├── entity_ranker/          # MonoBERT training and inference (Step 1.4)
│   └── doc_ranker/             # QDER training and inference (Step 2.2–2.3)
├── data/                       # Placeholder — see Data section below
├── models/                     # Saved checkpoints (created at runtime)
├── runs/                       # TREC-format run files (created at runtime)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/shubham526/SIGIR2025-QDER.git
cd SIGIR2025-QDER
python -m venv qder_env
source qder_env/bin/activate
pip install -r requirements.txt
```

### 2. Download data

See the [Data](https://github.com/shubham526/SIGIR2025-QDER/wiki/Data) wiki page for download links and the expected directory layout.

### 3. Run the pipeline

Full step-by-step instructions are in the wiki:

- [Entity Ranking Pipeline](https://github.com/shubham526/SIGIR2025-QDER/wiki/Entity-Ranking-Pipeline) — data prep + MonoBERT training + inference
- [Document Ranking Pipeline](https://github.com/shubham526/SIGIR2025-QDER/wiki/Document-Ranking-Pipeline) — data prep + QDER training + inference
- [Reproducing Paper Results](https://github.com/shubham526/SIGIR2025-QDER/wiki/Reproducibilty) — combining folds and running `trec_eval`

---

## Data

All data associated with this work is released under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Download links and file descriptions are available on the [Data](../../wiki/Data) wiki page. The pipeline requires the following inputs:

| File | Description |
|------|-------------|
| `queries.tsv` | Query ID + text |
| `corpus.jsonl` | Documents with entity annotations |
| `bm25.run` | Initial BM25 retrieval results |
| `qrels.txt` | Relevance judgements (TREC format) |
| `entity_embeddings.jsonl` | Entity ID + Wikipedia2Vec embedding |

---

## Citation

```bibtex
@inproceedings{10.1145/3726302.3730065,
  author    = {Chatterjee, Shubham and Dalton, Jeff},
  title     = {QDER: Query-Specific Document and Entity Representations for Multi-Vector Document Re-Ranking},
  year      = {2025},
  publisher = {Association for Computing Machinery},
  url       = {https://doi.org/10.1145/3726302.3730065},
  doi       = {10.1145/3726302.3730065},
  booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {2255–2265},
  series    = {SIGIR '25}
}
```

---

## Acknowledgements

This material is based upon work supported by the Engineering and Physical Sciences Research Council (EPSRC) grant EP/V025708/1. Any opinions, findings, and conclusions expressed are those of the authors and do not necessarily reflect the views of the EPSRC.

---

## License

Code: [MIT License](LICENSE)  
Data: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

---

## Contact

Questions about the code or paper? Reach out to **Shubham Chatterjee** at shubham.chatterjee@mst.edu or open an issue.
