# QDER: Query-Specific Document and Entity Re-Ranking

## Overview

QDER is a dual-channel neural re-ranking model that combines text and entity representations to re-rank documents. It uses query-specific attention, interaction modeling (Addition + Multiplication), and bilinear scoring.

**Paper configuration ("No-Subtract"):** Addition + Multiplication interactions only, with bilinear scoring.

---

## Module Structure

| File | Description |
|------|-------------|
| `train.py` | Main training script |
| `test.py` | Inference and evaluation script |
| `qder_model.py` | Core QDER model architecture |
| `base_model.py` | Abstract base class for QDER models |
| `dataset.py` | Data loading for JSONL files with pre-computed embeddings |
| `text_embedding.py` | Hugging Face transformer wrapper (BERT, RoBERTa, etc.) |
| `trainer.py` | Training loop utility |

---

## Prerequisites

```bash
pip install torch transformers tqdm numpy pytrec_eval
```

---

## Data Format

Each line in the JSONL file must be a valid JSON object:

```json
{
  "query_id": "301",
  "query": "International organized crime",
  "doc_id": "FBIS3-1",
  "doc": "Full document text here...",
  "query_ent_emb": [[0.12, 0.45, ...], ...],
  "doc_ent_emb": [[0.34, 0.67, ...], ...],
  "label": 1,
  "doc_score": 234.5
}
```

**Required Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `query_id` | str | Unique query identifier |
| `doc_id` | str | Unique document identifier |
| `query` | str | Raw query text |
| `doc` | str | Raw document/passage text |
| `query_ent_emb` | List[List[float]] | Query entity vectors — shape `[num_entities, 300]` |
| `doc_ent_emb` | List[List[float]] | Document entity vectors — shape `[num_entities, 300]` |
| `label` | int | Relevance label (1 = relevant, 0 = non-relevant) |
| `doc_score` | float | Initial retrieval score (e.g., BM25) used for weighting |

> **Note on entities:** Uses Wikipedia2Vec (300-dimensional) embeddings. If a query or document has no entities, pass an empty list `[]`; the code handles padding automatically.

---

## Training

### Basic Training Command

```bash
python train.py \
    --train data/doc_folds/fold-0/doc_train.jsonl \
    --dev data/doc_folds/fold-0/doc_test.jsonl \
    --qrels data/robust04/qrels.txt \
    --save-dir models/fold-0 \
    --save model.bin \
    --text-enc bert \
    --score-method bilinear \
    --enabled-interactions "add,multiply" \
    --epoch 20 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --n-warmup-steps 1000 \
    --metric map \
    --eval-every 1 \
    --num-workers 4 \
    --use-cuda \
    --cuda 0 \
    --run dev_run.txt
```

### All Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--train` | Path to training data (JSONL) | *(required)* |
| `--dev` | Path to validation data (JSONL) | *(required)* |
| `--qrels` | Ground truth file (TREC format) | *(required)* |
| `--save-dir` | Directory to save model checkpoints | *(required)* |
| `--save` | Checkpoint filename | `model.bin` |
| `--checkpoint` | Path to checkpoint to resume from | `None` |
| `--text-enc` | Text encoder backbone | `bert` |
| `--score-method` | Scoring method (`linear` or `bilinear`) | `bilinear` |
| `--enabled-interactions` | Comma-separated interaction list | `add,multiply` |
| `--max-len` | Maximum input sequence length | `512` |
| `--epoch` | Number of training epochs | `20` |
| `--batch-size` | Batch size | `8` |
| `--learning-rate` | Adam learning rate | `1e-5` |
| `--n-warmup-steps` | Linear warmup steps | `1000` |
| `--patience` | Early stopping patience (epochs) | `5` |
| `--metric` | Validation metric | `map` |
| `--eval-every` | Evaluate every N epochs | `1` |
| `--num-workers` | DataLoader workers | `0` |
| `--use-cuda` | Enable GPU training | flag |
| `--cuda` | CUDA device index | `0` |
| `--run` | Output dev run filename | `dev.run` |

**Supported `--text-enc` values:** `bert`, `bert-large`, `distilbert`, `roberta`, `roberta-large`, `deberta`, `deberta-large`, `ernie`, `electra`, `electra-base`, `conv-bert`, `t5`, `t5-large`

**Supported `--metric` values:** `map`, `ndcg`, `ndcg_cut_20`, `P_20`

### Resuming Training

Checkpoints saved by `train.py` include the full training state (model weights, optimizer, scheduler, epoch, and best metric), so training can be resumed seamlessly.

```bash
# Resume from interruption
python train.py \
    --train data/train.jsonl \
    --dev data/dev.jsonl \
    --qrels data/qrels.tsv \
    --save-dir checkpoints/ \
    --checkpoint checkpoints/model.bin \
    --text-enc bert \
    --score-method bilinear \
    --enabled-interactions "add,multiply" \
    --epoch 30 \
    --learning-rate 1e-5 \
    --use-cuda

# Fine-tune with lower learning rate
python train.py \
    --train data/train.jsonl \
    --dev data/dev.jsonl \
    --qrels data/qrels.tsv \
    --save-dir checkpoints_finetune/ \
    --checkpoint checkpoints/model.bin \
    --text-enc bert \
    --score-method bilinear \
    --enabled-interactions "add,multiply" \
    --epoch 10 \
    --learning-rate 5e-6 \
    --use-cuda
```

---

## Inference / Testing

```bash
python test.py \
    --test data/test.jsonl \
    --qrels data/qrels.test.tsv \
    --checkpoint checkpoints/model.bin \
    --output results/test.run \
    --text-enc bert \
    --score-method bilinear \
    --enabled-interactions "add,multiply" \
    --metrics "map,ndcg" \
    --use-cuda
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--test` | Test data (JSONL) | *(required)* |
| `--qrels` | Ground truth file (TREC format) | *(required)* |
| `--checkpoint` | Path to saved model checkpoint | *(required)* |
| `--output` | Output run file (TREC format) | `test.run` |
| `--text-enc` | Text encoder backbone | `bert` |
| `--score-method` | Scoring method | `bilinear` |
| `--enabled-interactions` | Comma-separated interaction list | `add,multiply` |
| `--max-len` | Maximum input sequence length | `512` |
| `--batch-size` | Batch size | `16` |
| `--metrics` | Metrics to report (comma-separated) | `map,ndcg` |
| `--use-cuda` | Enable GPU | flag |
| `--cuda` | CUDA device index | `0` |

---

## Ablation Studies

Reproduce the interaction ablations from Table 4 of the paper using `--enabled-interactions`:

| Configuration | Argument |
|---------------|----------|
| **No-Subtract (paper best)** | `--enabled-interactions "add,multiply"` |
| All interactions | `--enabled-interactions "add,multiply,subtract"` |
| Addition only | `--enabled-interactions "add"` |
| Multiplication only | `--enabled-interactions "multiply"` |

---

## 5-Fold Cross-Validation

```bash
#!/bin/bash

DATASET="robust04"
DATA_DIR="data/${DATASET}/doc_folds"
QRELS="data/${DATASET}/qrels.txt"
SAVE_ROOT="models/${DATASET}"

mkdir -p ${SAVE_ROOT}

for fold in {0..4}; do
    echo "========================================"
    echo "Training Fold ${fold}"
    echo "========================================"

    python train.py \
        --train ${DATA_DIR}/fold-${fold}/doc_train.jsonl \
        --dev ${DATA_DIR}/fold-${fold}/doc_test.jsonl \
        --qrels ${QRELS} \
        --save-dir ${SAVE_ROOT}/fold-${fold} \
        --save model.bin \
        --text-enc bert \
        --score-method bilinear \
        --enabled-interactions "add,multiply" \
        --epoch 20 \
        --batch-size 8 \
        --learning-rate 1e-5 \
        --n-warmup-steps 1000 \
        --metric map \
        --eval-every 1 \
        --num-workers 4 \
        --use-cuda \
        --cuda 0 \
        --run dev_run_fold${fold}.txt

    echo "Fold ${fold} complete."
done

echo "5-Fold Cross-Validation Complete!"
```

Save as `train_all_folds.sh` and run:
```bash
chmod +x train_all_folds.sh
./train_all_folds.sh
```

---

## Output Files

After training, the following files are saved in `--save-dir`:

```
models/robust04/fold-0/
├── model.bin                      # Best checkpoint (full training state)
├── best_model_epoch_015.bin       # Best checkpoint with epoch number
├── epoch_001_dev_run.txt          # Per-epoch validation run files
├── epoch_002_dev_run.txt
├── ...
├── best_dev_run.txt               # Run file from best epoch
└── training_history.json         # Loss and metric history per epoch
```

The `training_history.json` records train loss and validation metric per epoch, the best epoch, and the best metric achieved — useful for plotting training curves.

### TREC Run File Format

```
301 Q0 FBIS3-1 1 0.9234 BERT
301 Q0 FBIS3-5 2 0.8876 BERT
301 Q0 FBIS3-12 3 0.7654 BERT
```

Columns: Query ID, Q0 (constant), Document ID, Rank, Score, Run name.

---

## Model Architecture Details

### Components

The model processes query–document pairs through two parallel channels:

**Text Channel:** Query and document text are independently encoded with BERT. Cross-attention between query and document token embeddings produces a weighted document representation. Interaction features (addition and/or multiplication) are computed, then mean-pooled with masking over padding tokens.

**Entity Channel:** Query and document entity embeddings (Wikipedia2Vec, 300-dim) are processed with cross-attention. The same interaction operations are applied and mean-pooled with entity padding masks.

The pooled text and entity features are concatenated, normalized with LayerNorm, and passed to the scoring layer.

### Scoring

The default (and recommended) scoring method is **bilinear**: `score = x^T · M · x` where `M` is a learned matrix. Linear scoring (`score = W · x + b`) is also available but performs worse.

### Training Details

| Detail | Value |
|--------|-------|
| Optimizer | Adam |
| Loss | Binary Cross-Entropy with Logits |
| LR schedule | Linear warmup + decay |
| Gradient clipping | Max norm 1.0 |
| Early stopping | Patience of 5 epochs |
| Default LR | `1e-5` |
| Default warmup steps | `1000` |

---

## Expected Performance (TREC Robust04, Title Queries)

| Metric | BM25+RM3 | QDER (Expected) |
|--------|----------|-----------------|
| MAP | 0.291 | ~0.608 |
| nDCG@20 | 0.435 | ~0.769 |
| P@20 | 0.384 | ~0.736 |

Expected per-fold variance: MAP ≈ 0.598–0.615 across folds.

If any fold shows MAP < 0.50, check that: entity embeddings are correctly loaded, bilinear scoring is enabled (`--score-method bilinear`), and subtraction is not included in `--enabled-interactions`.

---

## Generating Final Rankings After 5-Fold CV

```bash
for fold in {0..4}; do
    python test.py \
        --test data/robust04/doc_folds/fold-${fold}/doc_test.jsonl \
        --qrels data/robust04/qrels.txt \
        --checkpoint models/robust04/fold-${fold}/model.bin \
        --output runs/robust04/test_run_fold${fold}.txt \
        --text-enc bert \
        --score-method bilinear \
        --enabled-interactions "add,multiply" \
        --use-cuda
done

# Combine fold results
cat runs/robust04/test_run_fold*.txt > runs/robust04/qder_final.txt

# Evaluate
trec_eval -c -m map -m ndcg_cut.20 -m P.20 \
    data/robust04/qrels.txt \
    runs/robust04/qder_final.txt
```

---

## Troubleshooting

**CUDA out of memory:** Reduce `--batch-size` (try 4) or `--max-len` (try 256).

**Low validation scores (MAP < 0.35):** Verify `--score-method bilinear`, confirm `--enabled-interactions` does not include `subtract`, and check that entity embeddings are present and correctly formatted in the JSONL data.

**NaN/Inf in training:** Gradient clipping is applied automatically (max norm 1.0). If NaN batches persist, try a lower learning rate (e.g., `5e-6`).

**Validation takes too long:** Use `--eval-every 2` to evaluate every other epoch instead of every epoch.