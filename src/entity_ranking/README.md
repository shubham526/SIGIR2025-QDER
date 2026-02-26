# Train Entity Ranking Model (Step 1.4)

## Overview

Fine-tune MonoBERT for pointwise entity ranking using 5-fold cross-validation. The model learns to distinguish entities semantically relevant to a query from non-relevant ones, based on their DBpedia descriptions. The resulting entity rankings guide document filtering in Stage 2.

The codebase supports two model variants via `--task`: **MonoBERT** (classification, pointwise) for entity ranking in this step, and **DuoBERT** (ranking, pairwise) for document ranking in Step 2.3.

| | MonoBERT | DuoBERT |
|:--|:--|:--|
| `--task` | `classification` | `ranking` |
| Training approach | Single query-doc pairs with binary labels | Query + positive/negative doc pairs |
| Head | `Linear(768 → 2)` | `Linear(768 → 1)` |
| Loss | CrossEntropyLoss | MarginRankingLoss (margin=1) |
| Inference score | Softmax prob of class 1 | Raw scalar |
| Use case | Entity ranking (this step) | Document re-ranking (Step 2.3) |

---

## Required Inputs

1. **Training Data** (`entity_train.jsonl`): From Step 1.3
2. **Validation Data** (`entity_test.jsonl`): From Step 1.3
3. **Entity QRELs** (`entity_qrels.txt`): From Step 1.1

---

## Data Format

### Classification Task — MonoBERT (this step)

Each line in `entity_train.jsonl` / `entity_test.jsonl` must be a JSON object:

```json
{
  "query_id": "301",
  "query": "International organized crime",
  "doc_id": "Q30",
  "doc": "The United States of America is a country primarily located in North America...",
  "label": 1
}
```

| Field | Type | Description |
|:------|:-----|:------------|
| `query_id` | string | Query identifier |
| `query` | string | Query text |
| `doc_id` | string | Entity identifier (e.g., Wikidata/DBpedia ID) |
| `doc` | string | Entity description text — encoded as `[CLS] query [SEP] entity_description [SEP]` |
| `label` | int | Relevance label (1 = relevant, 0 = non-relevant) |

**Notes**:
- Data from Step 1.2 is already 1:1 balanced (equal positives/negatives per query)
- Training data contains only entities that appear exclusively in relevant OR non-relevant documents (shared entities are filtered in Step 1.1)
- Both `"doc"` and `"doc_text"` field names are supported for the entity description

### Ranking Task — DuoBERT (Step 2.3)

Training data requires positive/negative pairs per query:

```json
{
  "query_id": "301",
  "query": "International organized crime",
  "doc_id": "d1",
  "doc_pos_text": "Relevant document text...",
  "doc_neg_text": "Non-relevant document text..."
}
```

Test/dev data for both tasks uses the same pointwise format as the classification training data above.

---

## Model Architecture

Both models share the same BERT encoder base (`model.py`), differing only in the output head:

1. **Input**: `[CLS] query [SEP] document_text [SEP]`
2. **Encoder**: BERT-base-uncased (or specified via `--query-enc`)
3. **Pooling**: `[CLS]` token embedding (768-dim) if `--mode cls`, or pooled output if `--mode pooling`
4. **MonoBERT head**: `Linear(768 → 2)` — logits for [non-relevant, relevant]; inference score = softmax prob of class 1
5. **DuoBERT head**: `Linear(768 → 1)` — single relevance score; inference score = raw scalar

---

## Training

### Single Fold

```bash
python train.py \
  --train data/entity_folds/fold-0/entity_train.jsonl \
  --dev data/entity_folds/fold-0/entity_test.jsonl \
  --qrels data/robust04/entity_qrels.txt \
  --save-dir models/entity_ranker/fold-0 \
  --save model.bin \
  --run dev_entity_run.txt \
  --task classification \
  --max-len 512 \
  --query-enc bert \
  --mode cls \
  --epoch 10 \
  --batch-size 32 \
  --learning-rate 2e-5 \
  --n-warmup-steps 1000 \
  --metric map \
  --eval-every 1 \
  --num-workers 4 \
  --use-cuda \
  --cuda 0
```

### All 5 Folds

Save as `train_entity_ranker_all_folds.sh` and run with `chmod +x train_entity_ranker_all_folds.sh && ./train_entity_ranker_all_folds.sh`:

```bash
#!/bin/bash

DATASET="robust04"
DATA_DIR="data/${DATASET}/entity_folds"
QRELS="data/${DATASET}/entity_qrels.txt"
SAVE_ROOT="models/entity_ranker/${DATASET}"
RUNS_ROOT="runs/entity_ranker/${DATASET}"

EPOCHS=10
BATCH_SIZE=32
LR=2e-5
WARMUP=1000
MAX_LEN=512
QUERY_ENC="bert"
MODE="cls"

mkdir -p ${SAVE_ROOT}
mkdir -p ${RUNS_ROOT}

for fold in {0..4}; do
    echo "========================================"
    echo "Training Entity Ranker - Fold ${fold}"
    echo "========================================"

    python train.py \
        --train ${DATA_DIR}/fold-${fold}/entity_train.jsonl \
        --dev ${DATA_DIR}/fold-${fold}/entity_test.jsonl \
        --qrels ${QRELS} \
        --save-dir ${SAVE_ROOT}/fold-${fold} \
        --save model.bin \
        --run dev_entity_run_fold${fold}.txt \
        --task classification \
        --max-len ${MAX_LEN} \
        --query-enc ${QUERY_ENC} \
        --mode ${MODE} \
        --epoch ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --learning-rate ${LR} \
        --n-warmup-steps ${WARMUP} \
        --metric map \
        --eval-every 1 \
        --num-workers 4 \
        --use-cuda \
        --cuda 0

    echo "Fold ${fold} complete."
done

echo "Entity Ranker 5-Fold Cross-Validation Complete!"
```

---

## Command-Line Arguments

### Required

| Argument | Description |
|:---------|:------------|
| `--train` | Path to training data (JSONL) |
| `--dev` | Path to validation data (JSONL) |
| `--qrels` | Entity-level ground truth (TREC format) |
| `--save-dir` | Directory to save model checkpoints |

### Model Configuration

| Argument | Default | Options |
|:---------|:--------|:--------|
| `--task` | `classification` | `classification` (MonoBERT), `ranking` (DuoBERT) |
| `--query-enc` | `bert` | `bert`, `roberta`, `deberta`, `distilbert`, `electra`, `conv-bert`, `t5`, `ernie` |
| `--mode` | `cls` | `cls`, `pooling` |
| `--max-len` | `512` | Any integer ≤ 512 |

**Model mapping**:
- `bert` → `bert-base-uncased`
- `roberta` → `roberta-base`
- `deberta` → `microsoft/deberta-base`
- `distilbert` → `distilbert-base-uncased`
- `electra` → `google/electra-small-discriminator`
- `conv-bert` → `YituTech/conv-bert-base`
- `t5` → `t5-base`
- `ernie` → `nghuyong/ernie-2.0-base-en`

### Training Hyperparameters

| Argument | Default | Recommended |
|:---------|:--------|:------------|
| `--epoch` | `20` | `10–20` |
| `--batch-size` | `8` | `16–40` |
| `--learning-rate` | `2e-5` | `1e-5` to `3e-5` |
| `--n-warmup-steps` | `2` | `500–1000` |

### Evaluation

| Argument | Default | Options |
|:---------|:--------|:--------|
| `--metric` | `map` | `map`, `ndcg`, `ndcg_cut_20`, `P_20`, `mrr_cut_10` |
| `--eval-every` | `1` | Any positive integer |

### Output & System

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--save` | `model.bin` | Checkpoint filename |
| `--run` | `dev.run` | Validation run filename |
| `--checkpoint` | `None` | Path to checkpoint to resume from |
| `--use-cuda` | `False` | Enable GPU training |
| `--cuda` | `0` | CUDA device index |
| `--num-workers` | `0` | DataLoader workers |

---

## Inference

After training, generate entity rankings on test data using `test.py`. Use the same `--task`, `--query-enc`, and `--mode` as during training — MonoBERT and DuoBERT checkpoints are not interchangeable.

### Single Fold

```bash
python test.py \
  --test data/entity_folds/fold-0/entity_test.jsonl \
  --checkpoint models/entity_ranker/fold-0/model.bin \
  --run runs/entity_run_fold0.txt \
  --task classification \
  --max-len 512 \
  --query-enc bert \
  --mode cls \
  --batch-size 32 \
  --use-cuda \
  --cuda 0
```

### All 5 Folds

```bash
#!/bin/bash

DATASET="robust04"
DATA_DIR="data/${DATASET}/entity_folds"
MODEL_DIR="models/entity_ranker/${DATASET}"
RUNS_DIR="runs/entity_ranker/${DATASET}"

mkdir -p ${RUNS_DIR}

for fold in {0..4}; do
    echo "Generating entity rankings for fold ${fold}..."

    python test.py \
        --test ${DATA_DIR}/fold-${fold}/entity_test.jsonl \
        --checkpoint ${MODEL_DIR}/fold-${fold}/model.bin \
        --run ${RUNS_DIR}/entity_run_fold${fold}.txt \
        --task classification \
        --max-len 512 \
        --query-enc bert \
        --mode cls \
        --batch-size 32 \
        --use-cuda \
        --cuda 0

    echo "Fold ${fold} complete."
done

echo "All entity rankings generated!"
```

---

## Output Files

```
models/entity_ranker/robust04/fold-0/
├── model.bin                    # Best checkpoint
├── dev_entity_run_fold0.txt    # Dev run from best epoch
└── config.json                  # Model configuration

runs/entity_ranker/robust04/
└── entity_run_fold0.txt        # Test set entity rankings
```

The run files follow standard TREC format: `query_id Q0 entity_id rank score run_name`.

---

## Common Issues

**`KeyError: 'doc_text'`** — Your data uses `"doc"` as the field name. Both are supported; the script falls back automatically via `example.get('doc_text', example.get('doc', ''))`.

**Unexpected key warnings on model load** — Warnings about keys like `cls.predictions.transform.*` are normal. These are pre-training head weights not used in classification and can be safely ignored.

---

## Using Entity Rankings in Stage 2

The run files from this step feed directly into Stage 2 document ranking:

```bash
python make_doc_ranking_data.py \
  --entity_run runs/entity_ranker/robust04/entity_run_fold0.txt \
  --top_k 20 \
  ...
```

The top-20 entities per query are used to filter documents: only documents containing at least one top-K entity are retained as candidates. This entity-guided filtering is the core of the QDER pipeline.

---

## References

- [MonoBERT and DuoBERT (Nogueira et al., 2019)](https://arxiv.org/abs/1901.04085)