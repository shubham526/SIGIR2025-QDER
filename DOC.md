# QDER Data Preparation Pipeline

This README provides step-by-step instructions for preparing training and evaluation data for the QDER (Query-specific Document and Entity Representations) model.

## Overview

The pipeline consists of two main stages:

1. **Entity Ranking Pipeline**: Train a BERT-based entity ranker
2. **Document Ranking Pipeline**: Train the QDER document re-ranker using entity rankings

---

## Prerequisites

### Required Input Files

Before starting, ensure you have the following files:

1. **Document QRELs** (`doc_qrels.txt`): TREC-format relevance judgments
   ```
   <query_id> Q0 <doc_id> <relevance>
   ```
   Example:
   ```
   301 Q0 FBIS3-1 1
   301 Q0 FBIS3-2 0
   ```

2. **Entity-Linked Corpus** (`corpus.jsonl`): Documents with entity annotations
   ```json
   {"doc_id": "FBIS3-1", "entities": ["Q30", "Q1234", ...]}
   ```

3. **Entity Descriptions** (`entity_descriptions.jsonl`): DBpedia descriptions for entities
   ```json
   {"id": "Q30", "contents": "The United States of America is a country..."}
   ```

4. **Queries** (`queries.tsv`): Tab-separated query file
   ```
   301\tInternational organized crime
   302\tPolio eradication
   ```

5. **Fold Definitions** (`folds.json`): 5-fold cross-validation splits
   ```json
   {
     "0": {"training": ["301", "302", ...], "testing": ["350", "351", ...]},
     "1": {"training": [...], "testing": [...]},
     ...
   }
   ```

6. **Initial Retrieval Rankings** (`initial_ranking.txt`): BM25 or similar baseline
   ```
   301 Q0 FBIS3-1 1 234.5 BM25
   301 Q0 FBIS3-2 2 230.1 BM25
   ```

---

## Stage 1: Entity Ranking Pipeline

### Step 1.1: Create Entity QRELs

Generate entity-level relevance judgments from document-level judgments.

**Script**: `make_entity_ranking_qrels.py`

**Purpose**: Transfer relevance labels from documents to entities. An entity is labeled relevant if it appears in relevant documents, non-relevant if it appears in non-relevant documents. Entities appearing in both are excluded from training.

**Command**:
```bash
python make_entity_ranking_qrels.py \
  --qrels data/doc_qrels.txt \
  --docs data/corpus.jsonl \
  --save data/entity_qrels.txt
```

**Flags**:
- `--qrels`: Path to document-level TREC qrels file (REQUIRED)
- `--docs`: Path to entity-linked corpus in JSONL format (REQUIRED)
- `--save`: Output path for entity qrels file (REQUIRED)

**Input Requirements**:

1. **Document QRELs** (`--qrels`):
   - Format: `<query_id> Q0 <doc_id> <relevance>`
   - Relevance: `>=1` = relevant, `0` = non-relevant
   - One judgment per line

2. **Corpus** (`--docs`):
   - Format: JSONL (one JSON object per line)
   - Required fields:
     - `doc_id`: Document identifier (string)
     - `entities`: List of entity IDs (array of strings)
   - Example:
     ```json
     {"doc_id": "FBIS3-1", "entities": ["Q30", "Q1234", "Q5678"]}
     ```

**Output**:
- `entity_qrels.txt`: Entity-level relevance judgments in TREC format
  ```
  301 Q0 Q30 1
  301 Q0 Q1234 1
  301 Q0 Q5678 0
  ```

**Important Notes**:
- Entities appearing in BOTH relevant AND non-relevant documents are **excluded** from the entity QRELs
- This creates unambiguous training labels for the entity ranker
- This is a key architectural decision: we train the entity ranker only on discriminative entities

---

### Step 1.2: Create Entity Ranking Training Data

Convert entity QRELs into pointwise training examples for BERT.

**Script**: `make_entity_ranking_data.py`

**Purpose**: Create query-entity-description triples for training the BERT-based entity ranker. Applies 1:1 balancing between positive and negative examples per query.

**Command**:
```bash
python make_entity_ranking_data.py \
  --queries data/queries.tsv \
  --qrels data/entity_qrels.txt \
  --desc data/entity_descriptions.jsonl \
  --save data/entity_ranking_data.jsonl
```

**Flags**:
- `--queries`: Path to queries file (REQUIRED)
- `--qrels`: Path to entity QRELs from Step 1.1 (REQUIRED)
- `--desc`: Path to entity descriptions file (REQUIRED)
- `--save`: Output path for training data (REQUIRED)

**Input Requirements**:

1. **Queries** (`--queries`):
   - Format: Tab-separated values
   - Structure: `<query_id>\t<query_text>`
   - Example:
     ```
     301	International organized crime
     302	Polio and polio vaccination
     ```

2. **Entity QRELs** (`--qrels`):
   - Output from Step 1.1
   - Format: `<query_id> Q0 <entity_id> <relevance>`

3. **Entity Descriptions** (`--desc`):
   - Format: JSONL
   - Required fields:
     - `id`: Entity identifier (string)
     - `contents`: DBpedia description text (string)
   - Example:
     ```json
     {"id": "Q30", "contents": "The United States of America is a country primarily located in North America..."}
     ```

**Output**:
- `entity_ranking_data.jsonl`: Training examples in JSONL format
  ```json
  {"query_id": "301", "query": "International organized crime", "doc": "The United States...", "doc_id": "Q30", "label": 1}
  {"query_id": "301", "query": "International organized crime", "doc": "France is a country...", "doc_id": "Q142", "label": 0}
  ```

**Important Notes**:
- **Automatic 1:1 Balancing**: The script automatically balances positive and negative examples
  - For each query, `k = min(num_positive_entities, num_negative_entities)`
  - Only the first `k` examples from each class are included
  - This prevents class imbalance during training
- Entities without descriptions in `--desc` are skipped

---

### Step 1.3: Split Entity Data by Fold

Divide entity ranking data into 5 folds for cross-validation.

**Script**: `split_data_by_fold.py`

**Purpose**: Split the entity ranking data according to predefined query folds for k-fold cross-validation.

**Commands**:

**For Training Data**:
```bash
# Create training splits for each fold
python split_data_by_fold.py \
  --folds data/folds.json \
  --data data/entity_ranking_data.jsonl \
  --save data/entity_folds \
  --out entity_train.jsonl \
  --train
```

**For Evaluation Data**:
```bash
# Create test splits for each fold
python split_data_by_fold.py \
  --folds data/folds.json \
  --data data/entity_ranking_data.jsonl \
  --save data/entity_folds \
  --out entity_test.jsonl
```

**Flags**:
- `--folds`: Path to fold definitions JSON file (REQUIRED)
- `--data`: Path to entity ranking data from Step 1.2 (REQUIRED)
- `--save`: Directory where fold-specific files will be saved (REQUIRED)
- `--out`: Filename for output within each fold directory (REQUIRED)
- `--train`: Flag to indicate training data (OPTIONAL)
  - Include this flag when splitting training data
  - Omit this flag when splitting test data

**Input Requirements**:

1. **Folds File** (`--folds`):
   - Format: JSON
   - Structure:
     ```json
     {
       "0": {
         "training": ["301", "302", "303", ...],
         "testing": ["350", "351", "352", ...]
       },
       "1": {
         "training": ["350", "351", "352", ...],
         "testing": ["301", "302", "303", ...]
       },
       ...
       "4": { ... }
     }
     ```
   - Must contain exactly 5 folds (keys "0" through "4")
   - Each fold must have "training" and "testing" arrays of query IDs

2. **Entity Data** (`--data`):
   - Output from Step 1.2
   - JSONL format with `query_id` field

**Output Structure**:
```
data/entity_folds/
├── fold-0/
│   ├── entity_train.jsonl
│   └── entity_test.jsonl
├── fold-1/
│   ├── entity_train.jsonl
│   └── entity_test.jsonl
├── fold-2/
│   ├── entity_train.jsonl
│   └── entity_test.jsonl
├── fold-3/
│   ├── entity_train.jsonl
│   └── entity_test.jsonl
└── fold-4/
    ├── entity_train.jsonl
    └── entity_test.jsonl
```

**Important Notes**:
- You must run the script **twice**: once with `--train` flag, once without
- The `--save` directory must exist and contain subdirectories `fold-0/` through `fold-4/`
- Create these directories beforehand:
  ```bash
  mkdir -p data/entity_folds/fold-{0,1,2,3,4}
  ```

---

### Step 1.4: Train Entity Ranking Model

See details below.

**⚠️ CRITICAL**: These entity run files are required as input for Stage 2 (Document Ranking).

---

## Stage 2: Document Ranking Pipeline

### Step 2.1: Create Document Ranking Training Data

Generate training data for QDER using entity rankings.

**Script**: `make_doc_ranking_data.py`

**Purpose**: Create query-document pairs with entity-based features for training QDER. Documents are filtered based on whether they contain entities from the top-K entity ranking.

**Command**:
```bash
python make_doc_ranking_data.py \
  --queries data/queries.tsv \
  --qrels data/doc_qrels.txt \
  --docs data/corpus.jsonl \
  --entity_run runs/entity_run_fold0.txt \
  --top_k 20 \
  --save data/doc_ranking_data.jsonl
```

**Flags**:
- `--queries`: Path to queries file (REQUIRED)
- `--qrels`: Path to document QRELs (REQUIRED)
- `--docs`: Path to entity-linked corpus (REQUIRED)
- `--entity_run`: Path to entity ranking run file from Step 1.4 (REQUIRED)
- `--top_k`: Number of top entities to consider per query (REQUIRED)
  - Default: 20 (as used in paper)
  - Documents must contain at least one entity from top-K to be included
- `--save`: Output path for document ranking data (REQUIRED)
- `--balance`: Apply 1:1 balancing between relevant/non-relevant documents (OPTIONAL)
  - Include for training data
  - Omit for evaluation data

**Input Requirements**:

1. **Entity Run** (`--entity_run`):
   - Output from Step 1.4
   - TREC format: `<query_id> Q0 <entity_id> <rank> <score> <run_id>`
   - Must contain ranked entities for all queries

2. **Other inputs**: Same as Stage 1 (queries, doc_qrels, corpus)

**Output**:
- `doc_ranking_data.jsonl`: Document training examples
  ```json
  {
    "query_id": "301",
    "query": "International organized crime",
    "doc_id": "FBIS3-1",
    "doc_text": "...",
    "entities": ["Q30", "Q142"],
    "entity_scores": [0.95, 0.87],
    "label": 1
  }
  ```

**Important Notes**:
- **Document Filtering**: Only documents containing at least one top-K entity are included
  - This is the core of the entity-guided retrieval approach
  - Documents without relevant entities are filtered out
- **Entity Scores**: Entity rankings from Step 1.4 are included as features
- Run this script separately for **each fold** using the corresponding entity run file

---

### Step 2.2: Split Document Data by Fold

Divide document ranking data into folds.

**Script**: `split_data_by_fold.py` (same script as Step 1.3)

**Commands**:

**For Training Data**:
```bash
python split_data_by_fold.py \
  --folds data/folds.json \
  --data data/doc_ranking_data.jsonl \
  --save data/doc_folds \
  --out doc_train.jsonl \
  --train
```

**For Evaluation Data**:
```bash
python split_data_by_fold.py \
  --folds data/folds.json \
  --data data/doc_ranking_data.jsonl \
  --save data/doc_folds \
  --out doc_test.jsonl
```

**Output Structure**:
```
data/doc_folds/
├── fold-0/
│   ├── doc_train.jsonl
│   └── doc_test.jsonl
├── fold-1/
│   ├── doc_train.jsonl
│   └── doc_test.jsonl
...
```

**Important Notes**:
- Create directories before running:
  ```bash
  mkdir -p data/doc_folds/fold-{0,1,2,3,4}
  ```

---

### Step 2.3: Train QDER Model

See details below

---

## Complete Pipeline Example

### Full Workflow for TREC Robust 2004

```bash
#!/bin/bash

# ============================================
# STAGE 1: ENTITY RANKING PIPELINE
# ============================================

echo "Step 1.1: Creating entity QRELs..."
python make_entity_ranking_qrels.py \
  --qrels data/robust04/qrels.txt \
  --docs data/robust04/corpus.jsonl \
  --save data/robust04/entity_qrels.txt

echo "Step 1.2: Creating entity ranking data..."
python make_entity_ranking_data.py \
  --queries data/robust04/queries.tsv \
  --qrels data/robust04/entity_qrels.txt \
  --desc data/dbpedia/entity_descriptions.jsonl \
  --save data/robust04/entity_data.jsonl

echo "Step 1.3: Splitting entity data by fold..."
# Create directories
mkdir -p data/robust04/entity_folds/fold-{0,1,2,3,4}

# Split training data
python split_data_by_fold.py \
  --folds data/robust04/folds.json \
  --data data/robust04/entity_data.jsonl \
  --save data/robust04/entity_folds \
  --out entity_train.jsonl \
  --train

# Split test data
python split_data_by_fold.py \
  --folds data/robust04/folds.json \
  --data data/robust04/entity_data.jsonl \
  --save data/robust04/entity_folds \
  --out entity_test.jsonl

echo "Step 1.4: Training entity rankers (5-fold CV)..."
for fold in {0..4}; do
  python train_entity_ranker.py \
    --train data/robust04/entity_folds/fold-${fold}/entity_train.jsonl \
    --test data/robust04/entity_folds/fold-${fold}/entity_test.jsonl \
    --model bert-base-uncased \
    --epochs 10 \
    --save_model models/entity_ranker_fold${fold}.pt \
    --save_run runs/entity_run_fold${fold}.txt
done

# ============================================
# STAGE 2: DOCUMENT RANKING PIPELINE
# ============================================

echo "Step 2.1: Creating document ranking data (per fold)..."
mkdir -p data/robust04/doc_folds/fold-{0,1,2,3,4}

for fold in {0..4}; do
  python make_doc_ranking_data.py \
    --queries data/robust04/queries.tsv \
    --qrels data/robust04/qrels.txt \
    --docs data/robust04/corpus.jsonl \
    --entity_run runs/entity_run_fold${fold}.txt \
    --top_k 20 \
    --balance \
    --save data/robust04/doc_data_fold${fold}.jsonl
done

echo "Step 2.2: Splitting document data by fold..."
for fold in {0..4}; do
  # Training data
  python split_data_by_fold.py \
    --folds data/robust04/folds.json \
    --data data/robust04/doc_data_fold${fold}.jsonl \
    --save data/robust04/doc_folds \
    --out doc_train.jsonl \
    --train
  
  # Test data
  python split_data_by_fold.py \
    --folds data/robust04/folds.json \
    --data data/robust04/doc_data_fold${fold}.jsonl \
    --save data/robust04/doc_folds \
    --out doc_test.jsonl
done

echo "Step 2.3: Training QDER models (5-fold CV)..."
for fold in {0..4}; do
  python train_qder.py \
    --train data/robust04/doc_folds/fold-${fold}/doc_train.jsonl \
    --test data/robust04/doc_folds/fold-${fold}/doc_test.jsonl \
    --entity_embeddings data/wikipedia2vec/enwiki_20180420_100d.bin \
    --model bert-base-uncased \
    --epochs 10 \
    --scoring bilinear \
    --interactions add_multiply \
    --save_model models/qder_fold${fold}.pt \
    --save_run runs/qder_run_fold${fold}.txt
done

echo "Pipeline complete!"
```

---

# Train Entity Ranking Model (Step 1.4)

## Overview

Train a MonoBert-based entity ranker using 5-fold cross-validation to learn which entities are semantically relevant to queries. This ranker produces entity rankings that guide document filtering in Stage 2.

---

## Training Script

**Script**: `train.py`

**Purpose**: Fine-tune MonoBert (BERT with 2-class classification head) to rank entities based on their DBpedia descriptions and query text. The model learns to distinguish entities that appear exclusively in relevant documents from those in non-relevant documents.

**Task Type**: Classification (pointwise) - uses MonoBert with CrossEntropyLoss

---

## Required Inputs

Before training, ensure you have:

1. **Training Data** (`entity_train.jsonl`): From Step 1.3
2. **Validation Data** (`entity_test.jsonl`): From Step 1.3  
3. **Entity QRELs** (`entity_qrels.txt`): From Step 1.1

---

## Data Format Requirements

### Training/Validation Data Format

Each line in `entity_train.jsonl` / `entity_test.jsonl` must be a JSON object with:
```json
{
  "query_id": "301",
  "query": "International organized crime",
  "doc_id": "Q30",
  "doc": "The United States of America is a country primarily located in North America...",
  "label": 1
}
```

**Required Fields**:
- `query_id` (string): Query identifier
- `query` (string): Query text
- `doc_id` (string): Entity identifier (e.g., Wikidata/DBpedia ID)
- `doc` (string): Entity description text from DBpedia
  - This is the actual text that will be tokenized and encoded by BERT
  - Forms the query-entity pair: `[CLS] query [SEP] entity_description [SEP]`
- `label` (int): Relevance label (1 = relevant, 0 = non-relevant)

**Important Notes**:
1. **1:1 Balanced**: Data from Step 1.2 is already balanced (equal positive/negative per query)
2. **Exclusive Entities Only**: Training data contains only entities that appear exclusively in relevant OR non-relevant documents (entities in both are filtered out in Step 1.1)
3. **Field Name Flexibility**: The script supports both `"doc"` and `"doc_text"` field names for the entity description

---

## Command

### Basic Training (Single Fold)
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

**⚠️ CRITICAL**: You MUST include `--task classification` to use MonoBert for entity ranking.

---

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--train` | Path to training data (JSONL) | `data/entity_folds/fold-0/entity_train.jsonl` |
| `--dev` | Path to validation data (JSONL) | `data/entity_folds/fold-0/entity_test.jsonl` |
| `--qrels` | Entity-level ground truth (TREC format) | `data/robust04/entity_qrels.txt` |
| `--save-dir` | Directory to save model checkpoints | `models/entity_ranker/fold-0` |
| `--task` | Task type (REQUIRED) | `classification` |

### Model Configuration Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--query-enc` | Pre-trained language model | `bert` | `bert`, `roberta`, `deberta`, `distilbert`, `electra`, `conv-bert`, `t5`, `ernie` |
| `--mode` | BERT pooling mode | `cls` | `cls`, `pooling` |
| `--max-len` | Maximum sequence length | `512` | Any integer ≤ 512 |

**Model Mapping**:
- `bert` → `bert-base-uncased`
- `roberta` → `roberta-base`
- `deberta` → `microsoft/deberta-base`
- `distilbert` → `distilbert-base-uncased`
- `electra` → `google/electra-small-discriminator`
- `conv-bert` → `YituTech/conv-bert-base`
- `t5` → `t5-base`
- `ernie` → `nghuyong/ernie-2.0-base-en`

### Training Hyperparameters

| Argument | Description | Default | Recommended |
|----------|-------------|---------|-------------|
| `--epoch` | Number of training epochs | `20` | `10-20` |
| `--batch-size` | Batch size | `8` | `16-40` (entity ranking is less memory-intensive) |
| `--learning-rate` | AdamW learning rate | `2e-5` | `1e-5` to `3e-5` |
| `--n-warmup-steps` | Linear warmup steps | `2` | `500-1000` |

### Evaluation Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--metric` | Validation metric | `map` | `map`, `ndcg`, `ndcg_cut_20`, `P_20`, `mrr_cut_10` |
| `--eval-every` | Evaluate every N epochs | `1` | Any positive integer |

### System Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--use-cuda` | Enable GPU training | Flag (off by default) |
| `--cuda` | CUDA device index | `0` |
| `--num-workers` | DataLoader workers | `0` |

### Output Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--save` | Model checkpoint filename | `model.bin` |
| `--run` | Validation run filename | `dev.run` |
| `--checkpoint` | Path to checkpoint to resume from | `None` |

---

## Model Architecture Details

### Entity Ranking Model (MonoBert)

The model consists of:

1. **BERT Encoder**:
   - Base: BERT-base-uncased (or specified model)
   - Input: `[CLS] query [SEP] entity_description [SEP]`
   - Output: 768-dimensional [CLS] token embedding

2. **Classification Head**:
   - Linear layer: `768 → 2`
   - Outputs 2 classes: [non-relevant, relevant]

3. **Loss Function**:
   - CrossEntropyLoss
   - Treats entity ranking as binary classification

### Forward Pass
```python
# Input
query_text: "International organized crime"
entity_desc: "The United States of America is a country..."

# Tokenization
input_ids = tokenizer(
    text=query_text,
    text_pair=entity_desc,
    max_length=512,
    truncation=True,
    padding='max_length'
)
# Result: [CLS] query tokens [SEP] entity tokens [SEP]

# Process
output = BERT(input_ids)              # [batch, seq_len, 768]
cls_embedding = output[:, 0, :]       # [batch, 768]
logits = Linear(cls_embedding)        # [batch, 2]
probs = softmax(logits)               # [batch, 2]

# Output
score = probs[:, 1]  # Probability of relevant class
```

### Training Loss
```python
# Binary cross-entropy loss
labels = [1, 0, 1, ...]  # Binary labels
logits = model(input_ids, attention_mask, token_type_ids)  # [batch, 2]
loss = CrossEntropyLoss(logits, labels)
```

---

## Training Process

### What Happens During Training

1. **Data Loading**:
   - Training data loaded with automatic shuffling
   - Validation data processed in order
   - Both use 1:1 balanced data from Step 1.2

2. **Epoch Loop**:
```
   For each epoch:
     1. Train on training set
     2. If epoch % eval_every == 0:
        a. Evaluate on validation set
        b. Generate TREC run file
        c. Compute MAP using pytrec_eval
        d. Save checkpoint if best performance
```

3. **Forward Pass** (per batch):
```
   Query + Entity Description → Tokenize → BERT
   BERT → [CLS] embedding [768]
   [CLS] → Linear layer → [2] (non-relevant, relevant)
   Softmax → Probabilities
```

4. **Loss Computation**:
   - CrossEntropyLoss between predicted logits and binary labels

5. **Optimization**:
   - Adam optimizer
   - Linear warmup + decay schedule

6. **Validation**:
   - Generates entity rankings for all validation queries
   - Uses softmax probability of relevant class (index 1) as score
   - Saves as TREC run file
   - Computes MAP against entity QRELs
   - Saves best checkpoint based on MAP

---

## 5-Fold Cross-Validation Script

Train all 5 folds sequentially:
```bash
#!/bin/bash

# Configuration
DATASET="robust04"
DATA_DIR="data/${DATASET}/entity_folds"
QRELS="data/${DATASET}/entity_qrels.txt"
SAVE_ROOT="models/entity_ranker/${DATASET}"
RUNS_ROOT="runs/entity_ranker/${DATASET}"

# Training hyperparameters
EPOCHS=10
BATCH_SIZE=32
LR=2e-5
WARMUP=1000
MAX_LEN=512
QUERY_ENC="bert"
MODE="cls"

# Create output directories
mkdir -p ${SAVE_ROOT}
mkdir -p ${RUNS_ROOT}

# Train each fold
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
    echo ""
done

echo "Entity Ranker 5-Fold Cross-Validation Complete!"
```

Save as `train_entity_ranker_all_folds.sh` and run:
```bash
chmod +x train_entity_ranker_all_folds.sh
./train_entity_ranker_all_folds.sh
```

---

## Generating Entity Rankings (Inference)

After training, generate entity rankings on test data:

### Test Script

**Script**: `test.py`

**Command**:
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

**⚠️ CRITICAL**: You MUST include `--task classification` when testing, matching what you used during training.

**Arguments**:
- `--test`: Path to test data (JSONL)
- `--checkpoint`: Path to trained model checkpoint
- `--run`: Output TREC run file path
- `--task`: Must be `classification` (same as training)
- Other arguments same as training

### Generate All Fold Rankings
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

## Common Issues and Solutions

### Issue: AttributeError: 'BertTokenizer' has no attribute 'encode_plus'

**Error**: 
```
AttributeError: BertTokenizer has no attribute encode_plus. Did you mean: '_encode_plus'?
```

**Cause**: Using newer version of transformers where `encode_plus` is deprecated.

**Solution**: The provided scripts already use the updated tokenizer API (`tokenizer(...)` instead of `tokenizer.encode_plus(...)`). Make sure you're using the latest version of the scripts.

### Issue: KeyError: 'doc_text'

**Error**:
```
KeyError: 'doc_text'
```

**Cause**: Your data has field name `"doc"` but script expects `"doc_text"`.

**Solution**: The provided scripts support both field names automatically. The script checks for `doc_text` first, then falls back to `doc`:
```python
doc_text = example.get('doc_text', example.get('doc', '')).strip()
```

### Issue: Model outputs unexpected keys

**Warning**:
```
BertModel LOAD REPORT from: bert-base-uncased
Key                                        | Status     |
-------------------------------------------+------------+
cls.predictions.transform.LayerNorm.weight | UNEXPECTED |
cls.predictions.transform.dense.bias       | UNEXPECTED |
...
```

**This is NORMAL and expected!** These are pre-training head layers that aren't needed for classification. The warning can be safely ignored.

---

## Understanding MonoBert vs DuoBert

### MonoBert (Classification Task - For Entity Ranking)

- **Task**: Pointwise classification
- **Input**: Single query-entity pair
- **Output**: 2 classes [non-relevant, relevant]
- **Loss**: CrossEntropyLoss
- **Use Case**: Entity ranking (this step)

### DuoBert (Ranking Task - For Document Ranking)

- **Task**: Pairwise ranking
- **Input**: Query + positive doc, Query + negative doc
- **Output**: Single relevance score per document
- **Loss**: MarginRankingLoss
- **Use Case**: Document re-ranking (Step 2.3)

**For entity ranking, we use MonoBert (classification)** because:
1. We have clear binary labels (relevant/non-relevant entities)
2. Training data is pointwise (one entity per example)
3. Classification head outputs probability of relevance

---

## Task Types Explained

The training scripts use `--task` to determine the training approach:

### Classification Task (Pointwise)

**Use For**: Entity Ranking (Step 1.4)
```bash
--task classification
```

**Training Data Format**:
```json
{"query": "...", "doc": "...", "label": 1}
```

**Model**: MonoBert
- Processes ONE query-entity pair at a time
- Outputs 2 classes: [non-relevant, relevant]
- Loss: CrossEntropyLoss

### Ranking Task (Pairwise)

**Use For**: Document Re-Ranking (Step 2.3)
```bash
--task ranking
```

**Training Data Format**:
```json
{"query": "...", "doc_pos_text": "...", "doc_neg_text": "..."}
```

**Model**: DuoBert
- Processes TWO documents per query (positive and negative)
- Outputs single relevance score per document
- Loss: MarginRankingLoss (margin=1)
- Learns: "positive document should score > negative document"

### Summary Table

| Aspect | Classification (MonoBert) | Ranking (DuoBert) |
|--------|---------------------------|-------------------|
| **Command** | `--task classification` | `--task ranking` |
| **Use Case** | Entity Ranking | Document Ranking |
| **Input** | Single query-item pair | Query + pos/neg pair |
| **Output** | 2 classes [0, 1] | Single score |
| **Loss** | CrossEntropyLoss | MarginRankingLoss |
| **Evaluation** | Softmax prob of class 1 | Raw score |

---

## Model Architecture Details

### Entity Ranking Model Components

The model consists of:

1. **Query Encoder**:
   - Base: BERT-base-uncased (or specified model)
   - Input: Query text tokens
   - Output: 768-dimensional query embedding (from [CLS] token)

2. **Entity Representation**:
   - Input: Pre-computed Wikipedia2Vec embedding (300-dimensional)
   - **Not** encoded through BERT
   - Directly concatenated with query embedding

3. **Scoring Function**:
   - Linear layer: `[768 + 300] → 1`
   - Maps concatenated query-entity embedding to relevance score

### Forward Pass

```python
# Input
query_text: "International organized crime"
entity_id: "Q30" (United States)
entity_embedding: [300-dim Wikipedia2Vec vector]

# Process
query_emb = BERT([CLS] International organized crime [SEP])  # [768]
concat_emb = [query_emb; entity_embedding]  # [768 + 300 = 1068]
score = Linear(concat_emb)  # [1]

# Output
score: 0.87 (probability entity is relevant to query)
```

### Loss Function

- **Binary Cross-Entropy with Logits**
- Treats entity ranking as binary classification
- Positive: Entity appears exclusively in relevant documents
- Negative: Entity appears exclusively in non-relevant documents

---

## Training Process

### What Happens During Training

1. **Data Loading**:
   - Training data loaded with automatic shuffling
   - Validation data processed in order
   - Both use 1:1 balanced data from Step 1.2

2. **Epoch Loop**:
   ```
   For each epoch:
     1. Train on training set
     2. If epoch % eval_every == 0:
        a. Evaluate on validation set
        b. Generate TREC run file
        c. Compute MAP using pytrec_eval
        d. Save checkpoint if best performance
   ```

3. **Forward Pass** (per batch):
   ```
   Query → BERT → query_emb [batch, 768]
   Entity → Wikipedia2Vec (pre-computed) → entity_emb [batch, 300]
   Concat → [query_emb; entity_emb] → concat_emb [batch, 1068]
   Linear → concat_emb → score [batch, 1]
   ```

4. **Loss Computation**:
   - BCE with Logits loss
   - Compares predicted score to binary label (0 or 1)

5. **Optimization**:
   - Adam optimizer
   - Linear warmup + decay schedule
   - Gradient clipping (if implemented)

6. **Validation**:
   - Generates entity rankings for all validation queries
   - Saves as TREC run file
   - Computes MAP against entity QRELs
   - Saves best checkpoint based on MAP

---

## 5-Fold Cross-Validation Script

Train all 5 folds sequentially:

```bash
#!/bin/bash

# Configuration
DATASET="robust04"
DATA_DIR="data/${DATASET}/entity_folds"
QRELS="data/${DATASET}/entity_qrels.txt"
SAVE_ROOT="models/entity_ranker/${DATASET}"
RUNS_ROOT="runs/entity_ranker/${DATASET}"

# Training hyperparameters
EPOCHS=10
BATCH_SIZE=32
LR=2e-5
WARMUP=1000
MAX_LEN=512
QUERY_ENC="bert"

# Create output directories
mkdir -p ${SAVE_ROOT}
mkdir -p ${RUNS_ROOT}

# Train each fold
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
        --max-len ${MAX_LEN} \
        --query-enc ${QUERY_ENC} \
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
    echo ""
done

echo "Entity Ranker 5-Fold Cross-Validation Complete!"
```

Save as `train_entity_ranker_all_folds.sh` and run:
```bash
chmod +x train_entity_ranker_all_folds.sh
./train_entity_ranker_all_folds.sh
```

---

## Generating Entity Rankings (Inference)

After training, generate entity rankings on test data:

### Test Script

**Script**: `test.py`

**Command**:
```bash
python test.py \
  --test data/entity_folds/fold-0/entity_test.jsonl \
  --checkpoint models/entity_ranker/fold-0/model.bin \
  --run runs/entity_run_fold0.txt \
  --max-len 512 \
  --query-enc bert \
  --batch-size 32 \
  --use-cuda \
  --cuda 0
```

**Arguments**:
- `--test`: Path to test data (JSONL)
- `--checkpoint`: Path to trained model checkpoint
- `--run`: Output TREC run file path
- Other arguments same as training

### Generate All Fold Rankings

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
        --max-len 512 \
        --query-enc bert \
        --batch-size 32 \
        --use-cuda \
        --cuda 0
    
    echo "Fold ${fold} complete."
done

echo "All entity rankings generated!"
```

---

## Output Files

### Per Fold

After training fold-0, you'll have:

```
models/entity_ranker/robust04/fold-0/
├── model.bin                      # Best checkpoint
├── dev_entity_run_fold0.txt      # Dev run from best epoch
└── config.json                    # Model configuration

runs/entity_ranker/robust04/
└── entity_run_fold0.txt          # Test set entity rankings
```

### Entity Run File Format

`entity_run_fold0.txt`:
```
301 Q0 Q30 1 0.9234 BERT
301 Q0 Q142 2 0.8876 BERT
301 Q0 Q1234 3 0.7654 BERT
...
```

Columns:
1. Query ID
2. Q0 (constant)
3. Entity ID (e.g., Wikidata ID)
4. Rank
5. Score (model prediction)
6. Run name

---

## Monitoring Training

### Console Output

```
CUDA Device: cuda:0
Query Encoder: bert
Reading train data...
Reading dev data...
[Done].
Creating data loaders...
Number of workers = 4
Batch Size = 32
[Done].
Using device: cuda:0
Starting to train...

Epoch: 01 | Epoch Time: 3m 45s
	 Train Loss: 0.542| Val. Metric: 0.4521 | Best Val. Metric: 0.4521
Model saved to ==> models/entity_ranker/fold-0/model.bin

Epoch: 02 | Epoch Time: 3m 42s
	 Train Loss: 0.398| Val. Metric: 0.5234 | Best Val. Metric: 0.5234
Model saved to ==> models/entity_ranker/fold-0/model.bin

...

Epoch: 10 | Epoch Time: 3m 40s
	 Train Loss: 0.187| Val. Metric: 0.6432 | Best Val. Metric: 0.6543

Training complete.
```

### Key Metrics to Watch

1. **Train Loss**: Should steadily decrease
   - Typical range: 0.6 → 0.15 over 10 epochs
   - If stuck >0.4 after 5 epochs, check learning rate

2. **Validation MAP**: Should increase
   - Expected range: 0.45 → 0.70
   - If <0.40, check:
     - Entity embeddings loaded correctly
     - Data balancing applied properly
     - Entity QRELs exclude shared entities

3. **Training Time**:
   - Per epoch: ~3-5 minutes (RTX 3090, batch_size=32)
   - Full 10 epochs: ~30-50 minutes per fold
   - 5-fold CV: ~2.5-4 hours total

**Note**: Entity ranking is much faster than document ranking because:
- Simpler model architecture (no dual channels)
- Smaller input sequences (entity descriptions vs full documents)
- Pre-computed entity embeddings

---

## Using Entity Rankings in Stage 2

The entity run files generated here are **critical inputs** for Stage 2 (Document Ranking):

```bash
# Entity rankings from Step 1.4
runs/entity_ranker/robust04/entity_run_fold0.txt

# Used in Step 2.1 to filter documents
python make_doc_ranking_data.py \
  --entity_run runs/entity_ranker/robust04/entity_run_fold0.txt \
  --top_k 20 \
  ...
```

**How it works**:
1. Entity ranker scores all entities for each query
2. Top-K entities (K=20) selected per query
3. Documents filtered to only include those containing ≥1 top-K entity
4. Filtered documents become training data for QDER (Stage 2)

**This filtering is the core of entity-guided retrieval**: Documents must contain entities the model deems semantically relevant to the query.

---



# Train QDER Document Re-Ranking Model (Step 2.3)

## Overview

Train the QDER model using 5-fold cross-validation with the dual-channel (text + entity) architecture, bilinear scoring, and attention-guided interaction modeling.

---

## Training Script

**Script**: `train.py`

**Purpose**: Fine-tune BERT-based QDER model to re-rank documents using:
- Query-specific document representations
- Entity-aware attention mechanisms
- Bilinear interaction scoring (NOT linear)
- Addition and Multiplication interactions ONLY (NO subtraction)

---

## Required Inputs

Before training, ensure you have:

1. **Training Data** (`doc_train.jsonl`): From Step 2.2
2. **Validation Data** (`doc_test.jsonl`): From Step 2.2
3. **Document QRELs** (`doc_qrels.txt`): Ground truth relevance judgments
4. **Pre-computed Embeddings**:
   - Document text embeddings (from BERT/sentence encoder)
   - Entity embeddings (Wikipedia2Vec format)

---

## Data Format Requirements

### Training/Validation Data Format

Each line in `doc_train.jsonl` / `doc_test.jsonl` must be a JSON object with:

```json
{
  "query_id": "301",
  "query": "International organized crime",
  "doc_id": "FBIS3-1",
  "doc_chunk_embeddings": [[0.12, 0.45, ...], [0.23, 0.56, ...], ...],
  "doc_ent_emb": [[0.34, 0.67, ...], [0.45, 0.78, ...]],
  "label": 1,
  "doc_score": 234.5
}
```

**Required Fields**:
- `query_id` (string): Query identifier
- `query` (string): Query text
- `doc_id` (string): Document identifier
- `doc_chunk_embeddings` (list of lists): Pre-computed text embeddings for document chunks/passages
  - Shape: `[num_chunks, 768]` for BERT-base
  - Each chunk is a passage from the document (e.g., 10-sentence sliding window)
- `doc_ent_emb` (list of lists): Pre-computed entity embeddings
  - Shape: `[num_entities, 300]` for Wikipedia2Vec
  - Only entities from top-K entity ranking (Step 1.4)
- `label` (int): Relevance label (1 = relevant, 0 = non-relevant)

**Optional Fields**:
- `doc_score` (float): Initial retrieval score (e.g., BM25)
  - Used to weight interaction embeddings
  - Normalized to [0, 1] range recommended

---

## Command

### Basic Training (Single Fold)
```bash
python train.py \
  --train data/doc_folds/fold-0/doc_train.jsonl \
  --dev data/doc_folds/fold-0/doc_test.jsonl \
  --qrels data/robust04/qrels.txt \
  --save-dir models/fold-0 \
  --save model.pt \
  --task ranking \
  --max-len 512 \
  --query-enc bert \
  --mode cls \
  --epoch 10 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --n-warmup-steps 1000 \
  --metric map \
  --eval-every 1 \
  --num-workers 4 \
  --use-cuda \
  --cuda 0 \
  --run dev_run.txt
```

**⚠️ CRITICAL**: You MUST include `--task ranking` to use DuoBert for document re-ranking.

---

## Command-Line Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--train` | Path to training data (JSONL) | `data/doc_folds/fold-0/doc_train.jsonl` |
| `--dev` | Path to validation data (JSONL) | `data/doc_folds/fold-0/doc_test.jsonl` |
| `--qrels` | Ground truth file (TREC format) | `data/robust04/qrels.txt` |
| `--save-dir` | Directory to save model checkpoints | `models/fold-0` |
| `--task` | Task type (REQUIRED) | `ranking` |

### Model Architecture Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--query-enc` | Pre-trained language model | `bert` | `bert`, `roberta`, `deberta` |
| `--mode` | BERT pooling mode | `cls` | `cls`, `pooling` |
| `--max-len` | Maximum input sequence length | `512` | Any integer ≤ 512 |
| `--task` | Training task type | `classification` | `classification`, `ranking` |

**⚠️ CRITICAL**: Use `--task ranking` for document re-ranking with DuoBert (pairwise training).

### Training Hyperparameters

| Argument | Description | Default | Recommended |
|----------|-------------|---------|-------------|
| `--epoch` | Number of training epochs | `20` | `10-20` |
| `--batch-size` | Batch size | `8` | `8-32` (depends on GPU memory) |
| `--learning-rate` | AdamW learning rate | `2e-5` | `1e-5` to `3e-5` |
| `--n-warmup-steps` | Linear warmup steps | `1000` | `500-2000` |
### Evaluation Arguments

| Argument | Description | Default | Options |
|----------|-------------|---------|---------|
| `--metric` | Validation metric | `map` | `map`, `ndcg`, `ndcg_cut_20`, `P_20` |
| `--eval-every` | Evaluate every N epochs | `1` | Any positive integer |

### System Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--use-cuda` | Enable GPU training | Flag (off by default) |
| `--cuda` | CUDA device index | `0` |
| `--num-workers` | DataLoader workers | `0` |

### Output Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--save` | Model checkpoint filename | `model.bin` |
| `--run` | Validation run filename | `dev.run` |

---

## Model Architecture Details

### QDER Components

The model consists of:

1. **Dual Encoders**:
   - Query Encoder: BERT-base-uncased (or specified model)
   - Document Encoder: BERT-base-uncased (shared or separate)

2. **Dual Channels**:
   - **Text Channel**: Processes document text embeddings (`doc_chunk_embeddings`)
   - **Entity Channel**: Processes entity embeddings (`doc_ent_emb`)

3. **Attention Mechanisms**:
   - Text-to-Text Attention: `softmax(Q_text · D_text^T)`
   - Entity-to-Entity Attention: `softmax(Q_entity · D_entity^T)`

4. **Interaction Operations** (NO SUBTRACTION):
   - Addition: `Q + D_weighted`
   - Multiplication: `Q ⊙ D_weighted`

5. **Scoring Function**:
   - **Bilinear**: `score = x^T · M · x` where `M` is learned
   - Linear: `score = W · x + b` (not recommended)

### Key Architectural Differences from Code

The uploaded code (`model.py`) expects:
```python
def forward(
    query_input_ids,      # Query tokens
    query_attention_mask,
    query_token_type_ids,
    query_entity_emb,      # Query entity embeddings
    doc_input_ids,         # Document tokens
    doc_attention_mask,
    doc_token_type_ids,
    doc_entity_emb,        # Document entity embeddings
    doc_tfidf_weights,     # Optional: TF-IDF weights per token
    doc_scores            # Optional: BM25/retrieval scores
)
```

However, the dataset (`dataset.py`) provides:
```python
{
    'query_input_ids': ...,
    'doc_text_emb': ...,      # Pre-computed document embeddings
    'doc_entity_emb': ...,    # Pre-computed entity embeddings
    'doc_score': ...          # Optional BM25 score
}
```

**⚠️ Important**: There's a mismatch between `model.py` and `dataset.py`:
- Model expects raw document tokens (`doc_input_ids`)
- Dataset provides pre-computed embeddings (`doc_text_emb`)

You must either:
1. **Modify the dataset** to provide raw document text for encoding
2. **Modify the model** to accept pre-computed embeddings

For the published results, we used **pre-computed document embeddings** to speed up training.

---

## Training Process

### What Happens During Training

1. **Data Loading**:
   - Training data is loaded with automatic 1:1 balancing (from Step 2.1)
   - Validation data includes all candidate documents (no balancing)

2. **Forward Pass**:
   ```
   Query Text → BERT → Query Embedding (Q_text)
   Doc Text → Pre-computed → Doc Embedding (D_text)
   
   Attention Weighting:
   A_text = softmax(Q_text · D_text^T)
   D_text_weighted = A_text · D_text
   
   Interactions:
   I_add = Q_text + D_text_weighted
   I_mul = Q_text ⊙ D_text_weighted
   
   [Same for entity channel]
   
   Combined = [I_add_text, I_mul_text, I_add_entity, I_mul_entity]
   Score = Bilinear(Combined, Combined)
   ```

3. **Loss Computation**:
   - Binary Cross-Entropy with Logits
   - Compares predicted score to binary label (0 or 1)

4. **Optimization**:
   - AdamW optimizer
   - Linear warmup + decay schedule

5. **Validation**:
   - Every `--eval-every` epochs
   - Generates TREC run file
   - Computes MAP/nDCG using `pytrec_eval`
   - Saves best checkpoint

---

## 5-Fold Cross-Validation Script

Train all 5 folds sequentially:

```bash
#!/bin/bash

# Configuration
DATASET="robust04"
DATA_DIR="data/${DATASET}/doc_folds"
QRELS="data/${DATASET}/qrels.txt"
SAVE_ROOT="models/${DATASET}"
RUNS_ROOT="runs/${DATASET}"

# Training hyperparameters
EPOCHS=10
BATCH_SIZE=16
LR=2e-5
WARMUP=1000
MAX_LEN=512

# Create output directories
mkdir -p ${SAVE_ROOT}
mkdir -p ${RUNS_ROOT}

# Train each fold
for fold in {0..4}; do
    echo "========================================"
    echo "Training Fold ${fold}"
    echo "========================================"
    
    python train.py \
        --train ${DATA_DIR}/fold-${fold}/doc_train.jsonl \
        --dev ${DATA_DIR}/fold-${fold}/doc_test.jsonl \
        --qrels ${QRELS} \
        --save-dir ${SAVE_ROOT}/fold-${fold} \
        --save model.pt \
        --max-len ${MAX_LEN} \
        --text-enc bert \
        --score-method bilinear \
        --epoch ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --learning-rate ${LR} \
        --n-warmup-steps ${WARMUP} \
        --metric map \
        --eval-every 1 \
        --num-workers 4 \
        --use-cuda \
        --cuda 0 \
        --run dev_run_fold${fold}.txt
    
    echo "Fold ${fold} complete."
    echo ""
done

echo "5-Fold Cross-Validation Complete!"
```

Save this as `train_all_folds.sh` and run:
```bash
chmod +x train_all_folds.sh
./train_all_folds.sh
```

---

## Output Files

### Per Fold

After training fold-0, you'll have:

```
models/robust04/fold-0/
├── model.pt                  # Best checkpoint (highest MAP on validation)
└── dev_run_fold0.txt        # TREC run file from best epoch

logs/
└── fold-0_train.log         # Training logs (if redirected)
```

### TREC Run File Format

`dev_run_fold0.txt`:
```
301 Q0 FBIS3-1 1 0.9234 BERT
301 Q0 FBIS3-5 2 0.8876 BERT
301 Q0 FBIS3-12 3 0.7654 BERT
...
```

Columns:
1. Query ID
2. Q0 (constant)
3. Document ID
4. Rank
5. Score (model prediction)
6. Run name

---

## Monitoring Training

### Console Output

```
Creating datasets...
Loading train data from data/doc_folds/fold-0/doc_train.jsonl...
100%|████████| 45123/45123 [00:03<00:00]
Loading test data from data/doc_folds/fold-0/doc_test.jsonl...
100%|████████| 12456/12456 [00:01<00:00]

Starting Training on cuda:0...

Epoch 1/10
--------------------------
Training....
100%|████████| 2820/2820 [12:34<00:00, 3.73batch/s]
Running validation...
Evaluating: 100%|████████| 778/778 [03:21<00:00, 3.86batch/s]
New Best map: 0.4523. Saving checkpoint...
Model saved to ==> models/fold-0/model.pt
Epoch Time: 15m 55s
Train Loss: 0.4231 | Val map: 0.4523 | Best Val map: 0.4523
```

### Key Metrics to Watch

1. **Train Loss**: Should steadily decrease
   - Typical range: 0.6 → 0.2 over 10 epochs
   - If stuck >0.5 after 5 epochs, try lower learning rate

2. **Validation MAP**: Should increase
   - Baseline (BM25+RM3): ~0.29
   - QDER Expected: 0.55-0.61
   - If <0.35, check:
     - Entity embeddings loaded correctly
     - Bilinear scoring enabled
     - No subtraction interaction

3. **Training Time**:
   - Per epoch: ~15-20 minutes (RTX 3090, batch_size=16)
   - Full 10 epochs: ~2.5-3 hours per fold
   - 5-fold CV: ~12-15 hours total

---

## Troubleshooting

### Issue: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Reduce batch size
--batch-size 8   # or even 4

# Reduce sequence length
--max-len 256    # instead of 512

# Use gradient accumulation (requires code modification)
# Accumulate gradients over 2 steps to simulate batch_size=32
```

### Issue: Low Validation Scores (<0.35 MAP)

**Potential Causes**:

1. **Wrong scoring method**:
   ```bash
   # Check you're using bilinear, not linear
   --score-method bilinear
   ```

2. **Subtraction interaction included**:
   - Verify `model.py` only uses Add and Multiply
   - Check line ~100: should concatenate only 4 components (2 text + 2 entity)

3. **Missing entity embeddings**:
   ```python
   # In dataset.py, verify:
   item['doc_entity_emb'] = example['doc_ent_emb']  # Must exist
   ```

4. **BM25 scores not normalized**:
   - If using `doc_score`, normalize to [0, 1]:
   ```python
   doc_score = (score - min_score) / (max_score - min_score)
   ```

### Issue: Model Forward Signature Mismatch

**Error**: `forward() got an unexpected keyword argument 'doc_text_emb'`

**Cause**: `model.py` expects `doc_input_ids` but `dataset.py` provides `doc_text_emb`

**Solution**: Update `model.py` to accept pre-computed embeddings:

```python
def forward(
    self,
    query_input_ids,
    query_attention_mask,
    query_token_type_ids,
    doc_text_emb,         # Pre-computed, not doc_input_ids
    doc_entity_emb,
    doc_scores=None
):
    # Encode query
    query_text_emb = self.query_encoder(
        input_ids=query_input_ids,
        attention_mask=query_attention_mask,
        token_type_ids=query_token_type_ids
    )
    
    # Use pre-computed doc embeddings directly
    # doc_text_emb shape: [batch, num_chunks, 768]
    
    # Continue with attention and interactions...
```

### Issue: Validation Takes Too Long

**Solution**: Reduce validation data or evaluate less frequently

```bash
# Evaluate every 2 epochs instead of every epoch
--eval-every 2

# Or use a sample of validation data during training
# Full evaluation only at the end
```

---

## Expected Performance

### TREC Robust 2004 (Title Queries)

| Metric | BM25+RM3 | QDER (Expected) | Improvement |
|--------|----------|-----------------|-------------|
| MAP | 0.291 | 0.608 | +109% |
| nDCG@20 | 0.435 | 0.769 | +77% |
| P@20 | 0.384 | 0.736 | +92% |
| MRR | 0.669 | 0.975 | +46% |

### Per-Fold Variance

Expect some variance across folds:

```
Fold 0: MAP = 0.612
Fold 1: MAP = 0.601
Fold 2: MAP = 0.615
Fold 3: MAP = 0.598
Fold 4: MAP = 0.608

Average: 0.607 ± 0.007
```

**⚠️ If any fold shows MAP <0.50**, something is wrong:
- Check entity ranking quality from Step 1.4
- Verify data preparation in Step 2.1
- Ensure bilinear scoring and no subtraction

---

## Generating Final Rankings

After training all 5 folds, combine results:

```bash
# For each fold, generate test rankings
for fold in {0..4}; do
    python test.py \
        --model models/robust04/fold-${fold}/model.pt \
        --test data/robust04/doc_folds/fold-${fold}/doc_test.jsonl \
        --save runs/robust04/test_run_fold${fold}.txt \
        --text-enc bert \
        --score-method bilinear \
        --use-cuda
done

# Combine all fold test runs
cat runs/robust04/test_run_fold*.txt > runs/robust04/qder_final.txt

# Evaluate final performance
trec_eval -c -m map -m ndcg_cut.20 -m P.20 \
    data/robust04/qrels.txt \
    runs/robust04/qder_final.txt
```

---

## Hardware Requirements

### Minimum

- **GPU**: NVIDIA GTX 1080 Ti (11GB)
- **RAM**: 32GB
- **Storage**: 50GB free space
- **Training Time**: ~20 hours for 5-fold CV

### Recommended

- **GPU**: NVIDIA RTX 3090 / A6000 (24GB+)
- **RAM**: 64GB
- **Storage**: 100GB SSD
- **Training Time**: ~12 hours for 5-fold CV

### CPU-Only Training

**Not recommended** but possible:
```bash
# Remove --use-cuda flag
python train.py ... --batch-size 4

# Expected time: 5-7 days for 5-fold CV
```

---

## Next Steps

After training:

1. **Evaluate on test set** (Step 2.4)
2. **Combine with BM25 scores** using linear interpolation (λ = 0.3)
3. **Generate final TREC run files**
4. **Compute official metrics** using `trec_eval -c`
5. **Compare with baselines** and published results

---

## Citation

If you use this training procedure, please cite:

```bibtex
@inproceedings{chatterjee2025qder,
  title={QDER: Query-Specific Document and Entity Representations 
         for Multi-Vector Document Re-Ranking},
  author={Chatterjee, Shubham and Dalton, Jeff},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference},
  year={2025}
}
```