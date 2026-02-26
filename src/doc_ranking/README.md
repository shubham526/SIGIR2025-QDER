
# How To Run The Document Ranking Code

## Module Structure

* **`train.py`**: Main script for training the model.
* **`test.py`**: Script for running inference and evaluation on test sets.
* **`qder_model.py`**: Core model architecture implementing the QDER logic (Attention, Interaction, Scoring).
* **`base_model.py`**: Abstract base class defining the model interface.
* **`dataset.py`**: Data loading logic for reading JSONL files with pre-computed embeddings.
* **`text_embedding.py`**: Wrapper for Hugging Face transformers (BERT, RoBERTa, etc.).
* **`trainer.py`**: Training loop utility.
* **`evaluate.py`**: Inference loop utility.
* **`utils.py` & `metrics.py**`: (Assumed dependencies) Utilities for saving runs and calculating MAP/nDCG.

## Prerequisites

* Python 3.8+
* PyTorch
* Transformers (Hugging Face)
* tqdm
* numpy

```bash
pip install torch transformers tqdm numpy pytrec_eval

```

## Data Format

The model expects data in **JSONL (JSON Lines)** format. Each line must be a valid JSON object representing a single training/testing example.

**Crucial:** The dataset must include pre-computed entity embeddings.

### Required Fields for JSONL Input:

| Field | Type | Description |
| --- | --- | --- |
| `query_id` | str | Unique identifier for the query. |
| `doc_id` | str | Unique identifier for the document. |
| `label` | int/float | Relevance label (e.g., 0 for non-relevant, 1 for relevant). |
| `query` | str | The raw text of the query. |
| `doc` | str | The raw text of the document/passage. |
| `query_ent_emb` | List[List[float]] | List of entity vectors for the query (Shape: `[num_entities, 300]`). |
| `doc_ent_emb` | List[List[float]] | List of entity vectors for the document (Shape: `[num_entities, 300]`). |
| `doc_score` | float | Initial retrieval score (e.g., BM25) used for weighting.

 |

> 
> **Note on Embeddings:** The paper uses **Wikipedia2Vec** (300-dimensional) embeddings. If a query or document has no entities, pass an empty list `[]`; the code handles padding automatically.
> 
> 

## Usage

### 1. Training

To train the model, use `train.py`. The standard configuration from the paper uses the "No-Subtract" interaction mode (Addition + Multiplication) and Bilinear scoring.

```bash
python train.py \
    --train "data/train.jsonl" \
    --dev "data/dev.jsonl" \
    --qrels "data/qrels.dev.tsv" \
    --save-dir "checkpoints/" \
    --save "qder_model.bin" \
    --text-enc "bert" \
    --score-method "bilinear" \
    --enabled-interactions "add,multiply" \
    --epoch 20 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --eval-every 1 \
    --use-cuda
```

**Key Arguments:**

* `--train`, `--dev`: Path to training/validation data (JSONL).
* `--qrels`: Path to ground truth file (TREC format) for validation metrics.
* `--checkpoint`: (Optional) Path to checkpoint to resume training from.
* `--text-enc`: Text encoder backbone (default: `bert`). Options: `bert`, `roberta`, `deberta`, etc.
* `--score-method`: `bilinear` (recommended) or `linear`.
* `--enabled-interactions`: Comma-separated list of interactions. Paper recommends `add,multiply`.
* `--epoch`: Number of training epochs (default: 20).
* `--batch-size`: Batch size for training (default: 8).
* `--learning-rate`: Learning rate (default: 1e-5, changed from paper's 2e-5 for stability).
* `--eval-every`: Evaluate every N epochs (default: 1).
* `--use-cuda`: Flag to enable GPU training.
* `--patience`: Early stopping patience in epochs (default: 5). Training stops if validation metric doesn't improve for this many epochs.

#### Resuming Training

To resume training from a saved checkpoint (e.g., after interruption or for fine-tuning):

**Resume Features:**

* Loads model weights, optimizer state, and scheduler state
* Continues from the last completed epoch
* Preserves training history and best validation metric
* Can adjust learning rate for fine-tuning

**Usage Examples:**

**1. Resume from interruption (continue training):**
```bash
python train.py \
    --train "data/train.jsonl" \
    --dev "data/dev.jsonl" \
    --qrels "data/qrels.dev.tsv" \
    --save-dir "checkpoints/" \
    --checkpoint "checkpoints/qder_model.bin" \
    --text-enc "bert" \
    --score-method "bilinear" \
    --enabled-interactions "add,multiply" \
    --epoch 30 \
    --learning-rate 1e-5 \
    --use-cuda
```

**2. Resume from specific epoch checkpoint:**
```bash
python train.py \
    --train "data/train.jsonl" \
    --dev "data/dev.jsonl" \
    --qrels "data/qrels.dev.tsv" \
    --save-dir "checkpoints/" \
    --checkpoint "checkpoints/best_model_epoch_015.bin" \
    --text-enc "bert" \
    --score-method "bilinear" \
    --enabled-interactions "add,multiply" \
    --epoch 25 \
    --learning-rate 1e-5 \
    --use-cuda
```

**3. Fine-tune with lower learning rate:**
```bash
python train.py \
    --train "data/train.jsonl" \
    --dev "data/dev.jsonl" \
    --qrels "data/qrels.dev.tsv" \
    --save-dir "checkpoints_finetune/" \
    --checkpoint "checkpoints/qder_model.bin" \
    --text-enc "bert" \
    --score-method "bilinear" \
    --enabled-interactions "add,multiply" \
    --epoch 10 \
    --learning-rate 5e-6 \
    --use-cuda
```

**4. Start fresh training (no checkpoint):**
```bash
python train.py \
    --train "data/train.jsonl" \
    --dev "data/dev.jsonl" \
    --qrels "data/qrels.dev.tsv" \
    --save-dir "checkpoints/" \
    --save "qder_model.bin" \
    --text-enc "bert" \
    --score-method "bilinear" \
    --enabled-interactions "add,multiply" \
    --epoch 20 \
    --batch-size 8 \
    --learning-rate 1e-5 \
    --eval-every 1 \
    --use-cuda
```

> **Note:** When resuming, the `--save` argument is optional since the model will save as `qder_model.bin` by default.

#### Training Output Files

The training process saves multiple files for analysis and resumption:
```
checkpoints/
├── epoch_001_dev.run              # Validation results for each epoch
├── epoch_002_dev.run
├── ...
├── epoch_020_dev.run
│
├── best_dev.run                   # Best validation results
├── qder_model.bin                 # Best model checkpoint (includes full state)
├── best_model_epoch_015.bin       # Best model with epoch number
└── training_history.json          # Complete training history
```

**Training History JSON includes:**

* Loss values for each epoch
* Validation metrics for each epoch
* Best epoch and best metric achieved
* Can be used for plotting training curves

### 2. Inference / Testing

To evaluate a trained model, use `test.py`. This generates a run file in standard TREC format.

```bash
python test.py \
    --test "data/test.jsonl" \
    --qrels "data/qrels.test.tsv" \
    --checkpoint "checkpoints/qder_model.bin" \
    --output "results/test.run" \
    --text-enc "bert" \
    --score-method "bilinear" \
    --enabled-interactions "add,multiply" \
    --metrics "map,ndcg" \
    --use-cuda

```

**Key Arguments:**

* `--checkpoint`: Path to the saved `.bin` model file from training.
* `--output`: Filename for the output run file (TREC format).
* `--metrics`: Comma-separated metrics to report (e.g., `map,ndcg`).

### 3. Ablation Studies

You can reproduce the ablation studies from the paper (Table 4) by modifying the `--enabled-interactions` argument:

* 
**No-Subtract (Paper Best):** `--enabled-interactions "add,multiply"` 


* **All Interactions:** `--enabled-interactions "add,multiply,subtract"`
* **Only Addition:** `--enabled-interactions "add"`
* **Only Multiplication:** `--enabled-interactions "multiply"`

## Model Configuration Details

The implementation accurately reflects the "No-Subtract" configuration found to be most effective in the paper.

* **Text Encoder:** Defaults to `bert-base-uncased`.
* **Entity Dim:** Hardcoded to `300` (matching Wikipedia2Vec).
* **Max Length:** Defaults to `512` tokens.
* **Learning Rate:** Defaults to `1e-5` (more stable than paper's `2e-5`).
* **Early Stopping:** Patience of 5 epochs (stops if no improvement).
* **Gradient Clipping:** Max norm of 1.0 to prevent exploding gradients.
* **Warmup Steps:** 1000 steps with linear warmup schedule.



```