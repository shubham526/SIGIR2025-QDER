# How To Run The Entity Ranking Code

## Understanding MonoBert vs DuoBert

### **The Correct Mapping**

| Model | Task | Training Approach | Output | Loss Function |
|-------|------|-------------------|--------|---------------|
| **MonoBert** | Classification (Pointwise) | Single query-doc pairs with binary labels | 2 classes: [non-relevant, relevant] | CrossEntropyLoss |
| **DuoBert** | Ranking (Pairwise) | Query + positive doc vs query + negative doc | Single relevance score | MarginRankingLoss |

### **Why This Mapping?**

**MonoBert** = **Classification** (Pointwise):
- Processes ONE query-document pair at a time
- Outputs a binary classification: "Is this document relevant?"
- Training data: Individual (query, document, label) triples
- Label: 0 (non-relevant) or 1 (relevant)

**DuoBert** = **Ranking** (Pairwise):
- Processes TWO documents per query: positive and negative
- Learns to rank: "Which document is MORE relevant?"
- Training data: (query, doc_positive, doc_negative) triples
- No explicit label needed - positive should score > negative

## Data Format Requirements

### For Classification Task (MonoBert)

**Training Data Format:**
```json
{
  "query": "what is machine learning",
  "doc_text": "Machine learning is a subset of AI...",
  "label": 1,
  "query_id": "q1",
  "doc_id": "d1"
}
```

**Fields:**
- `query`: Query text
- `doc_text`: Document text
- `label`: 0 (non-relevant) or 1 (relevant)
- `query_id`: Query identifier
- `doc_id`: Document identifier

### For Ranking Task (DuoBert)

**Training Data Format:**
```json
{
  "query": "what is machine learning",
  "doc_pos_text": "Machine learning is a subset of AI...",
  "doc_neg_text": "The weather today is sunny...",
  "query_id": "q1",
  "doc_id": "d1"
}
```

**Fields:**
- `query`: Query text
- `doc_pos_text`: Positive (relevant) document text
- `doc_neg_text`: Negative (non-relevant) document text
- `query_id`: Query identifier
- `doc_id`: Document identifier (for the positive doc)

**Test/Dev Data Format (same for both tasks):**
```json
{
  "query": "what is machine learning",
  "doc_text": "Machine learning is a subset of AI...",
  "label": 1,
  "query_id": "q1",
  "doc_id": "d1"
}
```

## Usage

### Training with Classification (MonoBert)

```bash
python train.py \
    --train train.jsonl \
    --dev dev.jsonl \
    --save-dir ./checkpoints \
    --qrels dev.qrels.txt \
    --task classification \
    --mode cls \
    --query-enc bert \
    --batch-size 8 \
    --epoch 20 \
    --use-cuda \
    --cuda 0
```

### Training with Ranking (DuoBert)

```bash
python train.py \
    --train train_pairwise.jsonl \
    --dev dev.jsonl \
    --save-dir ./checkpoints \
    --qrels dev.qrels.txt \
    --task ranking \
    --mode cls \
    --query-enc bert \
    --batch-size 8 \
    --epoch 20 \
    --use-cuda \
    --cuda 0
```

### Testing

```bash
# Test with MonoBert
python test.py \
    --test test.jsonl \
    --run test.run \
    --checkpoint ./checkpoints/model.bin \
    --task classification \
    --use-cuda

# Test with DuoBert
python test.py \
    --test test.jsonl \
    --run test.run \
    --checkpoint ./checkpoints/model.bin \
    --task ranking \
    --use-cuda
```

## Arguments

### Common Arguments

- `--train`: Path to training data (JSONL format)
- `--dev`: Path to development/validation data
- `--test`: Path to test data
- `--save-dir`: Directory to save checkpoints
- `--qrels`: Ground truth relevance judgments (TREC format)
- `--task`: **[IMPORTANT]** `classification` or `ranking`
- `--mode`: Pooling mode (`cls` or `pooling`)
- `--query-enc`: Encoder type (bert, roberta, distilbert, etc.)
- `--max-len`: Maximum sequence length (default: 512)
- `--batch-size`: Batch size (default: 8)
- `--learning-rate`: Learning rate (default: 2e-5)
- `--epoch`: Number of epochs (default: 20)
- `--use-cuda`: Enable CUDA
- `--cuda`: CUDA device number

## How Training Works

### Classification (MonoBert)

1. **Input**: Single (query, document) pair
2. **Forward Pass**: `model(input_ids, attention_mask, segment_ids)`
3. **Output**: 2-dimensional logits [score_non_relevant, score_relevant]
4. **Loss**: CrossEntropyLoss between output and binary label
5. **Evaluation**: Use softmax probability of relevant class (index 1)

### Ranking (DuoBert)

1. **Input**: (query, doc_positive) and (query, doc_negative) pairs
2. **Forward Pass**: 
   - `score_pos = model(query + doc_positive)`
   - `score_neg = model(query + doc_negative)`
3. **Output**: Single relevance score for each document
4. **Loss**: MarginRankingLoss ensures score_pos > score_neg by margin=1
5. **Evaluation**: Use raw relevance score

## Model Architecture

Both MonoBert and DuoBert use the same base architecture from `model.py`:

```python
# MonoBert - outputs 2 classes
class MonoBert(Bert):
    def __init__(self, pretrained: str, mode: str = 'cls'):
        Bert.__init__(self, pretrained, mode)
        self._dense = nn.Linear(self._config.hidden_size, 2)

# DuoBert - outputs 1 score
class DuoBert(Bert):
    def __init__(self, pretrained: str, mode: str = 'cls'):
        Bert.__init__(self, pretrained, mode)
        self._dense = nn.Linear(self._config.hidden_size, 1)
```

## Available Encoders

- `bert`: bert-base-uncased
- `distilbert`: distilbert-base-uncased
- `roberta`: roberta-base
- `deberta`: microsoft/deberta-base
- `ernie`: nghuyong/ernie-2.0-base-en
- `electra`: google/electra-small-discriminator
- `conv-bert`: YituTech/conv-bert-base
- `t5`: t5-base

## Pooling Modes

- `cls`: Uses [CLS] token representation (recommended)
- `pooling`: Uses pooled output from BERT


## When to Use Which?

### Use Classification (MonoBert) when:
- You have labeled data with binary relevance judgments
- You want to classify documents as relevant/non-relevant
- You have limited training data
- You want faster training (processes one doc at a time)

### Use Ranking (DuoBert) when:
- You have or can create positive/negative pairs
- Your goal is to rank documents by relevance
- You want better ranking performance
- You can afford the computational cost of pairwise training

## Important Notes

1. **Task consistency**: Use the same `--task` when loading a checkpoint for testing as you used during training

2. **Data preparation**: 
   - Classification task needs: `doc_text` field
   - Ranking task needs: `doc_pos_text` and `doc_neg_text` fields

3. **Evaluation**: Both tasks use the same evaluation format (pointwise) - the difference is only in training

4. **Checkpoint compatibility**: MonoBert and DuoBert checkpoints are NOT interchangeable due to different output dimensions

## References

- [MonoBERT and DuoBERT paper](https://arxiv.org/abs/1901.04085)
- [OpenMatch Framework](https://github.com/thunlp/OpenMatch)