#!/bin/bash

###############################################################################
# 5-Fold Cross-Validation Pipeline for QDER
#
# Prerequisites:
# - Step 1 already completed: Master stacked entity run exists
# - Entity embeddings available
# - Fold assignment file exists
###############################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration - EDIT THESE PATHS
MASTER_ENTITY_RUN="data/master_entity_run.txt"       # Stacked entity run from step 1
FOLD_FILE="data/folds.json"                          # Fold assignment file
QUERIES_TSV="data/queries.tsv"                       # Query ID → Query text
DOCS_TSV="data/docs.tsv"                             # Doc ID → Doc text
QRELS="data/qrels.txt"                               # Relevance judgments
DOC_RUN="data/bm25.run"                              # Initial doc run (e.g., BM25)
ENTITY_EMBEDDINGS="data/enwiki_20180420_300d.pkl"   # Wikipedia2Vec embeddings file
WORK_DIR="data/cv_data"                              # Working directory
RESULTS_DIR="results/5fold_cv"                       # Results output

# make_doc_ranking_data.py parameters
NUM_EXPANSION_ENTITIES=20                            # Number of expansion entities (--k)
BALANCE_DATA=false                                   # Whether to balance training data

# Model configuration
TEXT_ENC="bert"
SCORE_METHOD="bilinear"
ENABLED_INTERACTIONS="add,multiply"
EPOCHS=20
BATCH_SIZE=8
LEARNING_RATE=1e-5
MAX_LEN=512

# Create directories
mkdir -p "$WORK_DIR"
mkdir -p "$RESULTS_DIR"
for fold in {0..4}; do
    mkdir -p "$WORK_DIR/fold-$fold"
    mkdir -p "$RESULTS_DIR/fold-$fold"
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}5-Fold Cross-Validation Pipeline${NC}"
echo -e "${BLUE}========================================${NC}\n"

###############################################################################
# STEP 2: Create Master Training Doc Data
###############################################################################
echo -e "${GREEN}[Step 2/6]${NC} Creating master training document data..."

if [ ! -f "$WORK_DIR/master_train.jsonl" ]; then
    if [ "$BALANCE_DATA" = true ]; then
        BALANCE_FLAG="--balance"
    else
        BALANCE_FLAG=""
    fi

    python make_doc_ranking_data.py \
        --queries "$QUERIES_TSV" \
        --docs "$DOCS_TSV" \
        --qrels "$QRELS" \
        --doc-run "$DOC_RUN" \
        --entity-run "$MASTER_ENTITY_RUN" \
        --embeddings "$ENTITY_EMBEDDINGS" \
        --k "$NUM_EXPANSION_ENTITIES" \
        --train \
        $BALANCE_FLAG \
        --save "$WORK_DIR/master_train.jsonl" \
        --save-stats "$WORK_DIR/master_train_stats.json"

    echo -e "${GREEN}✓${NC} Master training data created: $WORK_DIR/master_train.jsonl"
    echo -e "${GREEN}✓${NC} Statistics saved: $WORK_DIR/master_train_stats.json"
else
    echo -e "${YELLOW}⚠${NC} Master training data already exists, skipping..."
fi

###############################################################################
# STEP 3: Create Master Testing Doc Data
###############################################################################
echo -e "\n${GREEN}[Step 3/6]${NC} Creating master testing document data..."

if [ ! -f "$WORK_DIR/master_test.jsonl" ]; then
    python make_doc_ranking_data.py \
        --queries "$QUERIES_TSV" \
        --docs "$DOCS_TSV" \
        --qrels "$QRELS" \
        --doc-run "$DOC_RUN" \
        --entity-run "$MASTER_ENTITY_RUN" \
        --embeddings "$ENTITY_EMBEDDINGS" \
        --k "$NUM_EXPANSION_ENTITIES" \
        --save "$WORK_DIR/master_test.jsonl" \
        --save-stats "$WORK_DIR/master_test_stats.json"

    echo -e "${GREEN}✓${NC} Master testing data created: $WORK_DIR/master_test.jsonl"
    echo -e "${GREEN}✓${NC} Statistics saved: $WORK_DIR/master_test_stats.json"
else
    echo -e "${YELLOW}⚠${NC} Master testing data already exists, skipping..."
fi

###############################################################################
# STEP 4: Split Master Training Data by Fold
###############################################################################
echo -e "\n${GREEN}[Step 4/6]${NC} Splitting master training data into folds..."

python split_data_by_fold.py \
    --folds "$FOLD_FILE" \
    --data "$WORK_DIR/master_train.jsonl" \
    --save "$WORK_DIR" \
    --split training

echo -e "${GREEN}✓${NC} Training data split into folds"

###############################################################################
# STEP 5: Split Master Testing Data by Fold (Validation & Test)
###############################################################################
echo -e "\n${GREEN}[Step 5/6]${NC} Splitting master testing data into validation and test sets..."

# Create validation splits
echo "  Creating validation splits..."
python split_data_by_fold.py \
    --folds "$FOLD_FILE" \
    --data "$WORK_DIR/master_test.jsonl" \
    --save "$WORK_DIR" \
    --split validation

# Create testing splits
echo "  Creating testing splits..."
python split_data_by_fold.py \
    --folds "$FOLD_FILE" \
    --data "$WORK_DIR/master_test.jsonl" \
    --save "$WORK_DIR" \
    --split testing

echo -e "${GREEN}✓${NC} Testing data split into validation and test folds"

###############################################################################
# STEP 5.5: Split QRELs by Fold
###############################################################################
echo -e "\n${GREEN}[Step 5.5/6]${NC} Splitting QRELs into folds..."

# Split QRELs for validation
python split_run_or_qrels_by_fold.py \
    --folds "$FOLD_FILE" \
    --file "$QRELS" \
    --save "$WORK_DIR" \
    --split validation \
    --type qrels

# Split QRELs for testing
python split_run_or_qrels_by_fold.py \
    --folds "$FOLD_FILE" \
    --file "$QRELS" \
    --save "$WORK_DIR" \
    --split testing \
    --type qrels

echo -e "${GREEN}✓${NC} QRELs split into folds"

###############################################################################
# Optional: Split Queries TSV by Fold (if needed for reference)
###############################################################################
echo -e "\n${GREEN}[Optional]${NC} Splitting queries TSV into folds..."

python split_tsv_by_fold.py \
    --folds "$FOLD_FILE" \
    --file "$QUERIES_TSV" \
    --save "$WORK_DIR"

echo -e "${GREEN}✓${NC} Queries TSV split into folds"

###############################################################################
# STEP 6: Train, Validate, and Test on Each Fold
###############################################################################
echo -e "\n${GREEN}[Step 6/6]${NC} Training and evaluating on each fold..."
echo -e "${BLUE}========================================${NC}\n"

# Summary arrays
declare -a FOLD_RESULTS

for fold in {0..4}; do
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}Processing Fold $fold${NC}"
    echo -e "${BLUE}========================================${NC}"

    FOLD_DIR="$WORK_DIR/fold-$fold"
    SAVE_DIR="$RESULTS_DIR/fold-$fold"

    # Check if data exists
    if [ ! -f "$FOLD_DIR/training.jsonl" ]; then
        echo -e "${RED}✗${NC} Training data not found for fold $fold: $FOLD_DIR/training.jsonl"
        continue
    fi

    if [ ! -f "$FOLD_DIR/validation.jsonl" ]; then
        echo -e "${RED}✗${NC} Validation data not found for fold $fold: $FOLD_DIR/validation.jsonl"
        continue
    fi

    if [ ! -f "$FOLD_DIR/testing.jsonl" ]; then
        echo -e "${RED}✗${NC} Testing data not found for fold $fold: $FOLD_DIR/testing.jsonl"
        continue
    fi

    # Training
    echo -e "\n${YELLOW}Training on fold $fold...${NC}"
    python train.py \
        --train "$FOLD_DIR/training.jsonl" \
        --dev "$FOLD_DIR/validation.jsonl" \
        --qrels "$FOLD_DIR/validation.qrels.txt" \
        --save-dir "$SAVE_DIR" \
        --save "model.bin" \
        --text-enc "$TEXT_ENC" \
        --score-method "$SCORE_METHOD" \
        --enabled-interactions "$ENABLED_INTERACTIONS" \
        --epoch "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --max-len "$MAX_LEN" \
        --eval-every 1 \
        --use-cuda \
        --run "dev.run" \
        2>&1 | tee "$SAVE_DIR/training.log"

    # Check if training succeeded
    if [ ! -f "$SAVE_DIR/model.bin" ]; then
        echo -e "${RED}✗${NC} Training failed for fold $fold - checkpoint not found"
        continue
    fi

    # Testing
    echo -e "\n${YELLOW}Testing on fold $fold...${NC}"
    python test.py \
        --test "$FOLD_DIR/testing.jsonl" \
        --qrels "$FOLD_DIR/testing.qrels.txt" \
        --checkpoint "$SAVE_DIR/model.bin" \
        --output "$SAVE_DIR/test.run" \
        --text-enc "$TEXT_ENC" \
        --score-method "$SCORE_METHOD" \
        --enabled-interactions "$ENABLED_INTERACTIONS" \
        --max-len "$MAX_LEN" \
        --batch-size "$BATCH_SIZE" \
        --metrics "map,ndcg" \
        --use-cuda \
        2>&1 | tee "$SAVE_DIR/testing.log"

    # Extract test metrics
    if [ -f "$SAVE_DIR/testing.log" ]; then
        TEST_MAP=$(grep "MAP:" "$SAVE_DIR/testing.log" | tail -1 | awk '{print $2}')
        TEST_NDCG=$(grep "NDCG@20:" "$SAVE_DIR/testing.log" | tail -1 | awk '{print $2}')

        if [ -n "$TEST_MAP" ] && [ -n "$TEST_NDCG" ]; then
            FOLD_RESULTS[$fold]="Fold $fold: MAP=$TEST_MAP, nDCG@20=$TEST_NDCG"
            echo -e "${GREEN}✓${NC} Fold $fold complete: MAP=$TEST_MAP, nDCG@20=$TEST_NDCG"
        else
            echo -e "${YELLOW}⚠${NC} Could not extract metrics for fold $fold"
            FOLD_RESULTS[$fold]="Fold $fold: Metrics extraction failed"
        fi
    fi

    echo -e "${BLUE}========================================${NC}\n"
done

###############################################################################
# Summary
###############################################################################
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}5-Fold Cross-Validation Complete${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo "Results by fold:"
for result in "${FOLD_RESULTS[@]}"; do
    echo "  $result"
done

echo -e "\nDetailed results saved in: $RESULTS_DIR"
echo -e "\nTo calculate average metrics, run:"
echo "  python calculate_cv_average.py --results-dir $RESULTS_DIR"

echo -e "\n${GREEN}All done!${NC}"