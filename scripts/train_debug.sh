#!/bin/bash
#
# CLaRa Training Script - MacBook Debug Version
# Simplified for local development with MPS
#

set -ex

########################################
# Fixed Configuration - All params hardcoded
########################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model - using smallest variant for quick testing
MODEL_PATH="Qwen/Qwen2.5-0.5B"

# Dataset - use example data
DATA_PATH="${SCRIPT_DIR}/example"
DATASET="${DATA_PATH}/pretrain_data.jsonl"

# Output
SAVE_MODEL_NAME="clara_debug_mps"
SAVE_PATH="./checkpoints/${SAVE_MODEL_NAME}"

# Training stage
STAGE="stage1"

# Compression settings
DOC_MAX_LENGTH=128
COMPRESS_RATE=16
GENERATION_TOP_K=1

# Training hyperparameters - small for debugging
MAX_EPOCHS=50
LEARNING_RATE=1e-4
ENCODER_LR=1e-4  # Encoder adapter learning rate (can be different from decoder)
DECODER_LR=1e-8  # Decoder adapter learning rate
BATCH_SIZE=2
MICRO_BATCH_SIZE=1
MAX_LEN=512
MAX_SAMPLES=5

# Debug mode
DEBUG_MODE=true
DEBUG_EVERY_STEPS=5
DEBUG_MAX_PRINT_TOKENS=50

# Logging
LOGGING_STEPS=1
EVAL_STEPS=50
SAVE_STEPS=100

########################################
# Environment Setup
########################################
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Force MPS device
export PYTORCH_ENABLE_MPS_FALLBACK=1

########################################
# Create output directory
########################################
mkdir -p "$SAVE_PATH"

########################################
# Print configuration
########################################
echo "=============================================="
echo "CLaRa MacBook Debug Training (MPS)"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Stage: ${STAGE}"
echo "Dataset: ${DATASET}"
echo "Save path: ${SAVE_PATH}"
echo "Compress rate: ${COMPRESS_RATE}"
echo "Doc max length: ${DOC_MAX_LENGTH}"
echo "Device: MPS"
echo "=============================================="

########################################
# Run training with python directly
########################################
echo "Starting CLaRa training on MPS..."

python -m openrlhf.cli.train_sft \
    --pretrain ${MODEL_PATH} \
    --stage ${STAGE} \
    --dataset ${DATASET} \
    --max_len ${MAX_LEN} \
    --doc_max_length ${DOC_MAX_LENGTH} \
    --compress_rate ${COMPRESS_RATE} \
    --generation_top_k ${GENERATION_TOP_K} \
    --train_batch_size ${BATCH_SIZE} \
    --micro_train_batch_size ${MICRO_BATCH_SIZE} \
    --max_epochs ${MAX_EPOCHS} \
    --learning_rate ${LEARNING_RATE} \
    --encoder_lr ${ENCODER_LR} \
    --decoder_lr ${DECODER_LR} \
    --max_samples ${MAX_SAMPLES} \
    --save_path ${SAVE_PATH} \
    --ckpt_path ${SAVE_PATH} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --gradient_checkpointing \
    --qa_loss \
    --mse_loss \
    --debug_mode \
    --debug_every_steps ${DEBUG_EVERY_STEPS} \
    --debug_max_print_tokens ${DEBUG_MAX_PRINT_TOKENS} \
    --local_rank -1

echo "=============================================="
echo "Training completed!"
echo "Model saved to: ${SAVE_PATH}"
echo "=============================================="
