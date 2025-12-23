#!/bin/bash
#
# CLaRa Training Script - Single GPU
# Based on the official CLaRa paper implementation
# With color-coded confidence debug output feature
#

set -ex

########################################
# Configuration - Modify these paths
########################################
# Model configuration
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"  # Base model path (can use Mistral, Qwen, etc.)
PRETRAIN_CHECKPOINT="${PRETRAIN_CHECKPOINT:-}"  # Set this for stage1_2/stage2 training

# Dataset paths - Use example data by default
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/example}"
DATASET="${DATASET:-${DATA_PATH}/pretrain_data.jsonl}"

# Output configuration
SAVE_MODEL_NAME="${SAVE_MODEL_NAME:-clara_single_gpu}"
SAVE_PATH="${SAVE_PATH:-./checkpoints/${SAVE_MODEL_NAME}}"

# Training stage: stage1, stage1_2, stage2, stage2_reasoning
STAGE="${STAGE:-stage1}"

# Compression settings
DOC_MAX_LENGTH="${DOC_MAX_LENGTH:-256}"
COMPRESS_RATE="${COMPRESS_RATE:-32}"
GENERATION_TOP_K="${GENERATION_TOP_K:-1}"

# Training hyperparameters
MAX_EPOCHS="${MAX_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"  # Global batch size
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"  # Per GPU batch size
MAX_LEN="${MAX_LEN:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-1000}"

# Debug mode settings (color-coded confidence output)
DEBUG_MODE="${DEBUG_MODE:-true}"
DEBUG_EVERY_STEPS="${DEBUG_EVERY_STEPS:-5}"
DEBUG_MAX_PRINT_TOKENS="${DEBUG_MAX_PRINT_TOKENS:-100}"

# Logging
LOGGING_STEPS="${LOGGING_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-50}"
SAVE_STEPS="${SAVE_STEPS:-100}"

# Distributed training
MASTER_PORT="${MASTER_PORT:-29501}"

########################################
# Environment Setup
########################################
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# For memory efficiency
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

########################################
# Create output directory
########################################
mkdir -p "$SAVE_PATH"

########################################
# Print configuration
########################################
echo "=============================================="
echo "CLaRa Training Configuration (Single GPU)"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Stage: ${STAGE}"
echo "Dataset: ${DATASET}"
echo "Save path: ${SAVE_PATH}"
echo "Compress rate: ${COMPRESS_RATE}"
echo "Doc max length: ${DOC_MAX_LENGTH}"
echo "Debug mode: ${DEBUG_MODE}"
echo "=============================================="

########################################
# Build training command
########################################
TRAINING_CMD="openrlhf.cli.train_sft \
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
    --max_samples ${MAX_SAMPLES} \
    --save_path ${SAVE_PATH} \
    --ckpt_path ${SAVE_PATH} \
    --save_steps ${SAVE_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --eval_steps ${EVAL_STEPS} \
    --zero_stage 2 \
    --bf16 \
    --flash_attn \
    --gradient_checkpointing"

# Add stage-specific flags
if [ "$STAGE" = "stage1" ]; then
    TRAINING_CMD="${TRAINING_CMD} --qa_loss --mse_loss"
elif [ "$STAGE" = "stage1_2" ]; then
    TRAINING_CMD="${TRAINING_CMD} --mse_loss --do_eval_gen"
    if [ -n "$PRETRAIN_CHECKPOINT" ]; then
        TRAINING_CMD="${TRAINING_CMD} --pretrain_checkpoint ${PRETRAIN_CHECKPOINT}"
    fi
elif [ "$STAGE" = "stage2" ]; then
    TRAINING_CMD="${TRAINING_CMD} --qa_loss --do_eval_gen"
    if [ -n "$PRETRAIN_CHECKPOINT" ]; then
        TRAINING_CMD="${TRAINING_CMD} --pretrain_checkpoint ${PRETRAIN_CHECKPOINT}"
    fi
fi

# Add debug mode flags
if [ "$DEBUG_MODE" = "true" ]; then
    TRAINING_CMD="${TRAINING_CMD} --debug_mode --debug_every_steps ${DEBUG_EVERY_STEPS} --debug_max_print_tokens ${DEBUG_MAX_PRINT_TOKENS}"
fi

########################################
# Run training (single GPU with torchrun for DeepSpeed compatibility)
########################################
echo "Starting CLaRa training (single GPU)..."
echo "Command: python -m ${TRAINING_CMD}"

torchrun --nproc_per_node=1 --master_port=${MASTER_PORT} -m ${TRAINING_CMD}

echo "=============================================="
echo "Training completed!"
echo "Model saved to: ${SAVE_PATH}"
echo "=============================================="
