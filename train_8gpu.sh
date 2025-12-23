#!/bin/bash
#
# CLaRa Training Script - 8 GPU Parallel
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
SAVE_MODEL_NAME="${SAVE_MODEL_NAME:-clara_8gpu}"
SAVE_PATH="${SAVE_PATH:-./checkpoints/${SAVE_MODEL_NAME}}"

# Training stage: stage1, stage1_2, stage2, stage2_reasoning
STAGE="${STAGE:-stage1}"

# Compression settings
DOC_MAX_LENGTH="${DOC_MAX_LENGTH:-256}"
COMPRESS_RATE="${COMPRESS_RATE:-32}"
GENERATION_TOP_K="${GENERATION_TOP_K:-1}"

# Training hyperparameters
MAX_EPOCHS="${MAX_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-128}"  # Global batch size (will be distributed across GPUs)
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-2}"  # Per GPU batch size
MAX_LEN="${MAX_LEN:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-100000}"

# Debug mode settings (color-coded confidence output)
# Only rank 0 will print debug info
DEBUG_MODE="${DEBUG_MODE:-true}"
DEBUG_EVERY_STEPS="${DEBUG_EVERY_STEPS:-10}"
DEBUG_MAX_PRINT_TOKENS="${DEBUG_MAX_PRINT_TOKENS:-100}"

# Logging
LOGGING_STEPS="${LOGGING_STEPS:-1}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-500}"

########################################
# Distributed Training Configuration
########################################
NUM_GPUS="${NUM_GPUS:-8}"
NUM_NODES="${NUM_NODES:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"

# For multi-node training, set these environment variables:
# - MASTER_ADDR: IP address of the master node
# - NODE_RANK: Rank of the current node (0 for master)

########################################
# Environment Setup
########################################
export PYTHONPATH="${SCRIPT_DIR}/..:${PYTHONPATH}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-5400}"

# For memory efficiency
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Optional: WandB configuration
export WANDB_DIR="${WANDB_DIR:-${SAVE_PATH}/wandb}"

########################################
# Create output directory
########################################
mkdir -p "$SAVE_PATH"
mkdir -p "$WANDB_DIR"

########################################
# Calculate world size
########################################
WORLD_SIZE=$((NUM_GPUS * NUM_NODES))

########################################
# Print configuration
########################################
echo "=============================================="
echo "CLaRa Training Configuration (8 GPU Parallel)"
echo "=============================================="
echo "Model: ${MODEL_PATH}"
echo "Stage: ${STAGE}"
echo "Dataset: ${DATASET}"
echo "Save path: ${SAVE_PATH}"
echo "Compress rate: ${COMPRESS_RATE}"
echo "Doc max length: ${DOC_MAX_LENGTH}"
echo "Debug mode: ${DEBUG_MODE}"
echo "----------------------------------------------"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Number of nodes: ${NUM_NODES}"
echo "World size: ${WORLD_SIZE}"
echo "Master address: ${MASTER_ADDR}:${MASTER_PORT}"
echo "Node rank: ${NODE_RANK}"
echo "Global batch size: ${BATCH_SIZE}"
echo "Micro batch size per GPU: ${MICRO_BATCH_SIZE}"
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
# Build distributed arguments for torchrun
########################################
DISTRIBUTED_ARGS="--nproc_per_node ${NUM_GPUS} \
    --nnodes ${NUM_NODES} \
    --rdzv_id 101 \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --master_addr ${MASTER_ADDR} \
    --master_port ${MASTER_PORT} \
    --node_rank ${NODE_RANK}"

########################################
# Run training
########################################
echo "Starting CLaRa training (${NUM_GPUS} GPUs)..."
echo "Distributed args: ${DISTRIBUTED_ARGS}"

if [ $NUM_NODES -gt 1 ]; then
    # Multi-node training: check for EFA (AWS Elastic Fabric Adapter)
    if command -v fi_info >/dev/null 2>&1; then
        echo "EFA detected, using high-speed networking..."
        fi_info -p efa -t FI_EP_RDM
    fi
fi

torchrun ${DISTRIBUTED_ARGS} -m ${TRAINING_CMD}

echo "=============================================="
echo "Training completed!"
echo "Model saved to: ${SAVE_PATH}"
echo "=============================================="

########################################
# Copy model files for reference
########################################
if [ -f "${SCRIPT_DIR}/../openrlhf/models/modeling_clara.py" ]; then
    cp "${SCRIPT_DIR}/../openrlhf/models/modeling_clara.py" "${SAVE_PATH}/"
    echo "Copied modeling_clara.py to ${SAVE_PATH}/"
fi
