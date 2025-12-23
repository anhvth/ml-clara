# CLaRa Training Scripts

These training scripts are based on the **official CLaRa paper implementation** with added features from our custom implementation, including **color-coded confidence debug output**.

## Overview

CLaRa (Continuous Latent Reasoning Augmentation) uses a three-stage training approach:

1. **Stage 1 (Compression Pretraining)**: Train the compressor using SCP framework with QA pairs and paraphrases
2. **Stage 1_2 (Compression Instruction Tuning)**: Fine-tune the compressor on instruction-following tasks
3. **Stage 2 (End-to-End Training)**: Jointly train reranker and generator via a single language modeling loss

## Training Scripts

### Single GPU Training (`train_single_gpu.sh`)

```bash
# Basic usage with defaults
./train_single_gpu.sh

# Customize model and dataset
MODEL_PATH="mistralai/Mistral-7B-Instruct-v0.2" \
DATASET="path/to/your/data.jsonl" \
STAGE="stage1" \
./train_single_gpu.sh

# Stage 1_2 training (requires stage 1 checkpoint)
STAGE="stage1_2" \
PRETRAIN_CHECKPOINT="./checkpoints/stage1_model" \
./train_single_gpu.sh
```

### 8 GPU Parallel Training (`train_8gpu.sh`)

```bash
# Basic usage with 8 GPUs
./train_8gpu.sh

# Customize for your setup
MODEL_PATH="Qwen/Qwen3-14B" \
NUM_GPUS=8 \
BATCH_SIZE=128 \
MICRO_BATCH_SIZE=2 \
./train_8gpu.sh

# Multi-node training (set on each node)
MASTER_ADDR="192.168.1.100" \
NUM_NODES=2 \
NODE_RANK=0 \  # 0 for master, 1 for worker
./train_8gpu.sh
```

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `Qwen/Qwen3-14B` | Base model path |
| `PRETRAIN_CHECKPOINT` | (none) | Checkpoint for stage1_2/stage2 |
| `DATASET` | `example/pretrain_data.jsonl` | Training data path |
| `SAVE_PATH` | `./checkpoints/clara_*` | Output directory |
| `STAGE` | `stage1` | Training stage |
| `DOC_MAX_LENGTH` | `256` | Max document length |
| `COMPRESS_RATE` | `32` | Compression rate |
| `LEARNING_RATE` | `1e-4` | Learning rate |
| `BATCH_SIZE` | `4/128` | Global batch size |
| `MICRO_BATCH_SIZE` | `1/2` | Per-GPU batch size |
| `MAX_EPOCHS` | `1` | Training epochs |
| `DEBUG_MODE` | `true` | Enable debug output |
| `DEBUG_EVERY_STEPS` | `5/10` | Debug print frequency |

## Color-Coded Debug Output

When `DEBUG_MODE=true`, the training will print color-coded output showing model confidence for each token:

- 🟢 **Green**: High confidence (probability close to 1.0)
- 🔴 **Red**: Low confidence (probability close to 0.0)

This helps identify:
- Which tokens the model struggles with
- Whether compression preserves semantic information
- Training progress over time

Example output:
```
================================================================================
[Debug] CLaRa Training Debug - Sample 0
================================================================================
[Debug] Question: What is the capital of France?
[Debug] Expected Answer: Paris
[Debug] Answer metrics
  token_acc : 78.50% (12/15)
  ce_loss   : 0.234567

[Debug] Answer colored by P(gold token) - red=low confidence, green=high confidence:
<MEM...>*4 Question: What is the capital of France?
Assistant: Paris is the capital of France...
================================================================================
```

## Data Format

### Stage 1 (Pretraining)

```json
{
    "data_type": "qa",
    "question": ["Question 1", "Question 2"],
    "answers": ["Answer 1", "Answer 2"],
    "docs": ["Document content"]
}
```

### Stage 1_2 (Instruction Tuning)

```json
{
    "question": "What is X?",
    "answers": "X is...",
    "docs": ["Document 1", "Document 2"]
}
```

### Stage 2 (End-to-End)

```json
{
    "question": "Question text",
    "docs": ["Doc 1", "Doc 2", ...],
    "gold_answer": "Reference answer",
    "pos_index": [0, 2]
}
```

## Tips

1. **Start with Stage 1**: Always train stage 1 first with QA loss and MSE loss
2. **Memory Efficiency**: Use `gradient_checkpointing` for large models
3. **Debug Mode**: Enable initially to monitor training, disable for production runs
4. **Compression Rate**: Start with 32x, adjust based on your needs (higher = more compression, less info)

## Troubleshooting

### Out of Memory
- Reduce `MICRO_BATCH_SIZE`
- Enable `gradient_checkpointing`
- Use a smaller model

### Training Diverges
- Reduce `LEARNING_RATE`
- Check data format
- Enable debug mode to inspect predictions

### Slow Training
- Increase `MICRO_BATCH_SIZE` if memory allows
- Use more GPUs
- Enable `flash_attn`
