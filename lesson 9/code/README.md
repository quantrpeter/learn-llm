# Lesson 9: Distributed Training & Scaling

Learn the techniques that make it possible to train models too large for a single GPU — and how to choose the right model size.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- Lesson 6 completed (transformer architecture)
- Lesson 8 completed (pre-training your LLM)

## How to Run

```bash
python3 distributed_training.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Data Parallelism (DDP)
The simplest multi-GPU strategy: replicate the full model on every GPU, split data, and synchronize gradients via all-reduce. Shows the DDP wrapping pattern, simulates all-reduce, and explains effective batch size scaling.

**Key concepts:** All-reduce, gradient synchronization, `DistributedDataParallel`, effective batch size.

### Part 2 — FSDP (Fully Sharded Data Parallelism)
When models don't fit on a single GPU, FSDP shards parameters, gradients, and optimizer states across GPUs. Computes per-GPU memory for DDP vs ZeRO Stages 1/2/3, showing how a 7B model that needs 112 GB with DDP fits in 28 GB per GPU with FSDP.

**Key function:** `memory_per_gpu()` — calculates per-GPU memory for each sharding strategy.

### Part 3 — Tensor Parallelism
Splits individual weight matrices across GPUs. Demonstrates column-parallel and row-parallel splitting of a Linear(4096, 4096), verifying that the split outputs match the full computation.

**Key concepts:** Column parallel (split output dim), row parallel (split input dim), concat vs all-reduce.

### Part 4 — Mixed Precision Training
Uses FP16/BF16 for computation while keeping FP32 master weights. Shows 2× memory savings, `torch.amp.autocast` dtype behavior, and the difference between FP16 (needs GradScaler) and BF16 (same range as FP32, no scaling needed).

**Key concepts:** `torch.amp.autocast`, `GradScaler`, FP32 vs FP16 vs BF16 precision formats.

### Part 5 — Flash Attention
Standard attention materializes an O(n²) score matrix in HBM. Flash Attention computes attention in SRAM tiles, reducing memory to O(n) and HBM reads by 2-4×. Shows concrete memory savings at different sequence lengths.

**Key concepts:** IO-bound attention, tiled computation, online softmax, `scaled_dot_product_attention`.

### Part 6 — Scaling Laws
Implements the Chinchilla scaling law: given a compute budget C, the optimal split is N = sqrt(C/120) parameters and D = 20N tokens. Compares real models (GPT-3, Chinchilla, LLaMA) against optimality and predicts loss with the parametric formula.

**Key functions:** `chinchilla_optimal()`, `predicted_loss()`, `compute_flops()`.

## Exercises

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | All-Reduce | Simulate 8-GPU gradient sync, verify correctness |
| 2 | FSDP Memory | Calculate per-GPU memory for different model sizes |
| 3 | Tensor Parallelism | Split multi-head attention across 4 GPUs |
| 4 | Mixed Precision | Measure memory savings with BF16 training |
| 5 | Scaling Laws | Plan compute-optimal training for a given budget |

## Key Takeaways

- **DDP** replicates the model and syncs gradients — simplest but requires the model to fit on one GPU
- **FSDP** shards everything across GPUs — enables training models 4× larger than GPU memory
- **Tensor Parallelism** splits individual layers — used within a node (fast NVLink)
- **BF16** is the standard precision for LLM training — 2× savings, no overflow risk
- **Flash Attention** reduces memory from O(n²) to O(n) — essential for long sequences
- **Chinchilla law**: optimal tokens ≈ 20× parameters — balance model size with data

## What's Next

[Lesson 10: Text Generation & Decoding](../../lesson%2010/) — Greedy, top-k, top-p, temperature sampling, KV cache, and generation speed optimization.
