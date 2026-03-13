# Lesson 8: Pre-Training Your LLM

Train a language model from scratch — understand every component of the pre-training process from objective to checkpoint.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- Lesson 2 completed (PyTorch fundamentals, autograd, training loop basics)
- Lesson 6 completed (transformer architecture, GPT model)
- Lesson 7 completed (data pipeline concepts)

## How to Run

```bash
python3 pretraining.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Training Objective
Next-token prediction: the model sees [A, B, C, D] and learns to predict [B, C, D, E]. Every position is a training example. This single objective — causal language modeling — teaches grammar, facts, reasoning, and code.

**Key concepts:** Input→target shift by one position. A sequence of N tokens gives N-1 training pairs.

### Part 2 — Loss Function
Cross-entropy loss over the full vocabulary at every position. Connects back to Lesson 1's cross-entropy. Shows manual computation vs `nn.CrossEntropyLoss`, and explains perplexity as the exponential of the loss.

**Key concepts:** Reshape `(B, T, V) → (B*T, V)` for loss. Perplexity benchmarks from random (50k) to strong (5).

### Part 3 — Optimizer
AdamW: the standard optimizer for LLM pre-training. Manual step-by-step implementation showing momentum (β1), adaptive learning rates (β2), bias correction, and decoupled weight decay. Compares SGD vs Adam vs AdamW convergence speed.

**Key concepts:** Weight decay applied to matmul weights only, not biases/norms. β2=0.95 for LLMs (not 0.999).

### Part 4 — Learning Rate Schedule
Implements warmup + cosine decay from scratch — the exact schedule used by GPT-3 and LLaMA. Shows why warmup is necessary (stabilizes early training) and plots the full schedule as ASCII art.

**Key function:** `get_lr(step, warmup_steps, max_steps, max_lr, min_lr)`

### Part 5 — Training Loop
Complete pre-training loop with a tiny GPT model (~200K params) on synthetic text data. Tracks loss, perplexity, gradient norms, and learning rate. Includes proper weight decay grouping, gradient clipping, and a greedy generation test after training.

**Key classes:** `CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `TinyGPT`

### Part 6 — Checkpointing
Save and load model weights, optimizer state, training step, config, and RNG state. Demonstrates resuming training from a checkpoint with no loss in quality. Explains why saving optimizer state is critical.

**Key concepts:** Always save optimizer state. Without it, Adam's moment estimates reset and training quality drops.

## Exercises

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | LR Schedule | Build a 3-phase schedule (warmup → constant → cosine) |
| 2 | Manual AdamW | Implement AdamW from scratch and match PyTorch's output |
| 3 | Longer Corpus | Train on richer text, evaluate generation quality |
| 4 | Hyperparameter Sweep | Compare learning rates and model sizes systematically |
| 5 | Checkpoint Robustness | Verify checkpoint resume produces identical results |

## Key Takeaways

- **Next-token prediction** is the sole pre-training objective — it teaches everything
- **Cross-entropy loss** averaged over all positions; perplexity = exp(loss)
- **AdamW** is the universal LLM optimizer — momentum + adaptive LR + decoupled weight decay
- **Warmup + cosine decay** prevents early instability and smoothly reduces LR
- **Gradient clipping** (max_norm=1.0) prevents loss spikes from large gradients
- **Checkpointing** must save model + optimizer + step + RNG to resume perfectly

## What's Next

[Lesson 9: Distributed Training & Scaling](../../lesson%209/) — Data parallelism (DDP), FSDP, mixed precision, Flash Attention, gradient checkpointing, and Chinchilla scaling laws.
