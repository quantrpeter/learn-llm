# Lesson 2: Python & Deep Learning Tooling

Get fluent with PyTorch — the core tool you'll use to build your LLM from scratch.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- Lesson 1 completed (we build on vectors, matrix multiply, softmax, cross-entropy)

## How to Run

```bash
python3 pytorch_fundamentals.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Tensors
Creating tensors, understanding shapes (1D vector, 2D matrix, 3D batch), reshaping with `view`/`reshape`/`unsqueeze`/`squeeze`, indexing, device placement (CPU/GPU), and dtype selection (float32, float16, bfloat16).

**Key concepts:** Every piece of data in an LLM — embeddings, weights, activations — is a tensor.

### Part 2 — Tensor Operations
Element-wise operations, matrix multiplication with `@`, broadcasting rules, reductions along dimensions (`sum`, `mean`, `max`), and softmax + cross-entropy loss using PyTorch (connecting back to Lesson 1's manual implementations).

**Key concepts:** These are the same math from Lesson 1, now running on GPU with one line of code.

### Part 3 — Autograd
Automatic differentiation with `requires_grad`, the forward-backward cycle, computational graphs, `detach()`, `torch.no_grad()`, and gradient accumulation for simulating large batch sizes.

**Key concepts:** Autograd replaces the manual gradient calculation from Lesson 1 — it traces every operation and computes all gradients automatically.

### Part 4 — Building with torch.nn
`nn.Linear`, activation functions (ReLU vs GELU), `nn.Sequential`, custom `nn.Module` classes, and parameter counting.

**Key concepts:** Every LLM layer (attention projections, FFN blocks) is built from these nn.Module components.

### Part 5 — The Training Loop
The complete training cycle: forward pass → loss → backward → optimizer.step → zero_grad. Training a simple model on synthetic data (y = 2x + 3) with AdamW optimizer, verifying convergence.

**Key concepts:** This exact loop, repeated billions of times, is how every LLM is trained.

### Part 6 — Practical Tools
Mixed precision training (`torch.amp`), gradient clipping, checkpoint saving/loading, and reproducibility with `torch.manual_seed`.

**Key concepts:** These tools are essential for training LLMs efficiently without running out of memory.

## Exercises

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | Tensor Shapes | Create 3D tensor, reduce along dim=1 |
| 2 | Manual Softmax | Reimplement softmax in PyTorch, verify against F.softmax |
| 3 | Custom MLP | Build a 3-layer nn.Module, count parameters by hand |
| 4 | Train on sin(x) | Full training loop on a non-linear function |
| 5 | Gradient Accumulation | Simulate large batches with 4 mini-batch accumulation |

## Key Takeaways

- **Tensors** are GPU-accelerated arrays — the universal data container in deep learning
- **Autograd** automatically computes all gradients via `loss.backward()`
- **nn.Module** is how you build reusable, composable neural network layers
- **Training loop** = forward → loss → backward → step → zero_grad (repeat forever)
- **AdamW** is the standard optimizer for LLM training
- **Mixed precision** cuts memory usage in half with minimal quality loss

## What's Next

[Lesson 3: Neural Network Building Blocks](../../lesson%203/) — Linear layers, activations (GELU), LayerNorm, dropout, and residual connections — every component inside a transformer.
