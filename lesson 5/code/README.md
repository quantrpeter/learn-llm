# Lesson 5: Attention Mechanism — The Core Innovation

Deeply understand self-attention, the mechanism that makes LLMs possible.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- Lessons 1–4 completed (math, PyTorch, nn building blocks, tokenization)

## How to Run

```bash
python3 attention_mechanism.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Intuition: Soft Dictionary Lookup
Attention as a soft, differentiable dictionary: Query asks a question, Keys are the index, Values are the content. Similarity scores determine how much each value contributes to the output. Concrete example with 3 sentences.

**Key concepts:** Q/K/V, dot-product similarity, weighted sum of values.

### Part 2 — Scaled Dot-Product Attention
Full implementation from scratch: `scores = Q @ K^T`, scale by `1/sqrt(d_k)`, `weights = softmax(scores)`, `output = weights @ V`. Every step printed with shapes and intermediate values.

**Key concepts:** The scaling factor prevents softmax saturation when d_k is large.

### Part 3 — Multi-Head Attention
Split `d_model` into `n_heads` parallel attention heads, each operating on `d_k = d_model / n_heads` dimensions. Full `MultiHeadAttention` class with projection matrices W_q, W_k, W_v, W_o. Parameter counting included.

**Key concepts:** Multiple heads let the model attend to different relationship types simultaneously.

### Part 4 — Causal Masking
Lower-triangular mask ensures position i can only attend to positions 0..i, never the future. Visualized as a grid of allowed/blocked positions. Verified that masked positions get exactly zero attention weight.

**Key concepts:** This is what makes autoregressive generation work — training must match the inference constraint.

### Part 5 — KV Cache
Inference optimization: cache K and V from previous steps, only compute the new token's K and V each step. Step-by-step demo showing the cache growing. Verified that cached output matches full recomputation.

**Key concepts:** Reduces per-step cost from O(N·d) to O(d), making generation practical.

### Part 6 — Attention Complexity
Empirical demonstration of O(n²) scaling. Timing attention at sequence lengths 128, 256, 512, 1024 and showing the quadratic relationship. Memory analysis of the score matrix at various context lengths.

**Key concepts:** Quadratic cost is why long context is expensive and drives innovations like Flash Attention.

## Exercises

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | Single-Head Attention | Implement attention with raw tensor operations, no nn.Linear |
| 2 | Multi-Head Attention | Extend the MHA class to return attention weights, verify row sums |
| 3 | Causal Mask Verification | Prove that masking blocks all future positions |
| 4 | KV Cache Generate Loop | Process a prompt then generate tokens with growing cache |
| 5 | Time Scaling | Benchmark attention across sequence lengths, verify O(n²) |

## Key Takeaways

- **Attention** is a soft dictionary lookup — query compares to all keys, returns a weighted sum of values
- **Scaling by sqrt(d_k)** keeps dot products in a stable range for softmax
- **Multi-head** attention splits dimensions into parallel heads that learn different patterns
- **Causal mask** ensures each position can only see the past, enabling autoregressive generation
- **KV cache** avoids redundant computation during generation by storing past keys and values
- **O(n²) cost** means doubling sequence length quadruples compute — the fundamental bottleneck

## What's Next

[Lesson 6: The Transformer Architecture](../../lesson%206/) — Assemble attention, feed-forward networks, layer norm, and residual connections into a complete GPT-style decoder.
