# Lesson 6: The Transformer Architecture

Assemble all components into a complete transformer decoder (GPT-style) and understand every design choice.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- Lesson 3 completed (LayerNorm, residual connections, FFN)
- Lesson 4 completed (tokenization, embeddings)
- Lesson 5 completed (attention mechanism, multi-head attention, causal masking)

## How to Run

```bash
python3 transformer_architecture.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Encoder-Decoder vs Decoder-Only
Explains the original Transformer's encoder-decoder design (built for translation) vs the GPT-style decoder-only architecture. Why decoder-only won: simpler, unified, scalable, and better for generative tasks.

**Key concepts:** Decoder-only uses causal masking to replace the entire encoder. GPT, LLaMA, Mistral, Claude all use decoder-only.

### Part 2 — Transformer Block
Implements the full repeating unit of a GPT: CausalSelfAttention → LayerNorm → FeedForward → LayerNorm, with residual connections. Uses pre-norm (normalize before each sub-layer, not after).

**Key classes:** `CausalSelfAttention`, `FeedForward`, `TransformerBlock`

### Part 3 — Positional Encoding
Two approaches to injecting position information into the model. Sinusoidal PE (original Transformer) uses fixed sin/cos functions. RoPE (Rotary Position Embeddings) encodes position by rotating query/key vectors — used by LLaMA, Mistral, and most modern LLMs.

**Key functions:** `SinusoidalPE`, `precompute_rope_freqs()`, `apply_rope()`

### Part 4 — Full GPT Model
Stacks all components into a complete model: token embedding + positional embedding → N transformer blocks → final LayerNorm → LM head. Complete forward pass from token IDs to logits.

**Key classes:** `GPT` — the full model that takes token IDs in and produces next-token logits out.

### Part 5 — Model Dimensions
Explains the hyperparameters that define a transformer: `d_model`, `n_heads`, `n_layers`, `d_ff`, `d_head`. Shows GPT-2 (124M) configuration, counts parameters analytically, and compares all GPT-2 variants.

**Key function:** `count_parameters_detailed()` — breaks down params by component.

### Part 6 — Generating Text
Simple greedy autoregressive generation loop. Feeds tokens in, takes logits from the last position, picks the argmax, appends, and repeats. Demonstrates with random weights (gibberish output — training makes it coherent).

**Key function:** `generate_greedy()` — the autoregressive decoding loop.

## Exercises

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | Transformer Block | Build attention + FFN + residual + pre-norm from scratch |
| 2 | RoPE | Implement rotary position embeddings and verify position-awareness |
| 3 | Full GPT | Assemble token emb + blocks + LM head into a complete model |
| 4 | Parameter Counting | Match GPT-2 124M analytically, understand scaling |
| 5 | Text Generation | Greedy decoding from random weights, decode to characters |

## Key Takeaways

- **Decoder-only** is the dominant architecture — simpler than encoder-decoder for language modeling
- **Transformer block** = Attention + FFN + Residual + LayerNorm, repeated N times
- **Pre-norm** (normalize before sub-layers) is more stable than post-norm
- **RoPE** is the modern standard for positional encoding (LLaMA, Mistral)
- **GPT-2 124M** = vocab 50257, d_model 768, 12 heads, 12 layers, d_ff 3072
- **Generation** is autoregressive — one token at a time, O(n²) per step

## What's Next

[Lesson 7: Data Pipeline for Pre-Training](../../lesson%207/) — Data sources, cleaning, tokenization pipelines, streaming DataLoaders, and sequence packing.
