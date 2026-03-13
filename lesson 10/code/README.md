# Lesson 10: Text Generation & Decoding

Generate text from a trained language model using different decoding strategies — and understand why the choice of strategy matters more than you think.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- Lesson 5 completed (attention mechanism — KV cache concept)
- Lesson 6 completed (transformer architecture)
- Lesson 8 completed (pre-training — next-token prediction)

## How to Run

```bash
python3 text_generation.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Greedy Decoding
Always pick the highest-probability token (argmax). Greedy is deterministic and fast, but produces repetitive output because locally optimal choices don't lead to globally optimal sequences.

**Key function:** `greedy_decode()` — baseline generation showing the repetition problem.

### Part 2 — Temperature Sampling
Scale logits by temperature before softmax to control randomness. Low T makes the distribution sharper (more deterministic), high T makes it flatter (more creative). Demonstrates T=0.5, 1.0, and 2.0 side by side.

**Key function:** `temperature_sample()` — the single most important generation hyperparameter.

### Part 3 — Top-k Sampling
Keep only the top k most likely tokens, set everything else to -inf, then sample. Prevents garbage tokens from the tail of the distribution. Implements the filtering from scratch and visualizes which tokens survive.

**Key function:** `top_k_sample()` — used in GPT-2's original release (k=40).

### Part 4 — Top-p (Nucleus) Sampling
Keep the smallest set of tokens whose cumulative probability is >= p. Unlike top-k's fixed size, top-p adapts: fewer tokens when the model is confident, more when uncertain. Demonstrates the adaptive behavior on confident vs uncertain distributions.

**Key function:** `top_p_sample()` — the default strategy for most modern LLM APIs.

### Part 5 — Repetition Penalty
Reduce logits of already-generated tokens to prevent the model from repeating itself. Positive logits are divided by the penalty, negative logits are multiplied. Shows the effect with penalties of 1.0, 1.2, and 1.5.

**Key function:** `apply_repetition_penalty()` — breaks repetition loops.

### Part 6 — KV Cache Generation
Full `generate()` function with KV caching. Instead of recomputing attention for all previous tokens at every step (O(L²)), cache the K and V projections and only compute the new token (O(L)). Benchmarks the speedup and shows cache memory growth.

**Key function:** `generate_with_kv_cache()` — the standard optimization for all production LLM inference.

## Exercises

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | Temperature Extremes | Explore T→0 and T→∞, measure entropy vs temperature |
| 2 | Top-k vs Top-p | Compare fixed-k vs adaptive-p on different distributions |
| 3 | Beam Search | Implement beam search, compare to greedy, understand trade-offs |
| 4 | Frequency + Presence Penalty | Implement OpenAI-style dual penalties |
| 5 | KV Cache Memory | Calculate cache sizes for real models, understand GQA and PagedAttention |

## Key Takeaways

- **Greedy decoding** is deterministic but repetitive — locally optimal ≠ globally optimal
- **Temperature** is the most fundamental control: T<1 = sharp/safe, T>1 = flat/creative
- **Top-k** prevents garbage tokens but uses a fixed vocabulary size regardless of confidence
- **Top-p (nucleus)** adapts to the distribution shape — the default for modern LLM APIs
- **Repetition penalty** breaks loops by reducing probability of already-seen tokens
- **KV caching** is essential for fast inference — avoids recomputing attention for past tokens

## What's Next

[Lesson 11: Evaluation & Benchmarking](../../lesson%2011/) — Perplexity, benchmark suites (HellaSwag, MMLU), evaluation harnesses, and measuring how good your model actually is.
