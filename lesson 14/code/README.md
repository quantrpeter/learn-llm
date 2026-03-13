# Lesson 14: Quantization & Inference Optimization

Make your model smaller and faster for real-world deployment.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- No other external dependencies

## How to Run

```bash
python3 quantization.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Why Quantize?
Memory math for a 7B parameter model across FP32, FP16, INT8, and INT4. Shows how quantization enables running large models on consumer hardware — a 7B model goes from 28 GB (FP32) to 3.5 GB (INT4).

### Part 2 — Weight Quantization (Per-Tensor INT8)
From-scratch implementation of affine INT8 quantization: computing scale and zero_point, quantizing weights, dequantizing, and measuring the resulting error. Demonstrates the effect on matrix multiply outputs.

**Key functions:** `quantize_tensor_int8()`, `dequantize_tensor_int8()`

### Part 3 — Per-Channel Quantization
Independent scale/zero_point per output channel instead of one for the whole tensor. Shows dramatic error reduction when weight rows have different magnitudes — the standard approach in production quantization.

**Key functions:** `quantize_per_channel_int8()`, `dequantize_per_channel_int8()`

### Part 4 — GPTQ Concept
Conceptual walkthrough of GPTQ (post-training quantization with calibration data). Implements Hessian-based weight sensitivity analysis and simplified column-by-column error compensation. Explains the full pipeline from calibration data to quantized weights.

### Part 5 — Speculative Decoding
Simulates the draft-verify paradigm: a small draft model proposes tokens, a large target model verifies them in parallel. Measures acceptance rates across multiple rounds and discusses when speculative decoding helps most.

**Key classes:** `DraftModel`, `TargetModel`, `speculative_decode()`

### Part 6 — KV Cache Optimization (GQA vs MQA vs MHA)
Side-by-side implementation of Multi-Head Attention, Grouped Query Attention, and Multi-Query Attention. Computes KV cache sizes at scale and shows how GQA achieves 4-8x cache reduction with minimal quality loss.

**Key classes:** `MultiHeadAttention`, `GroupedQueryAttention`, `MultiQueryAttention`

## Exercises

5 exercises are printed at the end of the script:

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | INT4 Quantization | Implement 4-bit quantization, compare to INT8 |
| 2 | Quantize a Full MLP | End-to-end quantization of a multi-layer network |
| 3 | Speculative Decoding Sweep | Measure acceptance rate vs draft accuracy |
| 4 | GQA Group Size Sweep | Find the sweet spot between memory and quality |
| 5 | Calibration Data Impact | Test how calibration set size affects GPTQ quality |

## Key Takeaways

- **7B FP32 = 28 GB**, INT8 = 7 GB, INT4 = 3.5 GB — quantization enables consumer hardware
- **Per-channel** quantization is dramatically better than per-tensor for real weight matrices
- **GPTQ** uses Hessian-based sensitivity to minimize output error during quantization
- **Speculative decoding** gives 2-3x speedup with zero quality loss by draft-verify parallelism
- **GQA** reduces KV cache by 4-8x by sharing K/V heads across query head groups
- **MQA** is the extreme case (1 KV head) — maximum savings, slight quality trade-off

## What's Next

[Lesson 15: Serving & Deployment](../../lesson%2015/) — Deploy your LLM as a usable API with vLLM, streaming responses, and production monitoring.
