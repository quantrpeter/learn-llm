# Lesson 15: Serving & Deployment

Deploy your LLM as a usable API or application — serving frameworks, streaming, throughput optimization, cost math, and production guardrails.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- No other external dependencies (no actual server needed)

## How to Run

```bash
python3 serving_deployment.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Serving Frameworks
Compare the three major LLM serving frameworks: vLLM (PagedAttention, highest throughput), TGI (Rust-based, HuggingFace ecosystem), and llama.cpp (C++, CPU/Metal, local deployment). Includes startup commands and a PagedAttention memory efficiency analysis.

**Key concepts:** PagedAttention, KV cache management, framework trade-offs

### Part 2 — API Design (OpenAI-Compatible)
The OpenAI Chat Completions API format — request/response JSON structure, key fields, and request validation. Your self-hosted model becomes a drop-in replacement for any OpenAI-compatible client.

**Key functions:** `validate_chat_request()`

### Part 3 — Streaming Responses (SSE)
Server-Sent Events protocol for token-by-token streaming. Simulates realistic token generation with TTFT (time-to-first-token) and inter-token latency. Compares streaming vs non-streaming user experience.

**Key functions:** `format_sse_chunk()`, `SimulatedLLMServer`

### Part 4 — Throughput Optimization (Continuous Batching)
The single biggest throughput improvement in LLM serving. Compares naive (static) batching — where short requests wait for long ones — against continuous batching, which immediately fills empty slots. Includes side-by-side benchmarks.

**Key functions:** `NaiveBatchingSimulator`, `ContinuousBatchingSimulator`

### Part 5 — Cost Estimation
Cloud GPU pricing comparison (T4 through H100), cost-per-token calculations, and break-even analysis for self-hosting vs API. Covers tokens/second/dollar as the key decision metric.

### Part 6 — Guardrails
Production safety pipeline: input filtering (prompt injection detection), output filtering (PII redaction), token-bucket rate limiting, and a full end-to-end guardrail pipeline.

**Key functions:** `InputFilter`, `OutputFilter`, `TokenBucketRateLimiter`, `guardrail_pipeline()`

## Exercises

5 exercises are printed at the end of the script:

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | Mock Server | Build an OpenAI-compatible server with streaming and guardrails |
| 2 | Advanced Continuous Batching | Per-request latency, Poisson arrivals, p50/p95 metrics |
| 3 | GPU Selection Optimizer | Auto-select cheapest GPU for target throughput and latency |
| 4 | Prompt Injection Defense | Expand filters, canary tokens, false-positive measurement |
| 5 | Cost Dashboard | Track tokens, compute running costs, project monthly spend |

## Key Takeaways

- **vLLM** (PagedAttention) is the go-to for high-throughput GPU serving; **llama.cpp** for local/edge
- **OpenAI-compatible API** format makes your model a drop-in replacement for commercial APIs
- **Streaming (SSE)** reduces perceived latency from seconds to ~100ms time-to-first-token
- **Continuous batching** gives 2-4x throughput over naive batching by filling empty GPU slots
- **Cost math**: tokens/second/dollar is the key metric; self-hosting beats API above ~40K requests/day
- **Guardrails** are non-negotiable: input filters, output filters, rate limiting, defense-in-depth

## What's Next

[Lesson 16: Retrieval-Augmented Generation (RAG)](../../lesson%2016/) — Extend your LLM with external knowledge using embedding models, vector databases, and retrieval pipelines.
