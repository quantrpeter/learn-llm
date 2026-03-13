# Lesson 7: Data Pipeline for Pre-Training

Prepare a large-scale text dataset for training your LLM — from raw documents to GPU-ready token batches.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- Lesson 4 completed (tokenization, BPE concepts)
- Lesson 6 completed (transformer architecture — you know what the model expects)

## How to Run

```bash
python3 data_pipeline.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Data Sources
Overview of where LLM training data comes from: Common Crawl, Wikipedia, books, code, FineWeb, The Pile. Simulates a multi-source text corpus with wiki, book, and code documents plus intentional duplicates and junk for the cleaning pipeline.

**Key function:** `generate_corpus()` — creates a realistic simulated corpus with multiple sources and quality levels.

### Part 2 — Data Cleaning
Implements a three-stage cleaning pipeline: hash-based deduplication (SHA-256), length filtering, and quality filtering (special character ratio, alphabetic ratio). Shows how ~15-20% of raw data gets removed — in production, FineWeb removes ~90% of Common Crawl.

**Key functions:** `deduplicate()`, `filter_by_length()`, `quality_score()`, `filter_by_quality()`, `clean_pipeline()`

### Part 3 — Data Formatting
Compares two approaches to creating fixed-length sequences from variable-length documents. Naive padding wastes massive compute on pad tokens. Sequence packing concatenates documents with separators and slices into chunks — zero waste, used by every major LLM.

**Key functions:** `pad_sequences()`, `pack_sequences()`

### Part 4 — DataLoader Design
Builds both map-style and streaming PyTorch Datasets. Map-style (`TextDataset`) supports random access and easy shuffling. Streaming (`StreamingTextDataset`) handles datasets too large for memory. Measures throughput of both approaches.

**Key classes:** `TextDataset`, `StreamingTextDataset`

### Part 5 — Data Mixing
Implements weighted sampling from multiple data sources (wiki, books, code) to hit target proportions. Uses `WeightedRandomSampler` to over/under-sample sources. The mixing ratio is a critical hyperparameter for model capabilities.

**Key classes:** `MixedDataset`

### Part 6 — Tokenization Integration
Connects a character-level tokenizer to the packing pipeline, producing `(input_ids, labels)` pairs ready for next-token prediction training. Runs 5 actual training steps to demonstrate the complete end-to-end flow: raw text → clean → tokenize → pack → batch → model → loss.

**Key classes:** `CharTokenizer`, `PreTrainingDataset`

## Exercises

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | Near-Duplicate Detection | Implement Jaccard similarity with shingling |
| 2 | Packing Attention Masks | Prevent cross-document attention contamination |
| 3 | Custom Collate Function | Build training batches with proper masking |
| 4 | Pipeline Throughput | Benchmark each stage, find the bottleneck |
| 5 | Dynamic Data Mixing | Curriculum learning with shifting proportions |

## Key Takeaways

- **Data quality > quantity** — cleaning removes 90%+ of raw web data, and that's a good thing
- **Sequence packing** eliminates padding waste, used universally for pre-training
- **Map-style vs streaming** — use map-style for small data, streaming for terabyte-scale
- **Data mixing proportions** directly control what capabilities your model develops
- **The full pipeline** is: raw text → clean → tokenize → pack → batch → GPU
- **Throughput matters** — your data pipeline must feed the GPU fast enough to not bottleneck training

## What's Next

[Lesson 8: Pre-Training Your LLM](../../lesson%208/) — Training objective, AdamW optimizer, learning rate schedules, loss curves, and checkpointing.
