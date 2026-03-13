# Lesson 12: Supervised Fine-Tuning (SFT)

Turn a raw, pre-trained language model into an instruction-following chat assistant — the step that makes base models useful.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- Lesson 8 completed (pre-training your LLM)
- Lesson 11 completed (evaluation & benchmarking)

## How to Run

```bash
python3 supervised_finetuning.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Base vs Chat Models
Why raw pre-trained LLMs don't follow instructions. A base model is a completion engine — it continues text rather than answering questions. Simulates the probability shift that SFT creates, boosting answer tokens and suppressing continuation patterns.

**Key insight:** SFT teaches the model a new pattern: instruction in → helpful answer out.

### Part 2 — Instruction Datasets
The two dominant formats for SFT data: Alpaca (single-turn instruction/input/output triples) and ShareGPT (multi-turn conversations). Includes sample data in both formats and a survey of popular open-source datasets.

**Key insight:** 1,000 high-quality examples often beat 100,000 noisy ones.

### Part 3 — Chat Templates
How conversations are encoded into token sequences using ChatML, LLaMA-2, and Alpaca templates. Shows the same conversation formatted three ways, and demonstrates that the model only sees tokens — role boundaries are defined by the template.

**Key functions:** `format_chatml()`, `format_llama2()`, `format_alpaca()`

### Part 4 — SFT Training (Loss Masking)
The critical implementation detail: compute cross-entropy loss ONLY on assistant tokens. System and user tokens provide context but don't contribute to the loss. Implements full training loop with `ignore_index=-100` masking.

**Key insight:** Without masking, the model wastes capacity learning to mimic user questions instead of producing good answers.

### Part 5 — LoRA (Low-Rank Adaptation)
Implements LoRA from scratch: W' = W + BA where A is random and B is zeros at initialization. Shows that LoRA starts as an identity adapter (no disruption), computes parameter savings at different ranks, and trains LoRA vs full fine-tuning side-by-side.

**Key class:** `LoRALinear` — drop-in replacement for `nn.Linear` with trainable low-rank adapters.

### Part 6 — QLoRA Concept
Quantize base weights to 4-bit (NF4), keep LoRA adapters in FP16. Shows the complete memory math: Full FT (112 GB) → LoRA (16 GB) → QLoRA (6 GB). Covers NF4, double quantization, and paged optimizers.

**Key insight:** QLoRA reduced fine-tuning cost by ~50× — from 8× A100 GPUs to a single consumer card.

## Exercises

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | Instruction Formatting | Convert Alpaca data to ChatML template |
| 2 | Loss Masking | Build masked labels, verify non-assistant tokens don't affect loss |
| 3 | LoRA Merge | Merge adapters back into base weights for zero-overhead inference |
| 4 | Rank Sensitivity | Sweep LoRA ranks, find the parameter-efficiency sweet spot |
| 5 | Multi-Turn SFT | Build label masks for multi-turn conversations |

## Key Takeaways

- **Base models complete text**; SFT teaches them to **follow instructions**
- **Chat templates** encode role boundaries into the token stream — mismatched templates break the model
- **Loss masking** on assistant tokens only is the single most important SFT implementation detail
- **LoRA** (W' = W + BA) trains <1% of parameters with comparable quality to full fine-tuning
- **QLoRA** quantizes frozen base weights to 4-bit, enabling fine-tuning on consumer GPUs

## What's Next

[Lesson 13: Preference Alignment (RLHF / DPO)](../../lesson%2013/) — Reward modeling, PPO, DPO, and aligning your assistant to human preferences.
