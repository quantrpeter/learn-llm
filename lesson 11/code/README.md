# Lesson 11: Evaluation & Benchmarking

Measure how good your model actually is — understand what the numbers mean and when to trust them.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- Lesson 8 completed (pre-training your LLM)
- Lesson 10 completed (text generation & decoding)

## How to Run

```bash
python3 evaluation.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Perplexity
The primary metric for language models, implemented from scratch. Computes perplexity as exp(average cross-entropy loss), demonstrates how it tracks model quality (random baseline → trained model), and shows corpus-level evaluation across multiple sequences.

**Key functions:** `compute_perplexity_from_logits()`, `compute_perplexity_manual()`, `corpus_perplexity()`

### Part 2 — Benchmark Suites
How standardized benchmarks (HellaSwag, ARC, MMLU, TruthfulQA) actually work under the hood. Implements completion scoring (log-probability ranking) and multiple-choice evaluation. Includes a reference table of published scores from GPT-2 through GPT-4.

**Key concepts:** Log-likelihood scoring, completion ranking, normalized accuracy. HellaSwag tests commonsense, MMLU tests knowledge breadth.

### Part 3 — Few-Shot Evaluation
Implements 0-shot and k-shot prompting strategies from GPT-3. Builds few-shot prompts programmatically, shows how examples in the prompt improve accuracy without any fine-tuning, and runs a comparative evaluation across shot counts.

**Key function:** `build_few_shot_prompt()`, `evaluate_few_shot()`

### Part 4 — Eval Harness
Explains how to use EleutherAI's `lm-evaluation-harness` — the standard tool for reproducible LLM benchmarking. Covers installation, CLI usage for HuggingFace and GGUF models, and simulates the internal pipeline (prompt formatting → inference → scoring → aggregation).

**Key concepts:** `lm_eval` CLI, task registry, standardized evaluation for leaderboard comparability.

### Part 5 — Human Evaluation
When automated metrics fail: A/B testing between models, Elo rating systems (from chess), and the LMSYS Chatbot Arena concept. Simulates a full A/B test and a multi-model Elo tournament, showing how rankings emerge from pairwise comparisons.

**Key functions:** `simulate_ab_test()`, `update_elo()`

### Part 6 — Contamination Checks
Detecting benchmark leakage in training data using n-gram overlap. Implements a contamination detector, demonstrates it on clean vs. contaminated examples, and shows a full decontamination pipeline that removes tainted training documents.

**Key functions:** `ngram_overlap_ratio()`, `decontaminate_dataset()`

## Exercises

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | Perplexity Deep Dive | Compute PPL at different model quality levels |
| 2 | Build a Benchmark | Create a 20-question MC benchmark with scoring |
| 3 | Elo Rating Convergence | Simulate tournament, study convergence speed |
| 4 | N-gram Contamination | Build and test a contamination detector |
| 5 | Full Evaluation Pipeline | End-to-end eval: PPL + benchmark + contamination |

## Key Takeaways

- **Perplexity** = exp(cross-entropy) — lower is better; a PPL of 25 means "choosing among 25 equally likely tokens"
- **Benchmarks** test specific capabilities (reasoning, knowledge, truthfulness) beyond raw next-token prediction
- **Few-shot prompting** lets you evaluate without fine-tuning — more shots generally helps
- **lm-evaluation-harness** is the standard tool — always use it for comparable results
- **Human evaluation** (A/B testing, Elo) captures qualities that benchmarks miss
- **Contamination checks** (n-gram overlap) are essential — inflated scores from memorization are meaningless

## What's Next

[Lesson 12: Supervised Fine-Tuning (SFT)](../../lesson%2012/) — Turn a base language model into an instruction-following assistant using instruction datasets, chat templates, LoRA, and QLoRA.
