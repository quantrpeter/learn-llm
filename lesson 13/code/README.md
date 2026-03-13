# Lesson 13: Preference Alignment (RLHF / DPO)

Align a language model to human preferences — make outputs helpful, harmless, and honest.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- No other external dependencies

## How to Run

```bash
python3 preference_alignment.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Why Alignment Matters
SFT models follow instructions but still produce harmful, dishonest, or unhelpful outputs. This section shows concrete examples of alignment failures and introduces the HHH framework (Helpful, Harmless, Honest).

### Part 2 — The RLHF Pipeline
Conceptual overview of the three-stage RLHF pipeline: SFT → Reward Model → PPO. Explains why each stage exists, the data requirements, and why RLHF is difficult in practice (training instability, reward hacking, infrastructure cost).

### Part 3 — Reward Modeling
Train a reward model that scores response quality using the Bradley-Terry ranking loss. Demonstrates the training loop and evaluates accuracy on held-out preference pairs.

**Key functions:** `RewardModel`, `reward_loss()`

### Part 4 — DPO: Direct Preference Optimization
Full from-scratch implementation of the DPO loss function. Trains a toy language model with DPO on synthetic preference pairs and verifies that the policy shifts toward chosen responses.

**Key functions:** `dpo_loss()`, `sequence_log_prob()`, `TinyLM`

### Part 5 — Preference Datasets
How to create and format preference data (chosen/rejected pairs) for DPO training. Covers tokenization, batching, and the standard dataset format used by open-source alignment pipelines.

**Key functions:** `build_dpo_batch()`, `simple_tokenize()`

### Part 6 — KL Penalty
Prevent the aligned model from diverging too far from the reference policy. Implements KL divergence computation, demonstrates how divergence grows during training, and shows the KL-penalized reward formulation used in RLHF.

**Key functions:** `kl_divergence_categorical()`

## Exercises

5 exercises are printed at the end of the script:

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | Reward Model Exploration | Deeper architecture, wrong-label experiments |
| 2 | DPO Beta Sweep | Find optimal beta, understand instability |
| 3 | Implicit Reward from DPO | Extract the reward function DPO implicitly learns |
| 4 | KL Divergence Monitoring | Track divergence, implement early stopping |
| 5 | Complete Alignment Pipeline | End-to-end: data → DPO → evaluation |

## Key Takeaways

- **SFT is not enough** — models need preference signal to distinguish good from bad responses
- **RLHF** works but requires a reward model + PPO, which is complex and unstable
- **DPO** collapses reward modeling and RL into a single supervised loss
- **DPO loss** = `-log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))`
- **Beta** controls alignment strength: small beta → gentle, large beta → aggressive
- **KL penalty** prevents catastrophic forgetting and reward hacking
- **Preference data quality** is the single biggest factor in alignment success

## What's Next

[Lesson 14: Quantization & Inference Optimization](../../lesson%2014/) — Make your model smaller and faster with INT8/INT4 quantization, GPTQ, AWQ, and GGUF export.
