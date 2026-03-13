"""
============================================================================
 LESSON 11: Evaluation & Benchmarking
============================================================================

 Goal: Measure how good your model actually is — and understand what
       the numbers mean.

 Topics covered:
   1. Perplexity        — the primary LM metric, implemented from scratch
   2. Benchmark Suites  — HellaSwag, ARC, MMLU, TruthfulQA
   3. Few-Shot Evaluation — 0-shot and 5-shot prompting strategies
   4. Eval Harness      — using lm-evaluation-harness for standardized eval
   5. Human Evaluation  — when metrics fail: A/B testing, Elo, Chatbot Arena
   6. Contamination Checks — detecting benchmark leakage in training data

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites: Lesson 8 (pre-training), Lesson 10 (text generation)
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from collections import Counter

torch.manual_seed(42)
random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — PERPLEXITY
# ═══════════════════════════════════════════════════════════════════════════
#
# Perplexity is THE standard metric for language models.
#
#   perplexity = exp(average cross-entropy loss)
#
# Intuition: a perplexity of 25 means the model is "as confused as if
# it were choosing uniformly among 25 tokens" at each step.
#
#   - Random model on vocab of 50,000  →  PPL ≈ 50,000
#   - GPT-2 (124M) on WikiText-103     →  PPL ≈ 29
#   - GPT-3 (175B) on WikiText-103     →  PPL ≈ 15
#   - LLaMA 2 (70B) on WikiText-2      →  PPL ≈ 3.3
#
# Lower is always better. Always.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: PERPLEXITY")
print("=" * 70)


def compute_perplexity_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute perplexity from model logits and target token IDs.

    WHY THIS MATTERS:
    This is exactly what you compute after every training epoch to track
    model quality. During pre-training, you watch perplexity decrease:
    50000 → 500 → 50 → 20 → ... as the model learns language.

    Args:
        logits:  (seq_len, vocab_size) — raw model outputs
        targets: (seq_len,) — ground truth token IDs

    Returns:
        Perplexity (float). Lower is better.
    """
    ce_loss = F.cross_entropy(logits, targets)
    return torch.exp(ce_loss).item()


def compute_perplexity_manual(log_probs_at_targets: torch.Tensor) -> float:
    """
    Compute perplexity from scratch — no PyTorch loss function.

    perplexity = exp( -1/N * sum(log P(token_i | context)) )

    This is the definition: the exponential of the average negative
    log-probability of each token given its context.
    """
    avg_neg_log_prob = -log_probs_at_targets.mean()
    return torch.exp(avg_neg_log_prob).item()


# --- 1.1 Perplexity from scratch -----------------------------------------

vocab_size = 100
seq_len = 20

torch.manual_seed(42)
logits = torch.randn(seq_len, vocab_size)
targets = torch.randint(0, vocab_size, (seq_len,))

# Method 1: using cross-entropy loss
ppl_ce = compute_perplexity_from_logits(logits, targets)

# Method 2: manual computation
log_probs = F.log_softmax(logits, dim=-1)
log_probs_at_targets = log_probs[torch.arange(seq_len), targets]
ppl_manual = compute_perplexity_manual(log_probs_at_targets)

print(f"\n--- Perplexity computation ---")
print(f"Vocab size: {vocab_size}, Sequence length: {seq_len}")
print(f"Method 1 (cross-entropy):  PPL = {ppl_ce:.2f}")
print(f"Method 2 (manual):         PPL = {ppl_manual:.2f}")
print(f"Match: {abs(ppl_ce - ppl_manual) < 0.01}")


# --- 1.2 How perplexity changes with model quality -----------------------

"""
WHY THIS MATTERS:
This demonstrates the core intuition: a model that assigns higher
probability to the correct tokens has lower perplexity.
During training, you watch this number drop.
"""

print(f"\n--- Perplexity vs model quality ---")

torch.manual_seed(42)
targets = torch.randint(0, vocab_size, (seq_len,))

# Simulate models of different quality
scenarios = {
    "Random (untrained)": torch.zeros(seq_len, vocab_size),
    "Slightly trained":   torch.zeros(seq_len, vocab_size),
    "Well trained":       torch.zeros(seq_len, vocab_size),
    "Near perfect":       torch.zeros(seq_len, vocab_size),
}

for i, tok in enumerate(targets):
    # Random: uniform logits
    scenarios["Random (untrained)"][i] = torch.randn(vocab_size) * 0.01

    # Slightly trained: correct token gets a small boost
    scenarios["Slightly trained"][i] = torch.randn(vocab_size) * 0.5
    scenarios["Slightly trained"][i, tok] += 2.0

    # Well trained: correct token gets a large boost
    scenarios["Well trained"][i] = torch.randn(vocab_size) * 0.3
    scenarios["Well trained"][i, tok] += 5.0

    # Near perfect: correct token dominates
    scenarios["Near perfect"][i] = torch.randn(vocab_size) * 0.1
    scenarios["Near perfect"][i, tok] += 10.0

for name, logits in scenarios.items():
    ppl = compute_perplexity_from_logits(logits, targets)
    avg_prob = F.softmax(logits, dim=-1)[torch.arange(seq_len), targets].mean().item()
    print(f"  {name:22s}  PPL = {ppl:8.2f}  avg P(correct) = {avg_prob:.4f}")

print(f"\n  Baseline: random model with vocab {vocab_size} → PPL ≈ {vocab_size}")
print(f"  Key insight: lower PPL = model assigns higher probability to correct tokens")


# --- 1.3 Perplexity over a corpus ----------------------------------------

def corpus_perplexity(all_logits: list[torch.Tensor],
                      all_targets: list[torch.Tensor]) -> float:
    """
    Compute perplexity over multiple sequences (an entire test set).

    WHY THIS MATTERS:
    You never evaluate on a single sequence. You compute PPL over the
    entire held-out test set (e.g., WikiText-103 test split = 245K tokens).
    The per-token losses are averaged across ALL tokens, then exponentiated.
    """
    total_loss = 0.0
    total_tokens = 0
    for logits, targets in zip(all_logits, all_targets):
        ce = F.cross_entropy(logits, targets, reduction="sum")
        total_loss += ce.item()
        total_tokens += targets.numel()
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


torch.manual_seed(42)
test_sequences = []
for _ in range(5):
    sl = random.randint(10, 30)
    lg = torch.randn(sl, vocab_size)
    tg = torch.randint(0, vocab_size, (sl,))
    # give correct tokens a moderate boost to simulate a trained model
    for j, t in enumerate(tg):
        lg[j, t] += 3.0
    test_sequences.append((lg, tg))

all_logits = [s[0] for s in test_sequences]
all_targets = [s[1] for s in test_sequences]
corpus_ppl = corpus_perplexity(all_logits, all_targets)
total_toks = sum(t.numel() for t in all_targets)
print(f"\n--- Corpus-level perplexity ---")
print(f"Test corpus: {len(test_sequences)} sequences, {total_toks} total tokens")
print(f"Corpus PPL = {corpus_ppl:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — BENCHMARK SUITES
# ═══════════════════════════════════════════════════════════════════════════
#
# Perplexity tells you the model is good at next-token prediction.
# But can it actually REASON? Answer questions? Understand the world?
#
# That's what benchmarks test. The standard ones:
#
#   HellaSwag   — Commonsense reasoning (sentence completion)
#   ARC         — Grade-school science questions (multiple choice)
#   MMLU        — 57-subject knowledge test (high school to expert)
#   TruthfulQA  — Resistance to common misconceptions
#   WinoGrande  — Pronoun disambiguation (commonsense)
#
# Each benchmark converts complex abilities into a single accuracy number.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: BENCHMARK SUITES")
print("=" * 70)


# --- 2.1 HellaSwag-style evaluation (completion scoring) ------------------

"""
WHY THIS MATTERS:
HellaSwag tests commonsense reasoning by asking the model to pick the
most likely continuation of a sentence. The model never sees explicit
"choose A/B/C/D" — instead you compute the log-probability of each
continuation and pick the one the model finds most likely.

This is how most benchmarks actually work under the hood.
"""

print(f"\n--- HellaSwag-style: Completion Scoring ---")


def score_completion(model_logits_fn, context_ids: list[int],
                     completion_ids: list[int]) -> float:
    """
    Score a completion by summing log P(token | context + preceding tokens).

    This is how LM-based benchmarks work:
    1. Feed [context + completion] through the model
    2. Sum the log-probs of ONLY the completion tokens
    3. Higher score = model thinks this completion is more likely
    """
    full_seq = context_ids + completion_ids
    logits = model_logits_fn(full_seq)

    total_log_prob = 0.0
    for i, token_id in enumerate(completion_ids):
        pos = len(context_ids) + i - 1
        token_logits = logits[pos]
        log_probs = F.log_softmax(token_logits, dim=-1)
        total_log_prob += log_probs[token_id].item()

    return total_log_prob


torch.manual_seed(42)

def mock_model_logits(token_ids: list[int]) -> torch.Tensor:
    """Simulates a model that assigns plausible completions higher probability."""
    seq_len = len(token_ids)
    logits = torch.randn(seq_len, vocab_size) * 0.5
    for i in range(1, seq_len):
        logits[i - 1, token_ids[i]] += 1.5
    return logits


context = [10, 20, 30, 40]
completions = {
    "A (plausible)":   [50, 51, 52],
    "B (random)":      [77, 88, 99],
    "C (implausible)": [1, 1, 1],
    "D (semi-ok)":     [50, 88, 52],
}

print(f"Context token IDs: {context}")
scores = {}
for label, comp in completions.items():
    s = score_completion(mock_model_logits, context, comp)
    scores[label] = s
    print(f"  {label:22s}  tokens={comp}  score = {s:.4f}")

best = max(scores, key=scores.get)
print(f"  Model picks: {best}")
print(f"  (HellaSwag accuracy = fraction of correctly picked completions)")


# --- 2.2 Multiple-choice benchmark format (ARC / MMLU style) -------------

"""
WHY THIS MATTERS:
ARC and MMLU present questions with 4 answer choices. The model scores
each choice and picks the one with the highest log-probability.
MMLU covers 57 subjects from elementary math to professional law.
"""

print(f"\n--- ARC / MMLU-style: Multiple Choice ---")


def multiple_choice_accuracy(
    model_logits_fn,
    questions: list[dict],
) -> float:
    """
    Evaluate multiple-choice accuracy.

    Each question has:
      - 'context': list of token IDs (the question)
      - 'choices': list of lists of token IDs (each answer)
      - 'correct': index of the correct answer (0-based)

    Returns accuracy (0.0 to 1.0).
    """
    correct_count = 0
    for q in questions:
        best_score = float("-inf")
        best_idx = -1
        for i, choice in enumerate(q["choices"]):
            score = score_completion(model_logits_fn, q["context"], choice)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx == q["correct"]:
            correct_count += 1
    return correct_count / len(questions)


sample_questions = [
    {
        "context": [5, 10, 15],
        "choices": [[20, 21], [30, 31], [40, 41], [50, 51]],
        "correct": 3,
    },
    {
        "context": [7, 14, 21],
        "choices": [[28, 35], [10, 11], [99, 98], [55, 56]],
        "correct": 0,
    },
    {
        "context": [2, 4, 6],
        "choices": [[8, 10], [3, 5], [12, 14], [7, 9]],
        "correct": 0,
    },
]

acc = multiple_choice_accuracy(mock_model_logits, sample_questions)
print(f"Sample benchmark: {len(sample_questions)} questions")
print(f"Accuracy: {acc:.1%}")
print(f"Random baseline (4 choices): 25.0%")


# --- 2.3 Benchmark reference scores --------------------------------------

print(f"\n--- Published benchmark scores (for reference) ---")
print(f"{'Model':<25s} {'HellaSwag':>10s} {'ARC-C':>8s} {'MMLU':>8s} {'TruthfulQA':>12s}")
print("-" * 65)
benchmarks = [
    ("Random baseline",       25.0, 25.0, 25.0, 25.0),
    ("GPT-2 (124M)",          31.6, 22.7, 25.9, 40.7),
    ("LLaMA 1 (7B)",          76.1, 47.6, 35.1, 33.3),
    ("LLaMA 2 (7B)",          77.2, 53.0, 45.3, 38.8),
    ("LLaMA 2 (70B)",         85.3, 67.3, 69.8, 44.9),
    ("GPT-4 (est.)",          95.3, 96.3, 86.4, 59.0),
]
for name, hs, arc, mmlu, tqa in benchmarks:
    print(f"{name:<25s} {hs:>9.1f}% {arc:>7.1f}% {mmlu:>7.1f}% {tqa:>11.1f}%")

print(f"\nKey insight: bigger models score higher, but gains are sublinear.")
print(f"TruthfulQA is special — larger models can score WORSE (they memorize")
print(f"common misconceptions more confidently).")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — FEW-SHOT EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
#
# Few-shot evaluation was introduced by GPT-3: instead of fine-tuning the
# model on a task, you just show it a few examples in the prompt.
#
#   0-shot: just the question (no examples)
#   1-shot: one example + the question
#   5-shot: five examples + the question
#
# The model is never fine-tuned — it learns the task pattern from the
# examples in the prompt. This tests "in-context learning" ability.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: FEW-SHOT EVALUATION")
print("=" * 70)


# --- 3.1 Prompt construction for 0-shot and k-shot -----------------------

"""
WHY THIS MATTERS:
Few-shot prompting is how ALL modern LLM benchmarks work. The number of
shots dramatically affects performance — especially for smaller models.
GPT-3's paper showed that 5-shot can double accuracy compared to 0-shot
on some tasks.
"""


def build_few_shot_prompt(
    question: str,
    choices: list[str],
    examples: list[dict] | None = None,
) -> str:
    """
    Build a k-shot prompt for a multiple-choice question.

    Format:
      Question: <example question>
      A. <choice 1>
      B. <choice 2>
      ...
      Answer: <correct letter>

      [repeated for k examples]

      Question: <actual question>
      A. <choice 1>
      ...
      Answer:
    """
    prompt = ""
    if examples:
        for ex in examples:
            prompt += f"Question: {ex['question']}\n"
            for i, c in enumerate(ex["choices"]):
                prompt += f"  {chr(65 + i)}. {c}\n"
            prompt += f"Answer: {chr(65 + ex['correct'])}\n\n"

    prompt += f"Question: {question}\n"
    for i, c in enumerate(choices):
        prompt += f"  {chr(65 + i)}. {c}\n"
    prompt += "Answer:"
    return prompt


# Demo: 0-shot vs 5-shot prompt construction
test_q = "What is the boiling point of water?"
test_choices = ["50°C", "100°C", "150°C", "200°C"]

examples_pool = [
    {"question": "What is the capital of France?",
     "choices": ["London", "Paris", "Berlin", "Madrid"], "correct": 1},
    {"question": "How many legs does a spider have?",
     "choices": ["4", "6", "8", "10"], "correct": 2},
    {"question": "What planet is closest to the Sun?",
     "choices": ["Venus", "Earth", "Mercury", "Mars"], "correct": 2},
    {"question": "What gas do plants absorb?",
     "choices": ["Oxygen", "Nitrogen", "CO2", "Helium"], "correct": 2},
    {"question": "How many days in a leap year?",
     "choices": ["364", "365", "366", "367"], "correct": 2},
]

print(f"\n--- 0-shot prompt ---")
zero_shot = build_few_shot_prompt(test_q, test_choices, examples=None)
print(zero_shot)

print(f"\n--- 5-shot prompt ---")
five_shot = build_few_shot_prompt(test_q, test_choices, examples=examples_pool)
print(five_shot)


# --- 3.2 Simulating few-shot evaluation -----------------------------------

def evaluate_few_shot(
    model_logits_fn,
    test_questions: list[dict],
    train_pool: list[dict],
    k_shots: int,
    tokenize_fn=None,
) -> float:
    """
    Run k-shot evaluation on a set of test questions.

    WHY THIS MATTERS:
    This is the actual evaluation loop that produces benchmark numbers.
    For each test question:
      1. Sample k examples from the training pool
      2. Build the k-shot prompt
      3. Score each answer choice using the model
      4. Check if the model picks the correct one
    """
    if tokenize_fn is None:
        tokenize_fn = lambda text: [hash(w) % vocab_size for w in text.split()]

    correct = 0
    for q in test_questions:
        examples = random.sample(train_pool, min(k_shots, len(train_pool))) if k_shots > 0 else None

        best_score = float("-inf")
        best_idx = -1
        for i, choice_text in enumerate(q["choices"]):
            prompt = build_few_shot_prompt(q["question"], q["choices"], examples)
            prompt_with_answer = prompt + f" {chr(65 + i)}"
            token_ids = tokenize_fn(prompt_with_answer)

            answer_tokens = tokenize_fn(f" {chr(65 + i)}")
            context_tokens = token_ids[: len(token_ids) - len(answer_tokens)]

            score = score_completion(model_logits_fn, context_tokens, answer_tokens)
            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx == q["correct"]:
            correct += 1

    return correct / len(test_questions)


torch.manual_seed(42)
test_qs = [
    {"question": "What is H2O?",
     "choices": ["Salt", "Water", "Oil", "Gold"], "correct": 1},
    {"question": "Largest ocean?",
     "choices": ["Atlantic", "Indian", "Arctic", "Pacific"], "correct": 3},
    {"question": "Speed of light medium?",
     "choices": ["Vacuum", "Water", "Glass", "Air"], "correct": 0},
    {"question": "DNA stands for?",
     "choices": ["Deoxyribonucleic acid", "Data Network Array",
                  "Digital Neural Axis", "Double Neutron Array"], "correct": 0},
]

print(f"\n--- Few-shot accuracy comparison ---")
for k in [0, 1, 3, 5]:
    acc = evaluate_few_shot(mock_model_logits, test_qs, examples_pool, k_shots=k)
    bar_len = int(acc * 30)
    bar = "█" * bar_len + "░" * (30 - bar_len)
    print(f"  {k}-shot: {bar} {acc:.1%}")

print(f"\nNote: With a real model, more shots generally improves accuracy.")
print(f"Our mock model shows the mechanism, not the trend.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — EVAL HARNESS
# ═══════════════════════════════════════════════════════════════════════════
#
# lm-evaluation-harness (by EleutherAI) is the standard tool for running
# benchmarks. It handles prompt formatting, scoring, and aggregation
# for 200+ tasks out of the box.
#
# You don't implement benchmarks yourself — you use the harness.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: EVAL HARNESS (lm-evaluation-harness)")
print("=" * 70)

"""
WHY THIS MATTERS:
When you publish a model or compare to baselines, you MUST use a
standardized evaluation framework. Otherwise results aren't comparable.
lm-evaluation-harness is the de facto standard used by Hugging Face's
Open LLM Leaderboard, EleutherAI, Meta (LLaMA), and most research labs.
"""

print(f"""
--- Installation ---
pip install lm-eval

--- Basic usage: evaluate a HuggingFace model ---
lm_eval --model hf \\
    --model_args pretrained=meta-llama/Llama-2-7b-hf \\
    --tasks hellaswag,arc_easy,arc_challenge,mmlu \\
    --num_fewshot 5 \\
    --batch_size 8 \\
    --device cuda

--- Evaluate your own local model ---
lm_eval --model hf \\
    --model_args pretrained=./my-model-checkpoint/ \\
    --tasks hellaswag \\
    --num_fewshot 10 \\
    --batch_size 4

--- Evaluate a GGUF model (via llama.cpp) ---
lm_eval --model gguf \\
    --model_args base_url=http://localhost:8080 \\
    --tasks mmlu \\
    --num_fewshot 5

--- List all available tasks ---
lm_eval --tasks list

--- Common task groups ---
  hellaswag          : commonsense reasoning (completion)
  arc_easy           : easy science questions
  arc_challenge      : hard science questions
  mmlu               : 57-subject knowledge (5-shot standard)
  truthfulqa_mc2     : resistance to misconceptions
  winogrande         : pronoun resolution (commonsense)
  gsm8k              : grade-school math (chain-of-thought)
""")


# --- 4.1 Simulating what the harness does internally ---------------------

def eval_harness_simulation(model_logits_fn, task_name: str, num_fewshot: int):
    """
    Simulates the eval harness pipeline for one task.
    In reality, the harness handles all of this automatically.
    """
    print(f"\n--- Simulated eval harness: {task_name} ({num_fewshot}-shot) ---")
    print(f"  Step 1: Load task config from registry")
    print(f"  Step 2: Download/cache evaluation dataset")
    print(f"  Step 3: Format {num_fewshot}-shot prompts for each example")
    print(f"  Step 4: Run model inference on all prompts")
    print(f"  Step 5: Score predictions (log-likelihood ranking)")
    print(f"  Step 6: Aggregate results → accuracy")

    # Simulate with random data
    torch.manual_seed(42)
    n_examples = 50
    correct = 0
    for i in range(n_examples):
        context = list(range(i, i + 5))
        choices = [
            [random.randint(0, vocab_size - 1) for _ in range(3)]
            for _ in range(4)
        ]
        scores = [
            score_completion(model_logits_fn, context, c) for c in choices
        ]
        pred = scores.index(max(scores))
        gold = i % 4
        if pred == gold:
            correct += 1

    acc = correct / n_examples
    print(f"\n  Results for {task_name}:")
    print(f"  | {'Metric':<20s} | {'Value':>10s} |")
    print(f"  |{'-'*22}|{'-'*12}|")
    print(f"  | {'acc':<20s} | {acc:>9.4f} |")
    print(f"  | {'acc_norm':<20s} | {acc:>9.4f} |")
    print(f"  | {'num_examples':<20s} | {n_examples:>10d} |")
    return acc


eval_harness_simulation(mock_model_logits, "hellaswag", num_fewshot=10)
eval_harness_simulation(mock_model_logits, "arc_challenge", num_fewshot=25)


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — HUMAN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════
#
# Automated benchmarks measure narrow capabilities. But what users care
# about is: "Does this model give GOOD answers?"
#
# Human evaluation fills this gap:
#   - A/B testing: show two model outputs, ask which is better
#   - Elo ratings: rank models using chess-style Elo from pairwise comparisons
#   - Chatbot Arena: large-scale blind A/B testing (LMSYS)
#
# Human eval is the gold standard — but expensive and slow.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: HUMAN EVALUATION")
print("=" * 70)


# --- 5.1 A/B Testing simulation -----------------------------------------

"""
WHY THIS MATTERS:
When two models score similarly on MMLU (say 65% vs 67%), the one that
users actually PREFER might be the 65% one — because it writes more
clearly, admits uncertainty, or avoids harmful outputs. A/B testing
captures these qualities that benchmarks miss.
"""

print(f"\n--- A/B Testing ---")


def simulate_ab_test(n_comparisons: int, model_a_win_rate: float) -> dict:
    """
    Simulate A/B testing between two models.

    In practice:
    1. Collect prompts from real users
    2. Generate responses from both models
    3. Show both to a human judge (random order, blind)
    4. Judge picks winner (or tie)
    5. Aggregate results
    """
    random.seed(42)
    results = {"A": 0, "B": 0, "tie": 0}
    for _ in range(n_comparisons):
        r = random.random()
        if r < model_a_win_rate:
            results["A"] += 1
        elif r < model_a_win_rate + (1 - model_a_win_rate) * 0.7:
            results["B"] += 1
        else:
            results["tie"] += 1
    return results


ab_results = simulate_ab_test(200, model_a_win_rate=0.45)
total = sum(ab_results.values())
print(f"  200 blind comparisons:")
print(f"  Model A wins: {ab_results['A']:3d} ({ab_results['A']/total:.1%})")
print(f"  Model B wins: {ab_results['B']:3d} ({ab_results['B']/total:.1%})")
print(f"  Ties:         {ab_results['tie']:3d} ({ab_results['tie']/total:.1%})")
winner = "A" if ab_results["A"] > ab_results["B"] else "B"
print(f"  Winner: Model {winner}")


# --- 5.2 Elo Rating system -----------------------------------------------

"""
WHY THIS MATTERS:
Elo ratings (from chess) solve the problem of ranking MANY models from
pairwise comparisons. LMSYS Chatbot Arena uses this to rank 60+ models.
A model with Elo 1200 is expected to beat a model with Elo 1000 about
76% of the time.
"""

print(f"\n--- Elo Rating System ---")


def update_elo(rating_a: float, rating_b: float, winner: str,
               k: float = 32.0) -> tuple[float, float]:
    """
    Update Elo ratings after a match.

    Expected score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    Update: R_A' = R_A + K * (S_A - E_A)
      where S_A = 1 (win), 0.5 (tie), 0 (loss)
    """
    e_a = 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))
    e_b = 1.0 - e_a

    if winner == "A":
        s_a, s_b = 1.0, 0.0
    elif winner == "B":
        s_a, s_b = 0.0, 1.0
    else:
        s_a, s_b = 0.5, 0.5

    new_a = rating_a + k * (s_a - e_a)
    new_b = rating_b + k * (s_b - e_b)
    return new_a, new_b


# Simulate a tournament between 4 models
models = {
    "GPT-4":      1200.0,
    "Claude-3":   1200.0,
    "LLaMA-70B":  1200.0,
    "Mistral-7B": 1200.0,
}

# True win probabilities (GPT-4 > Claude-3 > LLaMA-70B > Mistral-7B)
true_strength = {"GPT-4": 0.85, "Claude-3": 0.75, "LLaMA-70B": 0.55, "Mistral-7B": 0.35}

random.seed(42)
model_names = list(models.keys())
n_rounds = 500
for _ in range(n_rounds):
    a, b = random.sample(model_names, 2)
    prob_a = true_strength[a] / (true_strength[a] + true_strength[b])
    winner = "A" if random.random() < prob_a else "B"
    models[a], models[b] = update_elo(models[a], models[b], winner)

print(f"  After {n_rounds} pairwise matches:")
print(f"  {'Model':<15s} {'Elo':>8s}")
print(f"  {'-'*15} {'-'*8}")
for name, elo in sorted(models.items(), key=lambda x: -x[1]):
    print(f"  {name:<15s} {elo:>8.0f}")


# --- 5.3 Chatbot Arena concept -------------------------------------------

print(f"\n--- Chatbot Arena (LMSYS) ---")
print(f"  How it works:")
print(f"  1. User submits a prompt")
print(f"  2. Two anonymous models generate responses")
print(f"  3. User votes for the better response (or tie)")
print(f"  4. Elo ratings updated after each vote")
print(f"  5. After 100K+ votes, rankings stabilize")
print(f"")
print(f"  Why it matters:")
print(f"  - Most trusted LLM ranking (200K+ human votes)")
print(f"  - Captures 'vibes' that benchmarks miss")
print(f"  - Models can't game it (blind, adversarial users)")
print(f"  - Top models: GPT-4o ≈ 1287, Claude 3.5 ≈ 1271, Gemini 1.5 ≈ 1260")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — CONTAMINATION CHECKS
# ═══════════════════════════════════════════════════════════════════════════
#
# A model that memorized the test set will score perfectly — but that
# number is meaningless. This is "benchmark contamination" or "data leakage."
#
# It happens when benchmark questions appear in the pre-training data:
#   - HellaSwag questions crawled from the web
#   - MMLU questions in Wikipedia articles
#   - Exact copies or paraphrases of test items
#
# Contamination checking is essential for trustworthy evaluation.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 6: CONTAMINATION CHECKS")
print("=" * 70)


# --- 6.1 N-gram overlap detection ----------------------------------------

"""
WHY THIS MATTERS:
The simplest contamination check: compute n-gram overlap between
training data and benchmark examples. If a benchmark question shares
a suspiciously long n-gram with the training data, it's contaminated.

GPT-3's paper used 13-gram overlap for contamination detection.
LLaMA used 10-gram overlap.
"""

print(f"\n--- N-gram Overlap Detection ---")


def get_ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    """Extract all n-grams from a token sequence."""
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def ngram_overlap_ratio(benchmark_text: str, training_text: str,
                        n: int = 13) -> float:
    """
    Compute the fraction of benchmark n-grams found in training data.

    Returns a ratio between 0.0 (no overlap) and 1.0 (fully contained).
    Ratio > 0.5 with n ≥ 10 is a strong contamination signal.
    """
    bench_tokens = benchmark_text.lower().split()
    train_tokens = training_text.lower().split()

    if len(bench_tokens) < n:
        return 0.0

    bench_ngrams = get_ngrams(bench_tokens, n)
    train_ngrams = get_ngrams(train_tokens, n)

    if not bench_ngrams:
        return 0.0

    overlap = bench_ngrams & train_ngrams
    return len(overlap) / len(bench_ngrams)


# Demo: contamination detection
training_corpus = (
    "the quick brown fox jumps over the lazy dog "
    "the cat sat on the mat near the window "
    "water boils at one hundred degrees celsius under normal pressure "
    "the capital of france is paris which is known for the eiffel tower "
    "machine learning models learn patterns from data "
    "the mitochondria is the powerhouse of the cell providing energy"
)

benchmark_examples = {
    "Clean (novel question)":
        "what is the airspeed velocity of an unladen swallow in kilometers",
    "Contaminated (exact match)":
        "water boils at one hundred degrees celsius under normal pressure",
    "Partial overlap":
        "the capital of france is paris a beautiful city in europe",
    "Paraphrased (harder to detect)":
        "paris serves as the capital city of the french republic",
}

print(f"  N-gram size: 8 (shorter for demo; real systems use 10-13)")
print(f"  Training corpus: {len(training_corpus.split())} tokens\n")

for label, bench_text in benchmark_examples.items():
    ratio = ngram_overlap_ratio(bench_text, training_corpus, n=8)
    flag = " ⚠ CONTAMINATED" if ratio > 0.3 else "   clean"
    print(f"  {label:35s}  overlap = {ratio:.2%}{flag}")


# --- 6.2 Full decontamination pipeline -----------------------------------

print(f"\n--- Decontamination Pipeline ---")


def decontaminate_dataset(
    training_docs: list[str],
    benchmark_texts: list[str],
    n: int = 10,
    threshold: float = 0.5,
) -> tuple[list[str], int]:
    """
    Remove contaminated documents from training data.

    WHY THIS MATTERS:
    Before publishing model benchmarks, responsible labs check for
    contamination and report both:
      - Raw scores (may include contaminated examples)
      - Clean scores (contaminated examples removed from eval set)

    GPT-4's technical report dedicates an entire section to this.
    """
    clean_docs = []
    removed_count = 0

    # Build a set of all benchmark n-grams
    all_bench_ngrams = set()
    for text in benchmark_texts:
        tokens = text.lower().split()
        all_bench_ngrams |= get_ngrams(tokens, n)

    for doc in training_docs:
        doc_tokens = doc.lower().split()
        doc_ngrams = get_ngrams(doc_tokens, n)

        if not doc_ngrams:
            clean_docs.append(doc)
            continue

        overlap_ratio = len(doc_ngrams & all_bench_ngrams) / len(doc_ngrams)
        if overlap_ratio >= threshold:
            removed_count += 1
        else:
            clean_docs.append(doc)

    return clean_docs, removed_count


training_docs = [
    "the quick brown fox jumps over the lazy dog in the garden on a sunny day in summer",
    "water boils at one hundred degrees celsius under normal atmospheric pressure conditions at sea level",
    "deep learning has revolutionized natural language processing and computer vision fields in recent years",
    "the capital of france is paris which is known for the eiffel tower and the louvre museum",
    "transformers use self attention mechanisms to process sequences in parallel for efficiency",
]

benchmark_texts = [
    "water boils at one hundred degrees celsius under normal atmospheric pressure conditions at sea level",
    "the capital of france is paris which is known for the eiffel tower and the louvre museum",
]

clean, removed = decontaminate_dataset(training_docs, benchmark_texts, n=8, threshold=0.4)
print(f"  Original training docs: {len(training_docs)}")
print(f"  Benchmark examples:     {len(benchmark_texts)}")
print(f"  Removed (contaminated): {removed}")
print(f"  Clean training docs:    {len(clean)}")
print(f"\n  Remaining clean documents:")
for i, doc in enumerate(clean):
    print(f"    {i+1}. {doc[:70]}...")


# --- 6.3 Contamination report --------------------------------------------

print(f"\n--- Contamination Report (what labs publish) ---")
print(f"""
  Example from GPT-4 Technical Report:
  ┌──────────────────┬────────────┬─────────────┐
  │ Benchmark        │ Raw Score  │ Clean Score  │
  ├──────────────────┼────────────┼─────────────┤
  │ HellaSwag        │  95.3%     │  95.0%       │
  │ ARC-Challenge    │  96.3%     │  96.0%       │
  │ MMLU             │  86.4%     │  85.8%       │
  │ DROP (F1)        │  80.9%     │  79.7%       │
  └──────────────────┴────────────┴─────────────┘

  Small drops → minimal contamination impact.
  Large drops → the model was partly memorizing benchmarks.

  Best practices:
  1. Check n-gram overlap (n=10-13) between training data and ALL benchmarks
  2. Report both raw and decontaminated scores
  3. Use held-out benchmarks not available at training time
  4. Be skeptical of scores that are "too good" for model size
""")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXERCISES")
print("=" * 70)
print("""
Exercise 1: Perplexity Deep Dive
---------------------------------
Create a mock language model (just random logits) with vocab_size=50000.
  a) Generate 10 sequences of length 50 with random logits.
  b) Compute corpus perplexity — it should be close to 50000 (random baseline).
  c) Now boost the correct token's logit by +5.0 in each position.
     What's the new perplexity?
  d) Try boost values of [1, 2, 5, 10, 20]. Plot how PPL decreases.
     At what boost does PPL drop below 10?

Exercise 2: Build a Benchmark
------------------------------
Create a 20-question multiple-choice benchmark about Python programming:
  a) Write 20 questions with 4 choices each, store as list of dicts.
  b) Implement a scoring function that uses log-likelihood ranking.
  c) Test with a random model — accuracy should be ~25%.
  d) Create a "cheating" model that always boosts the correct answer.
     Verify it gets ~100%.

Exercise 3: Elo Rating Convergence
------------------------------------
  a) Start 6 models at Elo 1200 with true strengths [0.95, 0.8, 0.65, 0.5, 0.35, 0.2].
  b) Run 2000 random pairwise matches, updating Elo after each.
  c) Print Elo ratings every 200 matches.
  d) How many matches until rankings stabilize? Try K=16 vs K=32.

Exercise 4: N-gram Contamination Detector
-------------------------------------------
  a) Generate a "training corpus" of 100 random sentences (5-15 words each,
     sampled from a vocabulary of 200 words).
  b) Plant 5 benchmark sentences verbatim into the corpus.
  c) Run contamination detection with n=8. Does it find all 5?
  d) Slightly modify the planted sentences (swap 1-2 words).
     Does detection still work? At what n does it start missing them?

Exercise 5: Full Evaluation Pipeline
--------------------------------------
Build a complete evaluation pipeline that:
  a) Takes a "model" (any function: token_ids → logits)
  b) Computes perplexity on a test set
  c) Runs a 10-question multiple-choice benchmark (0-shot and 5-shot)
  d) Prints a clean report card with all metrics
  e) Checks for contamination between the benchmark and a sample "training set"
""")
