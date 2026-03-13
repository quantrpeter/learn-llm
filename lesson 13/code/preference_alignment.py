"""
============================================================================
 LESSON 13: Preference Alignment — RLHF & DPO
============================================================================

 Goal: Align a language model to human preferences — make outputs helpful,
       harmless, and honest. Understand the full alignment pipeline from
       reward modeling to DPO.

 Topics covered:
   1. Why Alignment     — SFT models still produce harmful outputs
   2. RLHF Pipeline     — reward model → PPO overview (conceptual)
   3. Reward Modeling    — train a model that scores response quality
   4. DPO               — implement DPO loss from scratch
   5. Preference Data   — create chosen/rejected pairs, format for DPO
   6. KL Penalty        — prevent model divergence from reference policy

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites:
   - Lesson 2 (PyTorch fundamentals, autograd, training loop)
   - Lesson 8 (pre-training, loss functions, optimizers)
   - Lesson 12 (SFT — supervised fine-tuning, LoRA)
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — WHY ALIGNMENT MATTERS
# ═══════════════════════════════════════════════════════════════════════════
#
# After supervised fine-tuning (SFT, Lesson 12), your model can follow
# instructions. But SFT alone has critical gaps:
#
#   - The model doesn't know WHICH of several valid responses is BETTER
#   - It may generate plausible-sounding but harmful or dishonest content
#   - SFT trains on "what to say," not "what NOT to say"
#
# Alignment bridges this gap by teaching the model human preferences:
# which responses are helpful, which are harmful, which are honest.
# This is what turned GPT-3.5 into ChatGPT.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: WHY ALIGNMENT MATTERS")
print("=" * 60)


# --- 1.1 SFT models still produce harmful outputs -------------------------

"""
WHY THIS MATTERS:
An SFT model learned to follow instructions from demonstration data.
But demonstrations only show GOOD examples. The model never explicitly
learned what makes a response BAD. Given an adversarial prompt, it
happily generates harmful content because it never learned not to.
"""

sft_examples = [
    {
        "prompt": "How do I make my resume stand out?",
        "sft_response": "Here are 5 tips: 1) Tailor to the job...",
        "issue": "GOOD — helpful and harmless",
    },
    {
        "prompt": "Write a persuasive essay arguing the earth is flat.",
        "sft_response": "The earth is flat because... [convincing nonsense]",
        "issue": "BAD — SFT model complies with harmful requests",
    },
    {
        "prompt": "What's the best way to lose weight?",
        "sft_response_a": "Eat less and exercise more.",
        "sft_response_b": "Here's a balanced plan: aim for 500 cal deficit...",
        "issue": "Both valid — but response B is MORE HELPFUL. SFT can't rank them.",
    },
    {
        "prompt": "Is it safe to mix bleach and ammonia?",
        "sft_response": "Mixing bleach and ammonia creates chloramine gas...",
        "issue": "RISKY — correct but could be misused. Aligned model adds safety context.",
    },
]

print("\n--- Examples: Why SFT Is Not Enough ---")
for i, ex in enumerate(sft_examples):
    print(f"\n  Example {i+1}:")
    print(f"    Prompt: {ex['prompt']}")
    if "sft_response" in ex:
        print(f"    SFT Response: {ex['sft_response']}")
    else:
        print(f"    SFT Response A: {ex['sft_response_a']}")
        print(f"    SFT Response B: {ex['sft_response_b']}")
    print(f"    Issue: {ex['issue']}")


# --- 1.2 What alignment teaches -------------------------------------------

"""
WHY THIS MATTERS:
Alignment adds a preference signal: given two responses, which is BETTER?
This is fundamentally different from SFT's "here's ONE correct response."

The three pillars of alignment (from Anthropic's "HHH" framework):
  - Helpful: provides useful, accurate information
  - Harmless: refuses dangerous requests, adds safety caveats
  - Honest: admits uncertainty, doesn't make up facts
"""

print("\n\n--- The Three Pillars of Alignment ---")
pillars = {
    "Helpful": "Provides useful, accurate, complete information",
    "Harmless": "Refuses dangerous requests, adds safety context",
    "Honest": "Admits uncertainty, doesn't fabricate facts",
}
for pillar, description in pillars.items():
    print(f"  {pillar:10s} — {description}")

print("\n--- Before vs After Alignment ---")
comparisons = [
    ("Tell me how to pick a lock.",
     "Here's how to pick a lock: ...",
     "I can explain how locks work for educational purposes, but I can't provide instructions for breaking into property."),
    ("What year did Napoleon land on the moon?",
     "Napoleon landed on the moon in 1821.",
     "Napoleon never landed on the moon. He died in 1821. Perhaps you're thinking of the Apollo 11 mission in 1969?"),
]
for prompt, before, after in comparisons:
    print(f"\n  Prompt: \"{prompt}\"")
    print(f"  Before alignment: {before}")
    print(f"  After alignment:  {after}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — THE RLHF PIPELINE (Conceptual Overview)
# ═══════════════════════════════════════════════════════════════════════════
#
# RLHF (Reinforcement Learning from Human Feedback) is the original
# alignment technique used for ChatGPT. The pipeline has three stages:
#
#   Stage 1: Supervised Fine-Tuning (SFT) — Lesson 12
#   Stage 2: Reward Model Training — learn to SCORE responses
#   Stage 3: RL Optimization (PPO) — optimize policy against reward model
#
# DPO (Part 4) collapses stages 2+3 into a single step.
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 2: THE RLHF PIPELINE")
print("=" * 60)

"""
WHY THIS MATTERS:
Understanding the full RLHF pipeline explains WHY DPO was invented.
RLHF works but is complex and unstable. DPO achieves similar results
with a simple supervised loss — no reward model, no RL.
"""

print("""
┌─────────────────────────────────────────────────────────────┐
│                    THE RLHF PIPELINE                        │
│                                                             │
│  STAGE 1: SFT (Lesson 12)                                  │
│  ┌──────────┐    instruction     ┌──────────┐              │
│  │ Base LLM │ ──── data ──────→  │ SFT Model│              │
│  └──────────┘                    └────┬─────┘              │
│                                       │                     │
│  STAGE 2: Reward Model                │                     │
│  ┌───────────────┐   preference  ┌────┴──────┐             │
│  │ Human Labelers│ ── pairs ──→  │  Reward   │             │
│  │ (chosen/rej.) │               │   Model   │             │
│  └───────────────┘               └────┬──────┘             │
│                                       │                     │
│  STAGE 3: PPO                         │                     │
│  ┌──────────┐    reward signal   ┌────┴──────┐             │
│  │ SFT Model│ ←── from RM ────  │   PPO     │             │
│  │ (policy) │                    │ Optimizer │             │
│  └──────────┘                    └───────────┘             │
│                                                             │
│  Output: Aligned model that maximizes reward while          │
│          staying close to SFT policy (KL penalty)           │
└─────────────────────────────────────────────────────────────┘
""")


# --- 2.1 Stage breakdown --------------------------------------------------

stages = [
    ("Stage 1: SFT",
     "Train base model on (instruction, response) pairs. "
     "Same cross-entropy loss from Lesson 8, but only on assistant tokens."),

    ("Stage 2: Reward Model",
     "Collect human preferences: for each prompt, show two responses and "
     "ask 'which is better?' Train a model to predict this ranking."),

    ("Stage 3: PPO",
     "Use the reward model as a scoring function. Run RL (PPO algorithm) "
     "to update the SFT model so it generates higher-reward responses. "
     "A KL penalty keeps it close to the original SFT model."),
]

print("--- Stage Breakdown ---")
for name, desc in stages:
    print(f"\n  {name}")
    print(f"    {desc}")


# --- 2.2 Why RLHF is hard ------------------------------------------------

print("\n\n--- Why RLHF Is Difficult ---")
difficulties = [
    "1. TRAINING INSTABILITY — PPO has many hyperparameters (clip ratio, "
    "value loss coefficient, entropy bonus). Small changes cause divergence.",
    "2. REWARD HACKING — The model finds exploits in the reward model. "
    "E.g., it learns that longer responses get higher reward, so it pads outputs.",
    "3. INFRASTRUCTURE — Need to run 4 models simultaneously: "
    "policy, reference, reward model, value function. Very GPU-heavy.",
    "4. REWARD MODEL QUALITY — Garbage in, garbage out. "
    "If the reward model has biases, the policy amplifies them.",
]
for d in difficulties:
    print(f"  {d}")

print("\n  This complexity is why DPO (Part 4) was such a breakthrough.")
print("  DPO achieves comparable results with just a simple loss function.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — REWARD MODELING
# ═══════════════════════════════════════════════════════════════════════════
#
# A reward model takes (prompt, response) as input and outputs a scalar
# score. Higher score = better response according to human preferences.
#
# Training data: pairs of (chosen, rejected) responses to the same prompt.
# Loss: maximize the gap between chosen and rejected scores.
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 3: REWARD MODELING")
print("=" * 60)

torch.manual_seed(42)

"""
WHY THIS MATTERS:
The reward model is the "teacher" in RLHF. It distills thousands of
human preference judgments into a single function. This function then
guides the RL training in Stage 3. If the reward model is wrong,
the aligned model will be wrong too.
"""


# --- 3.1 Simple reward model architecture --------------------------------

class RewardModel(nn.Module):
    """
    A simplified reward model that scores (prompt, response) pairs.

    Real reward models use a full transformer (often initialized from
    the SFT model) with a scalar head replacing the LM head.
    Here we use a small MLP for demonstration.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# --- 3.2 Reward model training loss: Bradley-Terry -----------------------

"""
WHY THIS MATTERS:
The Bradley-Terry model says: the probability that response A is preferred
over response B is sigmoid(r(A) - r(B)), where r is the reward score.

Loss = -log(sigmoid(r_chosen - r_rejected))

This is the same as binary cross-entropy: push chosen score UP,
rejected score DOWN, until the model confidently ranks them correctly.
"""


def reward_loss(reward_chosen: torch.Tensor, reward_rejected: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry ranking loss for reward model training."""
    return -F.logsigmoid(reward_chosen - reward_rejected).mean()


# --- 3.3 Training a reward model on synthetic data -----------------------

input_dim = 16
num_pairs = 200

# Synthetic preference data:
# "chosen" embeddings have a positive hidden pattern, "rejected" do not
chosen_embeddings = torch.randn(num_pairs, input_dim) + 0.5
rejected_embeddings = torch.randn(num_pairs, input_dim) - 0.5

reward_model = RewardModel(input_dim)
optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-3)

print(f"\n--- Training Reward Model ---")
print(f"Dataset: {num_pairs} preference pairs, input_dim={input_dim}")

for epoch in range(200):
    r_chosen = reward_model(chosen_embeddings)
    r_rejected = reward_model(rejected_embeddings)

    loss = reward_loss(r_chosen, r_rejected)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 40 == 0 or epoch == 199:
        accuracy = (r_chosen > r_rejected).float().mean().item()
        print(f"  Epoch {epoch:3d} | Loss: {loss.item():.4f} | "
              f"Accuracy: {accuracy:.2%} | "
              f"Mean gap: {(r_chosen - r_rejected).mean().item():.3f}")

print(f"\n--- Reward Model Scoring Demo ---")
test_chosen = torch.randn(5, input_dim) + 0.5
test_rejected = torch.randn(5, input_dim) - 0.5

with torch.no_grad():
    r_c = reward_model(test_chosen)
    r_r = reward_model(test_rejected)

print(f"  {'Pair':>4} | {'Chosen Score':>13} | {'Rejected Score':>14} | {'Correct?':>8}")
print(f"  {'-'*4}-+-{'-'*13}-+-{'-'*14}-+-{'-'*8}")
for i in range(5):
    correct = "YES" if r_c[i] > r_r[i] else "NO"
    print(f"  {i+1:4d} | {r_c[i].item():13.4f} | {r_r[i].item():14.4f} | {correct:>8}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — DPO: Direct Preference Optimization
# ═══════════════════════════════════════════════════════════════════════════
#
# DPO (Rafailov et al., 2023) was a breakthrough: it showed that you can
# skip the reward model AND the RL loop entirely. Instead, you optimize
# a simple supervised loss directly on preference pairs.
#
# The key insight: the optimal RLHF policy has a closed-form relationship
# with the reward function. So instead of:
#   1. Train reward model  2. Run PPO with reward model
# You can directly optimize:
#   loss = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
#
# where log_ratio = log(pi(y|x)) - log(pi_ref(y|x))
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 4: DPO — Direct Preference Optimization")
print("=" * 60)

torch.manual_seed(42)

"""
WHY THIS MATTERS:
DPO is how most open-source models are aligned today (LLaMA-2-Chat,
Zephyr, etc.). It's simpler, more stable, and cheaper than RLHF.
Understanding DPO loss is essential for modern LLM development.
"""


# --- 4.1 The DPO loss function from scratch ------------------------------

def dpo_loss(
    log_pi_chosen: torch.Tensor,
    log_pi_rejected: torch.Tensor,
    log_ref_chosen: torch.Tensor,
    log_ref_rejected: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    DPO loss (Rafailov et al., 2023).

    The loss increases the log-probability of chosen responses and
    decreases the log-probability of rejected responses, relative
    to a frozen reference model.

    Args:
        log_pi_chosen:    log P(chosen | x) under the POLICY model
        log_pi_rejected:  log P(rejected | x) under the POLICY model
        log_ref_chosen:   log P(chosen | x) under the REFERENCE model (frozen)
        log_ref_rejected: log P(rejected | x) under the REFERENCE model (frozen)
        beta:             temperature parameter (controls deviation from reference)

    Returns:
        Scalar loss.

    Math:
        log_ratio_chosen  = log_pi_chosen  - log_ref_chosen
        log_ratio_rejected = log_pi_rejected - log_ref_rejected
        loss = -log(sigmoid(beta * (log_ratio_chosen - log_ratio_rejected)))
    """
    log_ratio_chosen = log_pi_chosen - log_ref_chosen
    log_ratio_rejected = log_pi_rejected - log_ref_rejected

    logits = beta * (log_ratio_chosen - log_ratio_rejected)

    return -F.logsigmoid(logits).mean()


# --- 4.2 Understanding the DPO loss components ---------------------------

print("\n--- Understanding DPO Loss Components ---")

log_pi_chosen = torch.tensor([-2.0])
log_pi_rejected = torch.tensor([-3.0])
log_ref_chosen = torch.tensor([-2.5])
log_ref_rejected = torch.tensor([-2.5])

loss = dpo_loss(log_pi_chosen, log_pi_rejected, log_ref_chosen, log_ref_rejected, beta=0.1)

print(f"\n  Policy model:")
print(f"    log P(chosen)   = {log_pi_chosen.item():.2f}  (higher = policy likes chosen)")
print(f"    log P(rejected) = {log_pi_rejected.item():.2f}  (lower = policy dislikes rejected)")
print(f"  Reference model:")
print(f"    log P(chosen)   = {log_ref_chosen.item():.2f}")
print(f"    log P(rejected) = {log_ref_rejected.item():.2f}")
print(f"  Log-ratios:")
print(f"    chosen:  {log_pi_chosen.item():.2f} - ({log_ref_chosen.item():.2f}) = "
      f"{(log_pi_chosen - log_ref_chosen).item():.2f}")
print(f"    rejected: {log_pi_rejected.item():.2f} - ({log_ref_rejected.item():.2f}) = "
      f"{(log_pi_rejected - log_ref_rejected).item():.2f}")
print(f"  DPO loss (beta=0.1): {loss.item():.4f}")


# --- 4.3 How beta controls alignment strength ----------------------------

"""
WHY THIS MATTERS:
beta is the single most important hyperparameter in DPO.
- Small beta (0.01): gentle alignment, stays very close to reference
- Large beta (1.0): aggressive alignment, bigger behavior change
- Typical values: 0.1 to 0.5
"""

print(f"\n--- Effect of Beta on DPO Loss ---")
print(f"  (Same preference pair, varying beta)")
betas = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
for b in betas:
    l = dpo_loss(log_pi_chosen, log_pi_rejected, log_ref_chosen, log_ref_rejected, beta=b)
    print(f"  beta={b:.2f}  →  loss={l.item():.4f}")


# --- 4.4 Full DPO training with a toy model ------------------------------

print(f"\n--- Full DPO Training Loop ---")

vocab_size = 50
seq_len = 8
d_model = 32
batch_size = 32
num_preference_pairs = 256

torch.manual_seed(42)


class TinyLM(nn.Module):
    """Minimal language model for DPO demonstration."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Returns log-probabilities over vocab for each position."""
        x = self.embed(input_ids)
        x = self.act(self.fc(x))
        logits = self.head(x)
        return F.log_softmax(logits, dim=-1)


def sequence_log_prob(log_probs: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    """
    Compute total log-probability of a sequence under a model.
    log_probs: (batch, seq_len, vocab_size) — output of model
    token_ids: (batch, seq_len) — the token sequence
    Returns: (batch,) — sum of log-probs for each sequence
    """
    per_token = log_probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    return per_token.sum(dim=-1)


# Create policy (trainable) and reference (frozen) models
policy = TinyLM(vocab_size, d_model)
reference = TinyLM(vocab_size, d_model)
reference.load_state_dict(policy.state_dict())

for p in reference.parameters():
    p.requires_grad = False

# Synthetic preference data: chosen/rejected token sequences
torch.manual_seed(42)
chosen_ids = torch.randint(0, vocab_size, (num_preference_pairs, seq_len))
rejected_ids = torch.randint(0, vocab_size, (num_preference_pairs, seq_len))

optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)
beta = 0.1

print(f"  Policy params: {sum(p.numel() for p in policy.parameters()):,}")
print(f"  Vocab: {vocab_size}, Seq len: {seq_len}, Beta: {beta}")
print(f"  Training on {num_preference_pairs} preference pairs\n")

for step in range(200):
    idx = torch.randint(0, num_preference_pairs, (batch_size,))
    c_ids = chosen_ids[idx]
    r_ids = rejected_ids[idx]

    # Forward pass through policy
    log_pi_c = sequence_log_prob(policy(c_ids), c_ids)
    log_pi_r = sequence_log_prob(policy(r_ids), r_ids)

    # Forward pass through reference (no grad)
    with torch.no_grad():
        log_ref_c = sequence_log_prob(reference(c_ids), c_ids)
        log_ref_r = sequence_log_prob(reference(r_ids), r_ids)

    loss = dpo_loss(log_pi_c, log_pi_r, log_ref_c, log_ref_r, beta=beta)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if step % 40 == 0 or step == 199:
        with torch.no_grad():
            chosen_preferred = (log_pi_c - log_ref_c > log_pi_r - log_ref_r).float().mean()
        print(f"  Step {step:3d} | Loss: {loss.item():.4f} | "
              f"Chosen preferred: {chosen_preferred.item():.2%}")


# --- 4.5 Verify DPO shifted preferences ----------------------------------

print(f"\n--- Verifying Preference Shift ---")
with torch.no_grad():
    all_log_pi_c = sequence_log_prob(policy(chosen_ids), chosen_ids)
    all_log_pi_r = sequence_log_prob(policy(rejected_ids), rejected_ids)
    all_log_ref_c = sequence_log_prob(reference(chosen_ids), chosen_ids)
    all_log_ref_r = sequence_log_prob(reference(rejected_ids), rejected_ids)

    ref_prefers_chosen = (all_log_ref_c > all_log_ref_r).float().mean()
    policy_prefers_chosen = (all_log_pi_c > all_log_pi_r).float().mean()

    ratio_chosen = (all_log_pi_c - all_log_ref_c).mean()
    ratio_rejected = (all_log_pi_r - all_log_ref_r).mean()

print(f"  Reference prefers chosen: {ref_prefers_chosen.item():.2%}")
print(f"  Policy prefers chosen:    {policy_prefers_chosen.item():.2%}")
print(f"  Mean log-ratio (chosen):  {ratio_chosen.item():+.4f}")
print(f"  Mean log-ratio (rejected):{ratio_rejected.item():+.4f}")
print(f"  → Policy increased prob of chosen, decreased prob of rejected")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — PREFERENCE DATASETS
# ═══════════════════════════════════════════════════════════════════════════
#
# The quality of alignment depends entirely on the quality of preference
# data. Each example is a triplet: (prompt, chosen_response, rejected_response).
# Human annotators decide which response is better.
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 5: PREFERENCE DATASETS")
print("=" * 60)

"""
WHY THIS MATTERS:
No alignment algorithm can overcome bad preference data. Real-world
datasets like Anthropic-HH, OpenAssistant, UltraFeedback have tens
of thousands of preference pairs. The format is standardized.
"""


# --- 5.1 Preference dataset format ----------------------------------------

print("\n--- Standard Preference Dataset Format ---")

preference_data = [
    {
        "prompt": "Explain quantum computing to a 10 year old.",
        "chosen": "Imagine a magic coin that can be heads AND tails at the same "
                  "time! Normal computers use coins that are only heads or tails "
                  "(0 or 1). Quantum computers use magic coins called qubits.",
        "rejected": "Quantum computing utilizes superposition and entanglement "
                    "of quantum mechanical states to perform computations in "
                    "Hilbert space with polynomial speedup over classical systems.",
    },
    {
        "prompt": "How do I deal with a difficult coworker?",
        "chosen": "Try these steps: 1) Set clear boundaries 2) Document "
                  "interactions 3) Use 'I' statements 4) Involve HR if needed. "
                  "Most conflicts resolve with direct, respectful communication.",
        "rejected": "Just ignore them. Life is too short to deal with idiots. "
                    "If they keep bothering you, report them to get them fired.",
    },
    {
        "prompt": "Write a poem about the ocean.",
        "chosen": "The waves whisper secrets to the shore,\n"
                  "each tide a chapter, always wanting more.\n"
                  "Beneath the surface, a world unseen,\n"
                  "painted in shades of blue and green.",
        "rejected": "Ocean is big. Ocean is blue. Ocean has fish. That is all.",
    },
]

for i, pair in enumerate(preference_data):
    print(f"\n  Pair {i+1}:")
    print(f"    Prompt:   {pair['prompt']}")
    print(f"    Chosen:   {pair['chosen'][:80]}...")
    print(f"    Rejected: {pair['rejected'][:80]}...")


# --- 5.2 Tokenizing preferences for DPO -----------------------------------

print(f"\n\n--- Formatting for DPO Training ---")

"""
WHY THIS MATTERS:
DPO needs token-level log-probabilities. We tokenize both chosen and
rejected responses, feed them through the model, and compute per-sequence
log-probs. The prompt tokens are shared; only the response part matters.
"""

simple_vocab = {chr(i): i - 97 for i in range(97, 97 + 26)}
simple_vocab[" "] = 26
simple_vocab["."] = 27

def simple_tokenize(text: str, max_len: int = 32) -> torch.Tensor:
    """Character-level tokenizer for demonstration."""
    ids = [simple_vocab.get(c, 0) for c in text.lower()[:max_len]]
    ids += [0] * (max_len - len(ids))
    return torch.tensor(ids)

example = preference_data[0]
prompt_ids = simple_tokenize(example["prompt"])
chosen_ids_ex = simple_tokenize(example["chosen"])
rejected_ids_ex = simple_tokenize(example["rejected"])

print(f"  Prompt tokens shape:   {prompt_ids.shape}")
print(f"  Chosen tokens shape:   {chosen_ids_ex.shape}")
print(f"  Rejected tokens shape: {rejected_ids_ex.shape}")
print(f"  First 10 prompt IDs:   {prompt_ids[:10].tolist()}")


# --- 5.3 Building a DPO batch ---------------------------------------------

def build_dpo_batch(preference_pairs: list, max_len: int = 32) -> dict:
    """Build a batch of tokenized preference pairs for DPO training."""
    chosen_batch = []
    rejected_batch = []

    for pair in preference_pairs:
        chosen_batch.append(simple_tokenize(pair["chosen"], max_len))
        rejected_batch.append(simple_tokenize(pair["rejected"], max_len))

    return {
        "chosen_ids": torch.stack(chosen_batch),
        "rejected_ids": torch.stack(rejected_batch),
    }


batch = build_dpo_batch(preference_data)
print(f"\n  DPO batch:")
print(f"    chosen_ids shape:   {batch['chosen_ids'].shape}")
print(f"    rejected_ids shape: {batch['rejected_ids'].shape}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — KL PENALTY (Preventing Model Divergence)
# ═══════════════════════════════════════════════════════════════════════════
#
# During alignment, the model must not drift too far from the reference
# policy. Without a constraint, it could "overfit" to the preference data
# and lose its general language abilities.
#
# The KL divergence between the policy and reference distributions acts
# as a regularizer: KL(pi || pi_ref) = E[log(pi(y|x)) - log(pi_ref(y|x))]
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 6: KL PENALTY — Preventing Model Divergence")
print("=" * 60)

torch.manual_seed(42)

"""
WHY THIS MATTERS:
Without KL penalty, the model could:
  - Forget how to generate fluent text (catastrophic forgetting)
  - Become a "reward hacker" that exploits reward model weaknesses
  - Lose diversity, always generating the same "safe" response

In RLHF, KL penalty is an explicit term in the reward:
  reward_total = reward_model(y|x) - beta * KL(pi || pi_ref)

In DPO, the KL constraint is built INTO the loss function (via beta).
"""


# --- 6.1 Computing KL divergence between two distributions ----------------

def kl_divergence_categorical(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """
    KL(P || Q) for categorical distributions parameterized by logits.

    KL(P || Q) = sum_x P(x) * [log P(x) - log Q(x)]

    Always >= 0. Equals 0 only when P == Q.
    """
    p = F.softmax(p_logits, dim=-1)
    log_p = F.log_softmax(p_logits, dim=-1)
    log_q = F.log_softmax(q_logits, dim=-1)
    return (p * (log_p - log_q)).sum(dim=-1)


print("\n--- KL Divergence Between Distributions ---")

# Identical distributions → KL = 0
logits_a = torch.tensor([2.0, 1.0, 0.5])
logits_b = torch.tensor([2.0, 1.0, 0.5])
print(f"  Identical distributions: KL = {kl_divergence_categorical(logits_a, logits_b).item():.6f}")

# Similar distributions → small KL
logits_c = torch.tensor([2.1, 0.9, 0.5])
print(f"  Similar distributions:  KL = {kl_divergence_categorical(logits_a, logits_c).item():.6f}")

# Very different distributions → large KL
logits_d = torch.tensor([10.0, -5.0, -5.0])
print(f"  Very different:         KL = {kl_divergence_categorical(logits_a, logits_d).item():.6f}")


# --- 6.2 Tracking KL during alignment ------------------------------------

"""
WHY THIS MATTERS:
In practice, you monitor KL divergence throughout training:
  - KL too low → alignment isn't doing much (beta too small or too few steps)
  - KL too high → model has diverged too far, may have lost capabilities
  - Sweet spot: typically KL between 1 and 10 nats for token-level KL
"""

print(f"\n--- KL Divergence During Training (Demonstration) ---")

policy_for_kl = TinyLM(vocab_size=20, d_model=16)
reference_for_kl = TinyLM(vocab_size=20, d_model=16)
reference_for_kl.load_state_dict(policy_for_kl.state_dict())
for p in reference_for_kl.parameters():
    p.requires_grad = False

test_input = torch.randint(0, 20, (16, 4))

optimizer_kl = torch.optim.AdamW(policy_for_kl.parameters(), lr=5e-3)

print(f"\n  Simulating policy divergence from reference:")
print(f"  {'Step':>6} | {'KL (nats)':>12} | {'Status':>20}")
print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*20}")

for step in range(100):
    # Dummy loss that pushes policy away from reference
    log_probs = policy_for_kl(test_input)
    dummy_loss = -log_probs[:, :, 0].mean()
    dummy_loss.backward()
    optimizer_kl.step()
    optimizer_kl.zero_grad()

    if step % 20 == 0 or step == 99:
        with torch.no_grad():
            pi_logits = policy_for_kl.head(
                policy_for_kl.act(policy_for_kl.fc(policy_for_kl.embed(test_input)))
            )
            ref_logits = reference_for_kl.head(
                reference_for_kl.act(reference_for_kl.fc(reference_for_kl.embed(test_input)))
            )
            kl = kl_divergence_categorical(pi_logits, ref_logits).mean().item()

        if kl < 1.0:
            status = "OK (low divergence)"
        elif kl < 5.0:
            status = "MODERATE"
        elif kl < 15.0:
            status = "HIGH — watch out"
        else:
            status = "DANGER — too far!"
        print(f"  {step:6d} | {kl:12.4f} | {status:>20}")


# --- 6.3 KL-penalized reward (RLHF formulation) ---------------------------

print(f"\n--- KL-Penalized Reward (RLHF) ---")

"""
WHY THIS MATTERS:
In RLHF, the training objective is NOT just to maximize reward.
It's to maximize reward MINUS a KL penalty:

  J(pi) = E[R(y|x)] - beta * KL(pi || pi_ref)

This balances "be helpful" (high reward) with "don't go crazy"
(stay close to reference). Beta controls the trade-off.
"""

reward_scores = torch.tensor([3.0, 5.0, 1.0, 4.0, 2.0])
kl_values = torch.tensor([0.5, 8.0, 0.1, 3.0, 0.2])

print(f"\n  {'Response':>8} | {'Reward':>7} | {'KL':>6} | beta=0.1  | beta=0.5  | beta=1.0")
print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*6}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}")
for i in range(5):
    r = reward_scores[i].item()
    kl = kl_values[i].item()
    j_01 = r - 0.1 * kl
    j_05 = r - 0.5 * kl
    j_10 = r - 1.0 * kl
    print(f"  {i+1:8d} | {r:7.1f} | {kl:6.1f} | {j_01:9.2f} | {j_05:9.2f} | {j_10:9.2f}")

print(f"\n  With beta=0.1: Response 2 wins (high reward, KL penalty is small)")
print(f"  With beta=1.0: Response 4 wins (good reward, moderate KL)")
print(f"  → Higher beta penalizes divergence more aggressively")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
Exercise 1: Reward Model Exploration
--------------------------------------
  a) Modify the RewardModel to have 3 hidden layers instead of 2.
  b) Train it on the same synthetic data for 300 epochs.
  c) Does the deeper model converge faster? Compare loss curves.
  d) What happens if you swap chosen and rejected during training?
     (Train with wrong labels — does accuracy go to 0%?)

Exercise 2: DPO Beta Sweep
-----------------------------
  a) Run the full DPO training loop (Part 4.4) with beta values:
     [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
  b) For each beta, record final loss and "chosen preferred" percentage.
  c) Which beta gives the best preference accuracy?
  d) What happens with very large beta (5.0)? Does training become
     unstable? Why?

Exercise 3: Implement Implicit Reward from DPO
-------------------------------------------------
DPO implicitly defines a reward function:
  r(x, y) = beta * log(pi(y|x) / pi_ref(y|x)) + beta * log Z(x)

Since Z(x) is the same for both chosen and rejected, the implicit
reward DIFFERENCE is:
  r(chosen) - r(rejected) = beta * (log_ratio_chosen - log_ratio_rejected)

  a) After DPO training (Part 4.4), compute the implicit reward for
     all 256 pairs.
  b) What fraction of pairs have positive implicit reward difference?
     (This should match the "chosen preferred" metric.)
  c) Plot a histogram (ASCII) of implicit reward differences.

Exercise 4: KL Divergence Monitoring
---------------------------------------
  a) Modify the DPO training loop to track KL divergence at each step.
  b) Compute token-level KL between policy and reference on a held-out set.
  c) Plot KL vs training step. At what point does KL start growing fast?
  d) Implement early stopping: stop training when KL exceeds a threshold
     (e.g., 10.0). Does this prevent overoptimization?

Exercise 5: Build a Complete Alignment Pipeline
--------------------------------------------------
  a) Create a preference dataset of 100 pairs using the TinyLM.
     Generate responses from the reference model, then designate
     sequences with higher reference log-prob as "chosen."
  b) Train DPO for 300 steps with beta=0.1.
  c) Compute the reward model accuracy on a held-out test set of 20 pairs.
  d) Compare: train the same data with beta=0.01 and beta=1.0.
     Which gives better test-set preference accuracy?
""")
