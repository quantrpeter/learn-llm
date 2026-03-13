"""
============================================================================
 LESSON 5: Attention Mechanism — The Core Innovation
============================================================================

 Goal: Deeply understand self-attention, the mechanism that makes LLMs
       possible.

 Topics covered:
   1. Intuition                  — Attention as "soft dictionary lookup"
   2. Scaled Dot-Product Attention — QK^T / sqrt(d_k), softmax, output
   3. Multi-Head Attention        — Split into heads, parallel, concatenate
   4. Causal Masking              — Lower-triangular mask for autoregressive
   5. KV Cache                    — Caching keys/values for fast inference
   6. Attention Complexity        — O(n^2) scaling with sequence length

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites: Lessons 1-4 (math, PyTorch, nn building blocks, tokenization)
============================================================================
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — INTUITION: Attention as "Soft Dictionary Lookup"
# ═══════════════════════════════════════════════════════════════════════════
#
# Think of a regular Python dictionary:
#   d = {"cat": info_about_cats, "dog": info_about_dogs}
#   d["cat"]  → exact match, returns one value
#
# Attention is a SOFT dictionary lookup:
#   - Query (Q): what you're looking for
#   - Keys  (K): the index — what each entry is "about"
#   - Values (V): the content — what each entry actually contains
#
# Instead of matching one key exactly, the query compares to ALL keys,
# gets a similarity score for each, and returns a weighted mix of all
# values. The more similar a key is to the query, the more its value
# contributes to the output.
#
# This is how a transformer decides which other tokens to "pay attention
# to" when processing each token in a sentence.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: INTUITION — Attention as Soft Dictionary Lookup")
print("=" * 70)


# --- 1.1 Concrete example with 3 sentences --------------------------------

d_k = 4

sentences = ["The cat sat on the mat", "Dogs are loyal animals", "The kitten played with yarn"]
keys = torch.randn(3, d_k)
values = torch.randn(3, d_k)

print(f"\nOur 'database' of {len(sentences)} sentences:")
for i, s in enumerate(sentences):
    print(f"  Key[{i}] = {keys[i].tolist()}  →  '{s}'")

query = torch.randn(1, d_k)
print(f"\nQuery vector: {query.squeeze().tolist()}")
print("(Imagine this represents the question: 'What did the small cat do?')")


# --- 1.2 Compute similarity (dot product) ---------------------------------

"""
WHY THIS MATTERS:
The dot product measures how "aligned" two vectors are — the same
operation from Lesson 1. A higher dot product means the query is more
similar to that key.
"""

scores = query @ keys.T                        # (1, d_k) @ (d_k, 3) → (1, 3)
print(f"\nRaw similarity scores (Q @ K^T): {scores.squeeze().tolist()}")
print("Higher score = query is more similar to that key")


# --- 1.3 Normalize to weights (softmax) ------------------------------------

weights = F.softmax(scores, dim=-1)             # (1, 3) — sums to 1
print(f"\nAttention weights (softmax): {weights.squeeze().tolist()}")
print(f"Sum of weights: {weights.sum().item():.4f}")

for i, s in enumerate(sentences):
    print(f"  '{s}' gets weight {weights[0, i].item():.4f}")


# --- 1.4 Weighted sum of values = output -----------------------------------

output = weights @ values                       # (1, 3) @ (3, d_k) → (1, d_k)
print(f"\nOutput (weighted sum of values): {output.squeeze().tolist()}")
print("The output blends all values, favoring those with higher weights.")
print("This IS attention — a differentiable, soft lookup.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — SCALED DOT-PRODUCT ATTENTION (from scratch)
# ═══════════════════════════════════════════════════════════════════════════
#
# The full formula from "Attention Is All You Need" (Vaswani et al., 2017):
#
#   Attention(Q, K, V) = softmax( Q @ K^T / sqrt(d_k) ) @ V
#
# Why divide by sqrt(d_k)?  When d_k is large, the dot products grow in
# magnitude, pushing softmax into regions where its gradient is tiny.
# Dividing by sqrt(d_k) keeps the variance of scores around 1, making
# softmax well-behaved.
#
# We implement every step from scratch so you can see exactly what happens.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: SCALED DOT-PRODUCT ATTENTION (from scratch)")
print("=" * 70)

torch.manual_seed(42)


# --- 2.1 Setup: Q, K, V matrices ------------------------------------------

seq_len = 4
d_k = 8

Q = torch.randn(seq_len, d_k)   # 4 queries (one per token)
K = torch.randn(seq_len, d_k)   # 4 keys
V = torch.randn(seq_len, d_k)   # 4 values

print(f"\nQ shape: {Q.shape}  — {seq_len} tokens, d_k={d_k}")
print(f"K shape: {K.shape}")
print(f"V shape: {V.shape}")


# --- 2.2 Step 1: raw scores = Q @ K^T -------------------------------------

scores = Q @ K.T                 # (4, 8) @ (8, 4) → (4, 4)
print(f"\nStep 1 — Raw scores (Q @ K^T):")
print(f"  Shape: {scores.shape}  — each token has a score for every other token")
print(f"  scores[i][j] = how much token i should attend to token j")
print(f"  Scores:\n{scores}")


# --- 2.3 Step 2: scale by sqrt(d_k) ---------------------------------------

scale = math.sqrt(d_k)
scores_scaled = scores / scale   # (4, 4)
print(f"\nStep 2 — Scaled scores (/ sqrt({d_k}) = / {scale:.2f}):")
print(f"  Before scaling, variance of scores: {scores.var().item():.4f}")
print(f"  After scaling, variance of scores:  {scores_scaled.var().item():.4f}")
print(f"  Scaled scores:\n{scores_scaled}")


# --- 2.4 Step 3: softmax → attention weights ------------------------------

weights = F.softmax(scores_scaled, dim=-1)  # (4, 4)
print(f"\nStep 3 — Attention weights (softmax of scaled scores):")
print(f"  Shape: {weights.shape}")
print(f"  Each row sums to 1:")
for i in range(seq_len):
    print(f"    Row {i}: {weights[i].tolist()}  sum={weights[i].sum().item():.4f}")


# --- 2.5 Step 4: output = weights @ V -------------------------------------

output = weights @ V             # (4, 4) @ (4, 8) → (4, 8)
print(f"\nStep 4 — Output (weights @ V):")
print(f"  Shape: {output.shape}  — same shape as Q")
print(f"  Each token's output is a weighted sum of all values.")


# --- 2.6 Package as a function --------------------------------------------

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V"""
    d_k = Q.size(-1)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights


output2, weights2 = scaled_dot_product_attention(Q, K, V)
print(f"\nFunction output matches manual: {torch.allclose(output, output2)}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — MULTI-HEAD ATTENTION
# ═══════════════════════════════════════════════════════════════════════════
#
# A single attention head can only learn one kind of relationship.
# Multi-head attention runs MULTIPLE attention heads in parallel, each
# focusing on different aspects:
#   - Head 1 might learn syntactic relationships (subject → verb)
#   - Head 2 might learn semantic similarity (synonyms)
#   - Head 3 might learn positional patterns (nearby words)
#
# How it works:
#   1. Project input to Q, K, V with learned weight matrices
#   2. Split Q, K, V into n_heads pieces (d_k = d_model / n_heads)
#   3. Run scaled dot-product attention on each head independently
#   4. Concatenate all head outputs
#   5. Project back to d_model with a final linear layer
#
# d_model=32, n_heads=4 → each head works with d_k=8
# Total computation is the same — we just partition the dimensions.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: MULTI-HEAD ATTENTION")
print("=" * 70)

torch.manual_seed(42)


# --- 3.1 Understand the split ---------------------------------------------

d_model = 32
n_heads = 4
d_k = d_model // n_heads

print(f"\nd_model={d_model}, n_heads={n_heads}, d_k per head={d_k}")

x = torch.randn(1, 6, d_model)  # (batch=1, seq_len=6, d_model=32)
print(f"Input shape: {x.shape}")

W_q = nn.Linear(d_model, d_model, bias=False)
Q = W_q(x)                      # (1, 6, 32)
print(f"After projection: Q shape = {Q.shape}")

Q_heads = Q.view(1, 6, n_heads, d_k).transpose(1, 2)
print(f"After split into heads: Q shape = {Q_heads.shape}")
print(f"  → (batch, n_heads, seq_len, d_k) = (1, 4, 6, 8)")
print(f"  Each head now independently operates on d_k={d_k} dimensions")


# --- 3.2 Full MultiHeadAttention class ------------------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention as described in "Attention Is All You Need".

    This is the exact component used in every transformer block in GPT,
    BERT, LLaMA, and all modern LLMs. The only difference between a 124M
    parameter GPT-2 and a 175B parameter GPT-3 is the size of d_model
    and n_heads.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into heads: (batch, seq_len, d_model) → (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention per head
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        context = weights @ V  # (batch, n_heads, seq_len, d_k)

        # Concatenate heads: → (batch, seq_len, d_model)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)

        return self.W_o(context)


# --- 3.3 Run it -----------------------------------------------------------

mha = MultiHeadAttention(d_model=32, n_heads=4)
x = torch.randn(2, 6, 32)  # (batch=2, seq_len=6, d_model=32)
out = mha(x)

print(f"\nMultiHeadAttention(d_model=32, n_heads=4):")
print(f"  Input:  {x.shape}")
print(f"  Output: {out.shape}")

total_params = sum(p.numel() for p in mha.parameters())
print(f"\n  Parameters:")
for name, p in mha.named_parameters():
    print(f"    {name:12s}  {str(p.shape):18s}  {p.numel():,}")
print(f"    {'TOTAL':12s}  {'':18s}  {total_params:,}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — CAUSAL MASKING
# ═══════════════════════════════════════════════════════════════════════════
#
# In autoregressive language models (GPT, LLaMA), each token can only
# attend to tokens at the SAME or EARLIER positions — never the future.
#
# Why? Because at generation time, future tokens don't exist yet.
# Training must match this constraint, so we apply a causal mask:
#   - Lower triangle (including diagonal) = 1 (allowed)
#   - Upper triangle = 0 (blocked → set to -inf before softmax)
#
# After softmax, blocked positions become 0 weight — they contribute
# nothing to the output.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: CAUSAL MASKING")
print("=" * 70)

torch.manual_seed(42)


# --- 4.1 Create the causal mask -------------------------------------------

seq_len = 6
mask = torch.tril(torch.ones(seq_len, seq_len))

print(f"\nCausal mask (seq_len={seq_len}):")
print(f"  torch.tril creates a lower-triangular matrix:")
print(f"{mask}")


# --- 4.2 Visualize as a grid ----------------------------------------------

tokens = ["The", "cat", "sat", "on", "the", "mat"]

print(f"\nVisualization (✓ = can attend, ✗ = blocked):\n")
header = "Query\\Key  " + "  ".join(f"{t:>5s}" for t in tokens)
print(header)
print("-" * len(header))
for i in range(seq_len):
    row = f"  {tokens[i]:>5s}     "
    for j in range(seq_len):
        if mask[i, j] == 1:
            row += "    ✓ "
        else:
            row += "    ✗ "
    print(row)

print(f"\nKey insight: row i has 1s only in columns 0..i")
print(f"  → token at position i can ONLY see tokens at positions 0, 1, ..., i")
print(f"  → position 0 sees only itself (1 token)")
print(f"  → position 5 sees everything (6 tokens)")


# --- 4.3 Apply mask to attention ------------------------------------------

Q = torch.randn(seq_len, 8)
K = torch.randn(seq_len, 8)
V = torch.randn(seq_len, 8)

scores = Q @ K.T / math.sqrt(8)
print(f"\nScores BEFORE masking (sample row 2):")
print(f"  {scores[2].tolist()}")

scores_masked = scores.masked_fill(mask == 0, float("-inf"))
print(f"Scores AFTER masking (row 2 — positions 3,4,5 are -inf):")
print(f"  {scores_masked[2].tolist()}")

weights = F.softmax(scores_masked, dim=-1)
print(f"Weights after softmax (row 2 — positions 3,4,5 become 0.0):")
print(f"  {weights[2].tolist()}")
print(f"  Sum: {weights[2].sum().item():.4f}")
print(f"\n  -inf → softmax → 0.0 — future tokens contribute NOTHING")


# --- 4.4 Verify: position i gets zero weight for j > i --------------------

print(f"\nVerification — future attention weights are exactly 0:")
all_zero = True
for i in range(seq_len):
    for j in range(i + 1, seq_len):
        w = weights[i, j].item()
        if w != 0.0:
            print(f"  FAIL: position {i} attends to future position {j} with weight {w}")
            all_zero = False
if all_zero:
    print("  PASS: No position attends to any future position")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — KV CACHE (inference optimization)
# ═══════════════════════════════════════════════════════════════════════════
#
# During autoregressive generation (one token at a time):
#   Step 1: process token 1 → compute K1, V1
#   Step 2: process token 2 → need attention over [K1, K2], [V1, V2]
#   Step 3: process token 3 → need attention over [K1, K2, K3], [V1, V2, V3]
#
# PROBLEM: Without caching, at step N you'd recompute K and V for ALL N
# tokens from scratch — O(N^2) total work across all steps.
#
# SOLUTION: KV Cache. After each step, store the new K and V. At the
# next step, only compute K and V for the NEW token, then concatenate
# with the cache. The query only needs to be for the new token.
#
# This reduces per-step work from O(N * d) to O(d), and total generation
# from O(N^2 * d) to O(N * d).
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: KV CACHE")
print("=" * 70)

torch.manual_seed(42)


# --- 5.1 Attention module with KV cache -----------------------------------

class CachedAttention(nn.Module):
    """Single-head attention with KV cache support for autoregressive inference."""

    def __init__(self, d_model: int):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x_new: torch.Tensor,
        k_cache: torch.Tensor = None,
        v_cache: torch.Tensor = None,
    ):
        """
        x_new:   (batch, new_tokens, d_model) — during generation, new_tokens=1
        k_cache: (batch, past_len, d_model)   — cached keys from previous steps
        v_cache: (batch, past_len, d_model)   — cached values from previous steps

        Returns: output, updated_k_cache, updated_v_cache
        """
        q = self.W_q(x_new)
        k_new = self.W_k(x_new)
        v_new = self.W_v(x_new)

        if k_cache is not None:
            k = torch.cat([k_cache, k_new], dim=1)
            v = torch.cat([v_cache, v_new], dim=1)
        else:
            k = k_new
            v = v_new

        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
        weights = F.softmax(scores, dim=-1)
        output = weights @ v

        return output, k, v


# --- 5.2 Step-by-step generation with cache --------------------------------

d_model = 16
model = CachedAttention(d_model)
model.eval()

full_sequence = torch.randn(1, 4, d_model)  # 4-token input

print(f"\nSimulating autoregressive generation with KV cache:")
print(f"Full sequence shape: {full_sequence.shape}")

k_cache = None
v_cache = None

for step in range(4):
    x_new = full_sequence[:, step : step + 1, :]  # (1, 1, d_model)

    with torch.no_grad():
        out, k_cache, v_cache = model(x_new, k_cache, v_cache)

    print(f"\n  Step {step + 1}: process token {step}")
    print(f"    x_new shape:   {x_new.shape}         — only the NEW token")
    print(f"    k_cache shape: {k_cache.shape}  — cache grows by 1")
    print(f"    v_cache shape: {v_cache.shape}")
    print(f"    output shape:  {out.shape}          — output for new token only")


# --- 5.3 Verify: cached output matches full recomputation ------------------

print(f"\n--- Verify correctness ---")

with torch.no_grad():
    q_full = model.W_q(full_sequence)
    k_full = model.W_k(full_sequence)
    v_full = model.W_v(full_sequence)

    scores_full = q_full @ k_full.transpose(-2, -1) / math.sqrt(d_model)

    causal_mask = torch.tril(torch.ones(4, 4))
    scores_full = scores_full.masked_fill(causal_mask == 0, float("-inf"))

    weights_full = F.softmax(scores_full, dim=-1)
    out_full = weights_full @ v_full

    out_last_cached = out.squeeze()
    out_last_full = out_full[0, 3]

    print(f"Cached output (step 4):     {out_last_cached[:4].tolist()}")
    print(f"Full recompute (position 3): {out_last_full[:4].tolist()}")
    print(f"Match: {torch.allclose(out_last_cached, out_last_full, atol=1e-5)}")

print(f"\nKey insight: the cached version produces identical results but")
print(f"only computes Q, K, V for ONE new token per step instead of all.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — ATTENTION COMPLEXITY: O(n^2)
# ═══════════════════════════════════════════════════════════════════════════
#
# The attention score matrix is (seq_len × seq_len). Computing it requires
# seq_len^2 dot products, each of dimension d_k. Total: O(n^2 * d).
#
# This means:
#   - Double the sequence length → 4× the compute and memory
#   - GPT-4's 128K context window needs 128K^2 = 16 BILLION score entries
#
# This quadratic cost is why:
#   - Long context is expensive (Flash Attention, sparse attention)
#   - Context window sizes are limited
#   - KV cache is critical for inference
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 6: ATTENTION COMPLEXITY — O(n^2)")
print("=" * 70)

torch.manual_seed(42)


# --- 6.1 Measure attention time at different sequence lengths ---------------

d_model = 64
n_heads = 4
d_head = d_model // n_heads
seq_lengths = [128, 256, 512, 1024]
n_trials = 50

print(f"\nTiming attention with d_model={d_model}, n_heads={n_heads}:")
print(f"{'seq_len':>10s}  {'time (ms)':>10s}  {'relative':>10s}  {'score_matrix':>15s}")
print("-" * 55)

times = []
for seq_len in seq_lengths:
    Q = torch.randn(1, n_heads, seq_len, d_head)
    K = torch.randn(1, n_heads, seq_len, d_head)
    V = torch.randn(1, n_heads, seq_len, d_head)

    # Warmup
    for _ in range(5):
        _ = F.softmax(Q @ K.transpose(-2, -1) / math.sqrt(d_head), dim=-1) @ V

    start = time.perf_counter()
    for _ in range(n_trials):
        scores = Q @ K.transpose(-2, -1) / math.sqrt(d_head)
        weights = F.softmax(scores, dim=-1)
        _ = weights @ V
    elapsed = (time.perf_counter() - start) / n_trials * 1000  # ms

    times.append(elapsed)
    relative = elapsed / times[0]
    matrix_size = f"{seq_len}x{seq_len} = {seq_len**2:,}"
    print(f"{seq_len:>10d}  {elapsed:>10.2f}  {relative:>10.2f}x  {matrix_size:>15s}")


# --- 6.2 Show the quadratic relationship ----------------------------------

print(f"\nQuadratic scaling check (if O(n^2), doubling n → 4x time):")
for i in range(1, len(seq_lengths)):
    n_ratio = seq_lengths[i] / seq_lengths[0]
    expected_ratio = n_ratio ** 2
    actual_ratio = times[i] / times[0]
    print(
        f"  seq_len {seq_lengths[i]:>5d} vs {seq_lengths[0]:>5d}: "
        f"n_ratio={n_ratio:.1f}x  expected={expected_ratio:.1f}x  "
        f"actual={actual_ratio:.1f}x"
    )


# --- 6.3 Score matrix size in memory --------------------------------------

print(f"\nMemory for attention score matrix (float32, 4 bytes/element):")
for seq_len in [512, 2048, 8192, 32768, 131072]:
    n_entries = seq_len * seq_len * n_heads
    memory_mb = n_entries * 4 / (1024 ** 2)
    print(f"  seq_len={seq_len:>6d}  heads={n_heads}  "
          f"entries={n_entries:>15,}  memory={memory_mb:>10.1f} MB")

print(f"\nThis is why:")
print(f"  - Flash Attention fuses the computation to avoid materializing the full matrix")
print(f"  - Sparse attention patterns (local + global) reduce from O(n^2) to O(n*sqrt(n))")
print(f"  - Context window sizes are a major cost driver")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXERCISES")
print("=" * 70)
print("""
Exercise 1: Single-Head Attention from Scratch
------------------------------------------------
Implement single-head attention WITHOUT using nn.Linear or any class:
  a) Create Q, K, V weight matrices as raw tensors with requires_grad=True
     W_q, W_k, W_v: shape (d_model, d_model)
  b) Given input x of shape (1, 5, 16), compute:
     Q = x @ W_q,  K = x @ W_k,  V = x @ W_v
  c) Compute scores = Q @ K^T / sqrt(d_model)
  d) Apply softmax to get weights
  e) Compute output = weights @ V
  f) Verify output shape is (1, 5, 16)
Hint: use torch.randn(d_model, d_model, requires_grad=True)

Exercise 2: Multi-Head Attention
---------------------------------
Extend the MultiHeadAttention class to also return the attention weights:
  a) Modify the forward() to return both the output and the attention
     weight tensor of shape (batch, n_heads, seq_len, seq_len)
  b) Create an instance with d_model=64, n_heads=8
  c) Pass in a random tensor of shape (1, 10, 64)
  d) Print the attention weight shape — it should be (1, 8, 10, 10)
  e) Verify that weights[0, h, :, :].sum(dim=-1) is all 1s for each head h
     (each row in each head must sum to 1)

Exercise 3: Verify Causal Mask Blocks Future
----------------------------------------------
  a) Create random Q, K, V of shape (1, 8, d_k=16)
  b) Run attention WITHOUT a causal mask → get weights_no_mask
  c) Run attention WITH a causal mask → get weights_masked
  d) For every position i, check that weights_masked[0, i, j] == 0
     for all j > i
  e) Confirm that weights_no_mask does NOT have this property
     (some future weights should be > 0)
  f) Print a pass/fail summary

Exercise 4: Implement KV Cache Generate Loop
----------------------------------------------
  a) Create a CachedAttention module with d_model=32
  b) Create a "prompt" of 3 tokens: shape (1, 3, 32)
  c) Process the prompt in one forward pass (no cache)
     → this gives you the initial k_cache and v_cache
  d) Then generate 5 new tokens autoregressively:
     - For each step, create a random new token: (1, 1, 32)
     - Pass it through with the cache
     - Verify cache grows: k_cache.shape[1] should be 3, 4, 5, 6, 7
  e) Print the cache size at each step

Exercise 5: Measure Attention Time Scaling
-------------------------------------------
  a) Time the scaled_dot_product_attention function for seq_len in
     [64, 128, 256, 512, 1024, 2048]
  b) Use d_k=32, run each 100 times, take the average
  c) Plot the results as a text bar chart (like Lesson 2's loss curve)
  d) Compute the ratio: time(2N) / time(N) for each consecutive pair
     → it should be approximately 4x (proving O(n^2))
  e) At what sequence length does a single attention call take >10ms
     on your machine?
""")
