"""
============================================================================
 LESSON 10: Text Generation & Decoding
============================================================================

 Goal: Generate text from a trained language model using different decoding
       strategies — and understand why the choice of strategy matters.

 Topics covered:
   1. Greedy Decoding        — always pick the highest-probability token
   2. Temperature Sampling   — scale logits to control randomness
   3. Top-k Sampling         — restrict to the k most likely tokens
   4. Top-p (Nucleus)        — restrict to the smallest set with prob >= p
   5. Repetition Penalty     — discourage repeated tokens
   6. KV Cache Generation    — full generate() with caching for speed

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites:
   - Lesson 5 (attention mechanism — KV cache concept)
   - Lesson 6 (transformer architecture)
   - Lesson 8 (pre-training — next-token prediction objective)
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

torch.manual_seed(42)


# ─── Tiny transformer for demonstration ──────────────────────────────────
# We build a minimal GPT-style model so every decoding strategy runs on
# real logits from a real forward pass (not synthetic distributions).

class TinyTransformer(nn.Module):
    """
    Minimal decoder-only transformer (GPT-style).
    Small enough to run instantly on CPU, big enough to produce
    non-trivial probability distributions over a vocabulary.
    """

    def __init__(self, vocab_size=256, d_model=128, n_heads=4,
                 n_layers=2, max_seq_len=256):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, idx, kv_cache=None):
        """
        idx: (batch, seq_len) token indices
        kv_cache: list of (K, V) pairs per layer, or None

        Returns: logits (batch, seq_len, vocab), new_kv_cache
        """
        B, T = idx.shape
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0].shape[2]
        else:
            past_len = 0

        positions = torch.arange(past_len, past_len + T, device=idx.device)
        h = self.tok_emb(idx) + self.pos_emb(positions)

        new_kv_cache = []
        for i, block in enumerate(self.blocks):
            past_kv = kv_cache[i] if kv_cache is not None else None
            h, new_kv = block(h, past_kv=past_kv)
            new_kv_cache.append(new_kv)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits, new_kv_cache


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, past_kv=None):
        attn_out, new_kv = self.attn(self.ln1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, new_kv


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x, past_kv=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v)

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        # Causal mask: prevent attending to future tokens
        kv_len = k.shape[2]
        q_len = q.shape[2]
        mask = torch.triu(
            torch.ones(q_len, kv_len, device=x.device, dtype=torch.bool),
            diagonal=kv_len - q_len + 1,
        )
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out), new_kv


# Instantiate the model
torch.manual_seed(42)
model = TinyTransformer(vocab_size=256, d_model=128, n_heads=4, n_layers=2)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print(f"TinyTransformer: {total_params:,} parameters, vocab_size=256")
print(f"(Using random weights — outputs are nonsensical but distributions are real)")


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — GREEDY DECODING
# ═══════════════════════════════════════════════════════════════════════════
#
# The simplest strategy: at each step, pick the token with the highest
# probability (argmax of logits).
#
# Greedy decoding is:
#   + Deterministic — same input always produces the same output
#   + Fast — no sampling overhead
#   - Repetitive — tends to get stuck in loops
#   - Misses better sequences — locally optimal != globally optimal
#
# In practice, greedy is used for tasks where there's ONE right answer
# (e.g., translation, code completion) but NOT for creative generation.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 1: GREEDY DECODING")
print("=" * 60)

torch.manual_seed(42)


def greedy_decode(model, prompt_ids, max_new_tokens=30):
    """
    WHY THIS MATTERS:
    Greedy decoding is the baseline strategy. Always picking argmax is
    deterministic and fast, but produces repetitive, boring text.
    Understanding why it fails motivates all the sampling strategies below.
    """
    generated = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)
            next_logit = logits[:, -1, :]       # logits for last position
            next_token = next_logit.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


prompt = torch.tensor([[1, 2, 3, 4, 5]])
output = greedy_decode(model, prompt, max_new_tokens=30)
print(f"\nPrompt tokens:    {prompt[0].tolist()}")
print(f"Generated tokens: {output[0, 5:].tolist()}")

# Demonstrate determinism
output2 = greedy_decode(model, prompt, max_new_tokens=30)
print(f"Same output?      {torch.equal(output, output2)}")

# Show the repetition problem
unique_tokens = len(set(output[0, 5:].tolist()))
total_tokens = output.shape[1] - 5
print(f"Unique tokens:    {unique_tokens} / {total_tokens} "
      f"({unique_tokens/total_tokens*100:.0f}% unique)")
print(f"→ Greedy decoding often repeats the same tokens over and over.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — TEMPERATURE SAMPLING
# ═══════════════════════════════════════════════════════════════════════════
#
# Instead of always picking argmax, we SAMPLE from the probability
# distribution. Temperature controls how "sharp" or "flat" that
# distribution is:
#
#   logits_scaled = logits / temperature
#   probs = softmax(logits_scaled)
#
#   T < 1.0 → sharper (more confident, less random)
#   T = 1.0 → original distribution
#   T > 1.0 → flatter (more uniform, more creative/chaotic)
#   T → 0   → greedy (argmax)
#   T → ∞   → uniform random
#
# Temperature is the single most important generation hyperparameter.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: TEMPERATURE SAMPLING")
print("=" * 60)

torch.manual_seed(42)


def temperature_sample(logits, temperature=1.0):
    """
    WHY THIS MATTERS:
    Temperature is the simplest way to control the creativity vs coherence
    trade-off. ChatGPT's "temperature" slider uses exactly this.
    Low T → factual, repetitive. High T → creative, potentially nonsensical.
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)

    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_with_temperature(model, prompt_ids, max_new_tokens=30,
                              temperature=1.0):
    generated = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)
            next_logit = logits[:, -1, :]
            next_token = temperature_sample(next_logit, temperature)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


# Show the effect of different temperatures
prompt = torch.tensor([[1, 2, 3, 4, 5]])
print(f"\nPrompt: {prompt[0].tolist()}")

for temp in [0.5, 1.0, 2.0]:
    torch.manual_seed(42)
    out = generate_with_temperature(model, prompt, max_new_tokens=30,
                                    temperature=temp)
    tokens = out[0, 5:].tolist()
    unique = len(set(tokens))
    print(f"\n  T={temp:.1f}: {tokens}")
    print(f"        Unique: {unique}/{len(tokens)}")

# Show how temperature changes the distribution
print(f"\n--- Temperature effect on a sample distribution ---")
sample_logits = torch.tensor([[2.0, 1.0, 0.5, -0.5, -1.0, -2.0]])

for temp in [0.5, 1.0, 2.0]:
    probs = F.softmax(sample_logits / temp, dim=-1)
    top_prob = probs.max().item()
    entropy = -(probs * probs.log()).sum().item()
    probs_str = " ".join(f"{p:.3f}" for p in probs[0].tolist())
    print(f"  T={temp:.1f}: [{probs_str}]  top={top_prob:.3f}  entropy={entropy:.3f}")

print(f"\n  Lower T → peakier distribution (more confident)")
print(f"  Higher T → flatter distribution (more random)")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — TOP-K SAMPLING
# ═══════════════════════════════════════════════════════════════════════════
#
# Problem with pure temperature sampling: even with moderate T, there's
# always a small chance of sampling a very unlikely (garbage) token.
#
# Top-k sampling fixes this by zeroing out all but the top k tokens
# before sampling. This guarantees the output comes from the k most
# likely candidates.
#
# Top-k is simple and effective, but it has a fixed vocabulary size k
# regardless of the shape of the distribution. If the model is very
# confident (one token has 99% probability), k=50 still considers 49
# nearly-impossible tokens. That's where top-p helps (Part 4).
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3: TOP-K SAMPLING")
print("=" * 60)

torch.manual_seed(42)


def top_k_sample(logits, k=10, temperature=1.0):
    """
    WHY THIS MATTERS:
    Top-k prevents sampling garbage tokens from the tail of the
    distribution. GPT-2's original release used top-k=40, which
    dramatically improved output quality over pure sampling.

    Implementation: find the k-th largest logit, set everything
    below it to -inf, then apply temperature and sample.
    """
    scaled = logits / temperature

    # Find the k-th largest value as the cutoff threshold
    top_k_values, _ = torch.topk(scaled, k, dim=-1)
    threshold = top_k_values[:, -1].unsqueeze(-1)   # k-th largest

    # Zero out (set to -inf) everything below the threshold
    filtered = scaled.clone()
    filtered[filtered < threshold] = float("-inf")

    probs = F.softmax(filtered, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def generate_with_top_k(model, prompt_ids, max_new_tokens=30,
                        k=10, temperature=1.0):
    generated = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)
            next_logit = logits[:, -1, :]
            next_token = top_k_sample(next_logit, k=k, temperature=temperature)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


# Show effect of different k values
prompt = torch.tensor([[1, 2, 3, 4, 5]])
print(f"\nPrompt: {prompt[0].tolist()}")

for k in [5, 20, 50]:
    torch.manual_seed(42)
    out = generate_with_top_k(model, prompt, max_new_tokens=30,
                              k=k, temperature=1.0)
    tokens = out[0, 5:].tolist()
    unique = len(set(tokens))
    print(f"  k={k:2d}: {tokens}")
    print(f"        Unique: {unique}/{len(tokens)}")

# Demonstrate the filtering
print(f"\n--- Top-k filtering visualization ---")
sample_logits = torch.tensor([[3.0, 2.5, 2.0, 0.5, 0.0, -0.5, -1.0, -2.0]])
vocab = ["the", "a", "an", "cat", "dog", "ran", "sat", "xyz"]

orig_probs = F.softmax(sample_logits, dim=-1)
for k in [3, 5]:
    top_vals, top_idx = torch.topk(sample_logits, k, dim=-1)
    threshold = top_vals[0, -1].item()
    filtered = sample_logits.clone()
    filtered[filtered < threshold] = float("-inf")
    filtered_probs = F.softmax(filtered, dim=-1)

    print(f"\n  k={k} (threshold logit = {threshold:.1f}):")
    for i, (word, orig_p, filt_p) in enumerate(zip(
            vocab, orig_probs[0], filtered_probs[0])):
        kept = "✓" if filt_p > 0 else "✗"
        bar = "█" * int(filt_p * 40)
        print(f"    {kept} {word:4s}  orig={orig_p:.3f}  filtered={filt_p:.3f}  {bar}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — TOP-P (NUCLEUS) SAMPLING
# ═══════════════════════════════════════════════════════════════════════════
#
# Top-k uses a FIXED number of tokens regardless of the distribution shape.
# Top-p (nucleus sampling) is adaptive: it keeps the smallest set of tokens
# whose cumulative probability is >= p.
#
#   - If the model is very confident, only 1-2 tokens might be in the set.
#   - If the model is uncertain, 100+ tokens might be included.
#
# This automatically adapts to the "width" of the distribution, making it
# more principled than top-k.
#
# Top-p was introduced in "The Curious Case of Neural Text Degeneration"
# (Holtzman et al., 2020). Most modern LLM APIs use top-p as the default.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 4: TOP-P (NUCLEUS) SAMPLING")
print("=" * 60)

torch.manual_seed(42)


def top_p_sample(logits, p=0.9, temperature=1.0):
    """
    WHY THIS MATTERS:
    Top-p (nucleus sampling) is the default decoding strategy for most
    modern LLMs (ChatGPT, Claude, etc.). Unlike top-k, it adapts to
    the shape of the distribution — keeping fewer tokens when the model
    is confident, and more when it's uncertain.

    Implementation:
      1. Sort tokens by probability (descending)
      2. Compute cumulative sum
      3. Find the cutoff where cumsum >= p
      4. Zero out everything beyond the cutoff
      5. Re-normalize and sample
    """
    scaled = logits / temperature
    probs = F.softmax(scaled, dim=-1)

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find tokens to remove: cumulative prob BEFORE this token exceeds p
    # (shift right by 1 so the token that pushes past p is kept)
    sorted_mask = cumulative_probs - sorted_probs >= p

    # Set removed tokens to 0
    sorted_probs[sorted_mask] = 0.0

    # Scatter back to original ordering
    probs.scatter_(1, sorted_indices, sorted_probs)

    # Re-normalize
    probs = probs / probs.sum(dim=-1, keepdim=True)

    return torch.multinomial(probs, num_samples=1)


def generate_with_top_p(model, prompt_ids, max_new_tokens=30,
                        p=0.9, temperature=1.0):
    generated = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)
            next_logit = logits[:, -1, :]
            next_token = top_p_sample(next_logit, p=p, temperature=temperature)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


# Show effect of different p values
prompt = torch.tensor([[1, 2, 3, 4, 5]])
print(f"\nPrompt: {prompt[0].tolist()}")

for p_val in [0.5, 0.9, 0.95]:
    torch.manual_seed(42)
    out = generate_with_top_p(model, prompt, max_new_tokens=30,
                              p=p_val, temperature=1.0)
    tokens = out[0, 5:].tolist()
    unique = len(set(tokens))
    print(f"  p={p_val:.2f}: {tokens}")
    print(f"          Unique: {unique}/{len(tokens)}")

# Demonstrate adaptive behavior
print(f"\n--- Top-p adapts to distribution shape ---")

confident_logits = torch.tensor([[5.0, 1.0, 0.5, 0.1, -1.0, -2.0, -3.0, -4.0]])
uncertain_logits = torch.tensor([[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]])
labels = ["the", "a", "an", "cat", "dog", "ran", "sat", "xyz"]

for name, logits_ex in [("Confident", confident_logits),
                         ("Uncertain", uncertain_logits)]:
    probs = F.softmax(logits_ex, dim=-1)
    sorted_p, sorted_i = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_p, dim=-1)

    # Count tokens in nucleus for p=0.9
    nucleus_size = (cumsum - sorted_p < 0.9).sum().item()
    nucleus_tokens = [labels[sorted_i[0, j].item()] for j in range(nucleus_size)]

    print(f"\n  {name} distribution:")
    print(f"    Probs: {[f'{p:.3f}' for p in probs[0].tolist()]}")
    print(f"    p=0.9 nucleus: {nucleus_size} tokens → {nucleus_tokens}")
    print(f"    (Cumulative: {[f'{c:.3f}' for c in cumsum[0, :nucleus_size].tolist()]})")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — REPETITION PENALTY
# ═══════════════════════════════════════════════════════════════════════════
#
# Even with good sampling, LLMs sometimes repeat themselves — producing
# the same phrase or token over and over.
#
# Repetition penalty (Keskar et al., 2019) fixes this by reducing the
# logits of tokens that have already appeared in the output:
#
#   For each token t in the generated sequence so far:
#     if logit[t] > 0: logit[t] /= penalty
#     if logit[t] < 0: logit[t] *= penalty
#
# A penalty of 1.0 means no change. Typical values: 1.1 to 1.3.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 5: REPETITION PENALTY")
print("=" * 60)

torch.manual_seed(42)


def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """
    WHY THIS MATTERS:
    Without repetition penalty, models get stuck in loops like
    "the the the the..." or repeat entire paragraphs. This simple
    technique — dividing/multiplying logits of seen tokens — breaks
    the loop with minimal impact on output quality.
    """
    if penalty == 1.0:
        return logits

    logits = logits.clone()

    # Get the set of tokens that have already been generated
    for token_id in set(generated_ids.tolist()):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty

    return logits


def generate_with_repetition_penalty(model, prompt_ids, max_new_tokens=30,
                                     temperature=1.0, penalty=1.2):
    generated = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)
            next_logit = logits[:, -1, :]

            # Apply penalty based on all tokens generated so far
            next_logit = apply_repetition_penalty(
                next_logit, generated[0], penalty
            )

            next_token = temperature_sample(next_logit, temperature)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


# Compare with and without repetition penalty
prompt = torch.tensor([[1, 2, 3, 4, 5]])
print(f"\nPrompt: {prompt[0].tolist()}")

for penalty in [1.0, 1.2, 1.5]:
    torch.manual_seed(42)
    out = generate_with_repetition_penalty(
        model, prompt, max_new_tokens=30, temperature=1.0, penalty=penalty
    )
    tokens = out[0, 5:].tolist()
    unique = len(set(tokens))
    print(f"  penalty={penalty:.1f}: {tokens}")
    print(f"              Unique: {unique}/{len(tokens)}")

# Demonstrate the effect on logits
print(f"\n--- How repetition penalty modifies logits ---")
sample_logits = torch.tensor([[3.0, 2.0, 1.0, -0.5, -1.0]])
already_seen = torch.tensor([0, 1, 3])  # tokens 0, 1, and 3 were already generated
labels = ["tok_0", "tok_1", "tok_2", "tok_3", "tok_4"]

print(f"  Original logits:  {sample_logits[0].tolist()}")
print(f"  Already seen:     tokens {already_seen.tolist()}")

for pen in [1.0, 1.2, 1.5]:
    penalized = sample_logits.clone()
    for tid in already_seen:
        if penalized[0, tid] > 0:
            penalized[0, tid] /= pen
        else:
            penalized[0, tid] *= pen
    print(f"  penalty={pen:.1f}:       {[f'{v:.2f}' for v in penalized[0].tolist()]}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — KV CACHE GENERATION
# ═══════════════════════════════════════════════════════════════════════════
#
# Autoregressive generation WITHOUT caching recomputes attention for ALL
# previous tokens at every step. For a sequence of length L, this means:
#   Step 1: attend over 1 token
#   Step 2: attend over 2 tokens (recomputing token 1)
#   Step 3: attend over 3 tokens (recomputing tokens 1-2)
#   ...
#   Step L: attend over L tokens (recomputing tokens 1 to L-1)
#   Total work: O(L²) in sequence length
#
# KV caching stores the key and value projections from previous steps.
# At each new step, we only compute Q, K, V for the NEW token, then
# concatenate with the cached K, V from all previous tokens.
#   Total work: O(L) — each token is projected exactly once
#
# This is the standard optimization used in ALL production LLM inference.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 6: KV CACHE GENERATION")
print("=" * 60)

torch.manual_seed(42)


def generate_no_cache(model, prompt_ids, max_new_tokens=50,
                      temperature=1.0, top_k=20):
    """Generate WITHOUT KV cache — recomputes all tokens every step."""
    generated = prompt_ids.clone()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(generated)          # full sequence every time
            next_logit = logits[:, -1, :]
            next_token = top_k_sample(next_logit, k=top_k,
                                      temperature=temperature)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


def generate_with_kv_cache(model, prompt_ids, max_new_tokens=50,
                           temperature=1.0, top_k=20):
    """
    WHY THIS MATTERS:
    KV caching is the single biggest inference optimization for LLMs.
    Without it, generating 1000 tokens from GPT-3 would recompute
    attention across ~500K cumulative token positions. With caching,
    each new token only attends once, giving a massive speedup.

    Every production LLM server (vLLM, TGI, llama.cpp) uses KV caching.
    """
    generated = prompt_ids.clone()

    with torch.no_grad():
        # Prefill: process the entire prompt at once
        logits, kv_cache = model(generated)
        next_logit = logits[:, -1, :]
        next_token = top_k_sample(next_logit, k=top_k,
                                  temperature=temperature)
        generated = torch.cat([generated, next_token], dim=1)

        # Decode: process one token at a time, reusing cached K/V
        for _ in range(max_new_tokens - 1):
            logits, kv_cache = model(next_token, kv_cache=kv_cache)
            next_logit = logits[:, -1, :]
            next_token = top_k_sample(next_logit, k=top_k,
                                      temperature=temperature)
            generated = torch.cat([generated, next_token], dim=1)

    return generated


# Verify correctness: both methods should produce the same output
torch.manual_seed(42)
out_no_cache = generate_no_cache(model, prompt, max_new_tokens=20,
                                 temperature=0, top_k=1)
torch.manual_seed(42)
out_cached = generate_with_kv_cache(model, prompt, max_new_tokens=20,
                                    temperature=0, top_k=1)
print(f"\nCorrectness check (greedy, should match):")
print(f"  No cache:  {out_no_cache[0, 5:].tolist()}")
print(f"  KV cache:  {out_cached[0, 5:].tolist()}")
print(f"  Match:     {torch.equal(out_no_cache, out_cached)}")


# Benchmark speed
print(f"\n--- Speed comparison ---")
num_tokens = 100
prompt_bench = torch.tensor([[1, 2, 3, 4, 5]])

# Warm up
_ = generate_no_cache(model, prompt_bench, max_new_tokens=5, temperature=0, top_k=1)
_ = generate_with_kv_cache(model, prompt_bench, max_new_tokens=5, temperature=0, top_k=1)

# Time without cache
start = time.perf_counter()
for _ in range(3):
    _ = generate_no_cache(model, prompt_bench, max_new_tokens=num_tokens,
                          temperature=0, top_k=1)
no_cache_time = (time.perf_counter() - start) / 3

# Time with cache
start = time.perf_counter()
for _ in range(3):
    _ = generate_with_kv_cache(model, prompt_bench, max_new_tokens=num_tokens,
                               temperature=0, top_k=1)
cache_time = (time.perf_counter() - start) / 3

print(f"  Generating {num_tokens} tokens (averaged over 3 runs):")
print(f"  Without KV cache: {no_cache_time:.4f}s  ({num_tokens/no_cache_time:.0f} tok/s)")
print(f"  With KV cache:    {cache_time:.4f}s  ({num_tokens/cache_time:.0f} tok/s)")
print(f"  Speedup:          {no_cache_time/cache_time:.2f}×")

# Show cache growth
print(f"\n--- KV cache memory growth ---")
with torch.no_grad():
    _, kv = model(prompt_bench)
    for step in range(1, 6):
        new_tok = torch.tensor([[42]])
        _, kv = model(new_tok, kv_cache=kv)
        k_shape = kv[0][0].shape  # (batch, heads, seq_len, d_head)
        cache_elements = sum(
            k.numel() + v.numel() for k, v in kv
        )
        cache_mb = cache_elements * 4 / 1e6  # float32
        print(f"  Step {step}: K shape={list(k_shape)}  "
              f"total cache = {cache_elements:,} elements ({cache_mb:.3f} MB)")

print(f"\n  Cache grows linearly with sequence length.")
print(f"  For large models, KV cache can use tens of GB of GPU memory.")
print(f"  Techniques like GQA, MQA, and PagedAttention (vLLM) help manage this.")


# ═══════════════════════════════════════════════════════════════════════════
# COMPLETE generate() FUNCTION — COMBINING ALL STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

def generate(model, prompt_ids, max_new_tokens=50, temperature=1.0,
             top_k=0, top_p=1.0, repetition_penalty=1.0, use_cache=True):
    """
    Production-style generate() combining all decoding strategies.

    Args:
        model: the language model
        prompt_ids: (1, seq_len) tensor of prompt token IDs
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (0 = greedy)
        top_k: number of top tokens to keep (0 = disabled)
        top_p: cumulative probability threshold (1.0 = disabled)
        repetition_penalty: penalty for repeated tokens (1.0 = disabled)
        use_cache: whether to use KV caching
    """
    generated = prompt_ids.clone()
    kv_cache = None

    with torch.no_grad():
        if use_cache:
            logits, kv_cache = model(generated)
            next_logit = logits[:, -1:, :]
        else:
            logits, _ = model(generated)
            next_logit = logits[:, -1:, :]

        for step in range(max_new_tokens):
            logit = next_logit[:, -1, :]

            if repetition_penalty != 1.0:
                logit = apply_repetition_penalty(
                    logit, generated[0], repetition_penalty
                )

            if temperature == 0:
                next_token = logit.argmax(dim=-1, keepdim=True)
            else:
                scaled = logit / temperature

                if top_k > 0:
                    top_vals, _ = torch.topk(scaled, top_k, dim=-1)
                    threshold = top_vals[:, -1].unsqueeze(-1)
                    scaled[scaled < threshold] = float("-inf")

                probs = F.softmax(scaled, dim=-1)

                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(
                        probs, descending=True, dim=-1
                    )
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum - sorted_probs >= top_p
                    sorted_probs[mask] = 0.0
                    probs.scatter_(1, sorted_idx, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            if use_cache:
                next_logit, kv_cache = model(next_token, kv_cache=kv_cache)
            else:
                logits, _ = model(generated)
                next_logit = logits[:, -1:, :]

    return generated


# Demo: full generate() with combined strategies
print("\n" + "=" * 60)
print("COMPLETE generate() — ALL STRATEGIES COMBINED")
print("=" * 60)

prompt = torch.tensor([[1, 2, 3, 4, 5]])

configs = [
    {"temperature": 0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0},
    {"temperature": 0.8, "top_k": 50, "top_p": 1.0, "repetition_penalty": 1.0},
    {"temperature": 0.8, "top_k": 0, "top_p": 0.9, "repetition_penalty": 1.0},
    {"temperature": 0.8, "top_k": 50, "top_p": 0.9, "repetition_penalty": 1.2},
]

for cfg in configs:
    torch.manual_seed(42)
    out = generate(model, prompt, max_new_tokens=25, **cfg)
    tokens = out[0, 5:].tolist()
    unique = len(set(tokens))
    desc = ", ".join(f"{k}={v}" for k, v in cfg.items())
    print(f"\n  {desc}")
    print(f"  → {tokens}  ({unique} unique)")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
Exercise 1: Temperature Extremes
----------------------------------
  a) Generate 50 tokens with T=0.01 — is it identical to greedy (T=0)?
  b) Generate 50 tokens with T=100.0 — is it uniform random?
  c) Plot the entropy of the output distribution as a function of T
     for T in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0].
  d) At what temperature does entropy reach 90% of maximum entropy
     (which is log(vocab_size))?

Exercise 2: Top-k vs Top-p Head-to-Head
-----------------------------------------
  a) For each of these distributions, compute the nucleus size (p=0.9)
     and compare to a fixed k=10:
       - Confident: one token at 0.85, rest split evenly
       - Moderate:  top 5 tokens sum to 0.9, rest split evenly
       - Flat:      nearly uniform over 100 tokens
  b) In which case does top-k include too many junk tokens?
  c) In which case does top-k exclude good tokens?
  d) Why do most APIs default to top-p rather than top-k?

Exercise 3: Implement Beam Search
-----------------------------------
  Beam search maintains B hypotheses (beams) at each step:
  a) Start with the prompt. At step 1, expand to top-B tokens.
  b) At each subsequent step, expand each beam by top-B tokens
     (giving B×B candidates), then keep only the top-B by total
     log-probability.
  c) After max_new_tokens steps, return the beam with highest
     total log-probability.
  d) Compare beam search (B=5) output to greedy. Is it different?
  e) Why is beam search common for translation but rare for chat?

Exercise 4: Frequency + Presence Penalty
------------------------------------------
  OpenAI's API uses two separate penalties:
    - frequency_penalty: proportional to how many times a token appeared
    - presence_penalty:  flat penalty if a token appeared at all
  a) Implement both penalties:
       logit[t] -= frequency_penalty * count(t)
       logit[t] -= presence_penalty * (1 if t in seen else 0)
  b) Generate 50 tokens with frequency_penalty=0.5 and presence_penalty=0.5
  c) Compare to the repetition penalty from Part 5.
  d) Which approach is better at preventing short repetitive loops?
     Which is better at encouraging topic diversity?

Exercise 5: KV Cache Memory Calculator
----------------------------------------
  a) For a model with d_model=4096, n_heads=32, n_layers=32,
     calculate the KV cache size (in GB, FP16) for:
       - seq_len = 2048
       - seq_len = 8192
       - seq_len = 128000 (GPT-4 context)
  b) How does Grouped Query Attention (GQA) reduce KV cache?
     If n_kv_heads=8 instead of 32, what's the new cache size?
  c) For a batch of 32 concurrent users, each with 4K context,
     how much GPU memory does the KV cache alone consume?
  d) Explain why PagedAttention (vLLM) helps when batch sizes vary.
""")
