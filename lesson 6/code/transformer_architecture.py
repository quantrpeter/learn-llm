"""
============================================================================
 LESSON 6: The Transformer Architecture
============================================================================

 Goal: Assemble all components from previous lessons into a complete
       transformer decoder (GPT-style) and understand every design choice.

 Topics covered:
   1. Encoder-Decoder vs Decoder-Only  — why GPT/LLaMA use decoder-only
   2. Transformer Block                — attention + FFN + residual + norm
   3. Positional Encoding              — sinusoidal PE and RoPE
   4. Full GPT Model                   — stack blocks into a complete model
   5. Model Dimensions                 — d_model, n_heads, n_layers, d_ff
   6. Generating Text                  — greedy autoregressive generation

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites:
   - Lesson 3 (LayerNorm, residual connections, FFN)
   - Lesson 4 (tokenization, embeddings)
   - Lesson 5 (attention mechanism, multi-head attention, causal masking)
============================================================================
"""

import torch
import torch.nn as nn
import math

torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — ENCODER-DECODER vs DECODER-ONLY
# ═══════════════════════════════════════════════════════════════════════════
#
# The original Transformer (Vaswani et al., 2017) had TWO halves:
#   - Encoder: reads the full input, builds a representation
#   - Decoder: generates output tokens one at a time, attending to encoder
#
# This encoder-decoder design was built for TRANSLATION: encode a French
# sentence, then decode it into English.
#
# GPT changed everything by using DECODER-ONLY:
#   - No encoder at all
#   - The model just predicts the next token given all previous tokens
#   - Same architecture for understanding AND generation
#
# Why decoder-only won for language modeling:
#   1. Simpler — one stack of blocks instead of two
#   2. Unified — same model does "reading" and "writing"
#   3. Scalable — easier to scale to billions of parameters
#   4. Empirical — it just works better for generative tasks
#
# Models that use decoder-only: GPT-2, GPT-3, GPT-4, LLaMA, Mistral,
# Claude, Gemini (partially). It's the dominant architecture.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: ENCODER-DECODER vs DECODER-ONLY")
print("=" * 60)

print("""
Original Transformer (2017) — Encoder-Decoder:
  ┌─────────┐          ┌─────────┐
  │ Encoder │ ──────── │ Decoder │
  │ (reads  │ cross-   │ (writes │
  │  input) │ attention│ output) │
  └─────────┘          └─────────┘
  Good for: translation, summarization (input → output)

GPT-style (2018+) — Decoder-Only:
  ┌──────────────────┐
  │ Decoder-Only     │
  │ (reads + writes) │
  │ with causal mask │
  └──────────────────┘
  Good for: language modeling, text generation, chat

Key difference: decoder-only uses CAUSAL MASKING (from Lesson 5)
to ensure each token can only attend to previous tokens.
This single mechanism replaces the entire encoder.
""")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — TRANSFORMER BLOCK (the repeating unit)
# ═══════════════════════════════════════════════════════════════════════════
#
# A GPT-style transformer is just the same block repeated N times.
# Each block has:
#
#   x ──┐
#       ├──→ LayerNorm → MultiHeadAttention ──→ + (residual)
#   x ──┘                                       │
#       ┌────────────────────────────────────────┘
#       ├──→ LayerNorm → FFN ──→ + (residual)
#       └────────────────────────┘
#
# This is PRE-NORM (norm before attention/FFN), used in GPT-2, LLaMA.
# Post-norm (original Transformer) puts norm AFTER — less stable.
#
# The residual connections let gradients flow straight through,
# enabling training of very deep networks (96 layers in GPT-3).
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: TRANSFORMER BLOCK")
print("=" * 60)

torch.manual_seed(42)


class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention (from Lesson 5, assembled here).
    Each token attends only to itself and previous tokens.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scale = math.sqrt(self.d_head)
        scores = (q @ k.transpose(-2, -1)) / scale

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        out = attn_weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    This is a 2-layer MLP applied independently to each token position.
    FFN(x) = Linear2(GELU(Linear1(x)))

    d_ff is typically 4 * d_model (GPT-2 convention).
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """
    A single transformer decoder block with pre-norm.

    The data flow:
      x  →  x + Attention(LayerNorm(x))  →  x + FFN(LayerNorm(x))

    Pre-norm means we normalize BEFORE each sub-layer, not after.
    This is more stable for training deep models (GPT-2, LLaMA, etc.).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


d_model, n_heads, d_ff = 64, 4, 256
block = TransformerBlock(d_model, n_heads, d_ff)

x = torch.randn(2, 10, d_model)
y = block(x)

print(f"\nTransformerBlock(d_model={d_model}, n_heads={n_heads}, d_ff={d_ff})")
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Same shape? {x.shape == y.shape}  (residual connections preserve dimensions)")

block_params = sum(p.numel() for p in block.parameters())
print(f"Block parameters: {block_params:,}")
print(f"\nBlock breakdown:")
for name, param in block.named_parameters():
    print(f"  {name:30s} {str(param.shape):20s} {param.numel():>8,}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════════════════════
#
# Attention is permutation-invariant — "the cat sat" and "sat the cat"
# produce the same attention scores. The model has NO sense of order.
#
# Positional encoding injects position information so the model knows
# that token 0 comes before token 1, etc.
#
# Two approaches:
#   1. Sinusoidal PE (original Transformer) — fixed, deterministic
#   2. RoPE (Rotary Position Embeddings) — modern, used in LLaMA/Mistral
#
# Learned PE (GPT-2) is a third option: just an nn.Embedding for positions.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3: POSITIONAL ENCODING")
print("=" * 60)

torch.manual_seed(42)


# --- 3.1 Sinusoidal Positional Encoding (original Transformer) -----------

class SinusoidalPE(nn.Module):
    """
    Sinusoidal positional encoding from "Attention Is All You Need".

    For each position pos and dimension i:
      PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Properties:
    - Deterministic — no learned parameters
    - Theoretically supports any sequence length (extrapolation)
    - Each dimension oscillates at a different frequency
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


pe_layer = SinusoidalPE(d_model=64, max_len=512)
x = torch.randn(2, 10, 64)
x_pe = pe_layer(x)

print(f"\nSinusoidal PE")
print(f"Input shape: {x.shape}")
print(f"Output shape: {x_pe.shape}")
print(f"PE buffer shape: {pe_layer.pe.shape}")
print(f"Learned parameters: {sum(p.numel() for p in pe_layer.parameters())} (none — it's fixed)")

print(f"\nSample PE values for position 0: {pe_layer.pe[0, 0, :8].tolist()}")
print(f"Sample PE values for position 1: {pe_layer.pe[0, 1, :8].tolist()}")
print(f"Different positions → different patterns → model can distinguish order")


# --- 3.2 Rotary Position Embeddings (RoPE) — the modern approach --------

def precompute_rope_freqs(d_head: int, max_len: int = 2048, theta: float = 10000.0):
    """
    Precompute the rotation frequencies for RoPE.

    RoPE encodes position by ROTATING query/key vectors in 2D subspaces.
    Instead of adding a position vector, it multiplies by a rotation matrix.

    This has major advantages:
    - Relative position is encoded in the dot product (q_m · k_n depends on m-n)
    - No extra parameters
    - Better length extrapolation than learned PE
    - Used by LLaMA, Mistral, Qwen, and most modern LLMs
    """
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, freqs)
    cos_cached = torch.cos(angles)
    sin_cached = torch.sin(angles)
    return cos_cached, sin_cached


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embeddings to a tensor.
    x shape: (batch, n_heads, seq_len, d_head)

    The rotation pairs up consecutive dimensions:
      (x0, x1) → (x0*cos - x1*sin, x0*sin + x1*cos)

    This is a 2D rotation applied independently to each pair of dimensions.
    """
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]

    seq_len = x.size(2)
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    return rotated


d_head = 16
rope_cos, rope_sin = precompute_rope_freqs(d_head, max_len=128)

q = torch.randn(1, 4, 10, d_head)
k = torch.randn(1, 4, 10, d_head)

q_rotated = apply_rope(q, rope_cos, rope_sin)
k_rotated = apply_rope(k, rope_cos, rope_sin)

print(f"\nRoPE (Rotary Position Embeddings)")
print(f"d_head={d_head}, precomputed for max_len=128")
print(f"Frequency table shape: {rope_cos.shape}")
print(f"Q shape before RoPE: {q.shape}")
print(f"Q shape after RoPE:  {q_rotated.shape}  (same shape — rotation preserves norm)")

scores_no_rope = (q @ k.transpose(-2, -1)) / math.sqrt(d_head)
scores_with_rope = (q_rotated @ k_rotated.transpose(-2, -1)) / math.sqrt(d_head)

print(f"\nAttention scores WITHOUT RoPE (position-blind):")
print(f"  score[0,0,0,:5] = {scores_no_rope[0, 0, 0, :5].tolist()}")
print(f"Attention scores WITH RoPE (position-aware):")
print(f"  score[0,0,0,:5] = {scores_with_rope[0, 0, 0, :5].tolist()}")
print(f"RoPE makes the dot product depend on RELATIVE position (m - n)")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — FULL GPT MODEL
# ═══════════════════════════════════════════════════════════════════════════
#
# Now we stack everything together:
#   1. Token embedding: token_id → vector of size d_model
#   2. Positional encoding: add position information
#   3. N transformer blocks: the core computation
#   4. Final LayerNorm: stabilize outputs
#   5. LM Head: project from d_model → vocab_size (predict next token)
#
# The forward pass: tokens → logits
#   input_ids (B, T)
#     → token_embed(input_ids) + pos_embed  → (B, T, d_model)
#     → block_1 → block_2 → ... → block_N  → (B, T, d_model)
#     → LayerNorm                            → (B, T, d_model)
#     → lm_head                              → (B, T, vocab_size)
#
# Each position in the output is a probability distribution over the
# vocabulary — the model's prediction for what comes next.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 4: FULL GPT MODEL")
print("=" * 60)

torch.manual_seed(42)


class GPT(nn.Module):
    """
    A complete GPT-style decoder-only transformer.

    Architecture:
      Token Embedding + Positional Embedding
        ↓
      N × TransformerBlock(Attention + FFN + Residual + Norm)
        ↓
      Final LayerNorm
        ↓
      LM Head (Linear projection to vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following GPT-2 conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: token indices → logits over vocabulary.

        Args:
            input_ids: (batch, seq_len) tensor of token indices

        Returns:
            logits: (batch, seq_len, vocab_size) — raw scores for next token
        """
        B, T = input_ids.shape

        tok_emb = self.token_emb(input_ids)
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos_ids)
        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits


model = GPT(
    vocab_size=1000,
    d_model=64,
    n_heads=4,
    n_layers=4,
    d_ff=256,
    max_seq_len=128,
)

input_ids = torch.randint(0, 1000, (2, 20))
logits = model(input_ids)

print(f"\nGPT(vocab=1000, d_model=64, n_heads=4, n_layers=4, d_ff=256)")
print(f"Input shape:  {input_ids.shape}   (batch=2, seq_len=20)")
print(f"Output shape: {logits.shape}  (batch=2, seq_len=20, vocab=1000)")

print(f"\nLogits at position 0 (first 10 vocab entries):")
print(f"  {logits[0, 0, :10].tolist()}")
print(f"These are raw scores — apply softmax to get probabilities.")

probs = torch.softmax(logits[0, 0], dim=-1)
top5_probs, top5_ids = probs.topk(5)
print(f"\nTop 5 predictions for next token after position 0:")
for i, (prob, idx) in enumerate(zip(top5_probs, top5_ids)):
    print(f"  #{i+1}: token {idx.item():4d} with prob {prob.item():.4f}")
print(f"(Random weights → roughly uniform predictions — training changes this)")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — MODEL DIMENSIONS
# ═══════════════════════════════════════════════════════════════════════════
#
# The key hyperparameters that define a transformer:
#
#   d_model   — hidden dimension (width of the model)
#   n_heads   — number of attention heads
#   n_layers  — number of transformer blocks (depth)
#   d_ff      — hidden dimension of FFN (typically 4 * d_model)
#   d_head    — dimension per head = d_model / n_heads
#
# These directly determine the parameter count and computational cost.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 5: MODEL DIMENSIONS")
print("=" * 60)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_parameters_detailed(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
    """
    Count parameters analytically for a GPT model.

    Token embedding:    vocab_size * d_model
    Position embedding: max_seq_len * d_model

    Per block:
      QKV projection:  d_model * (3 * d_model) + 3 * d_model
      Output projection: d_model * d_model + d_model
      FFN layer 1:     d_model * d_ff + d_ff
      FFN layer 2:     d_ff * d_model + d_model
      LayerNorm × 2:   2 * (d_model + d_model) = 4 * d_model

    Final LayerNorm: 2 * d_model
    LM Head: d_model * vocab_size (no bias)
    """
    token_emb = vocab_size * d_model
    pos_emb = max_seq_len * d_model

    qkv = d_model * 3 * d_model + 3 * d_model
    out_proj = d_model * d_model + d_model
    ffn1 = d_model * d_ff + d_ff
    ffn2 = d_ff * d_model + d_model
    layer_norms = 4 * d_model
    per_block = qkv + out_proj + ffn1 + ffn2 + layer_norms

    final_ln = 2 * d_model
    lm_head = d_model * vocab_size

    total = token_emb + pos_emb + (n_layers * per_block) + final_ln + lm_head

    return {
        "token_embedding": token_emb,
        "position_embedding": pos_emb,
        "per_block": per_block,
        "per_block_x_n": n_layers * per_block,
        "final_layernorm": final_ln,
        "lm_head": lm_head,
        "total": total,
    }


print("\n--- Our toy model ---")
actual = count_parameters(model)
print(f"Actual parameter count: {actual:,}")

analytical = count_parameters_detailed(
    vocab_size=1000, d_model=64, n_heads=4, n_layers=4, d_ff=256, max_seq_len=128
)
print(f"Analytical count:       {analytical['total']:,}")
print(f"Match: {actual == analytical['total']}")

print(f"\nBreakdown:")
for key, val in analytical.items():
    if key != "total":
        print(f"  {key:25s}: {val:>10,}")
print(f"  {'TOTAL':25s}: {analytical['total']:>10,}")


# --- GPT-2 (124M) config ---
print("\n--- GPT-2 (124M) Configuration ---")
gpt2_config = {
    "vocab_size": 50257,
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "d_ff": 3072,       # 4 * 768
    "max_seq_len": 1024,
}

for key, val in gpt2_config.items():
    print(f"  {key:15s}: {val:>8,}")

gpt2_counts = count_parameters_detailed(**gpt2_config)
print(f"\nGPT-2 parameter breakdown:")
for key, val in gpt2_counts.items():
    if key != "total":
        pct = 100 * val / gpt2_counts["total"]
        print(f"  {key:25s}: {val:>12,}  ({pct:5.1f}%)")
print(f"  {'TOTAL':25s}: {gpt2_counts['total']:>12,}")
print(f"\nExpected: ~124M parameters")
print(f"Note: GPT-2 ties token embedding and LM head weights (weight tying),")
print(f"which would subtract {gpt2_config['vocab_size'] * gpt2_config['d_model']:,} params.")
print(f"With weight tying: {gpt2_counts['total'] - gpt2_config['vocab_size'] * gpt2_config['d_model']:,}")

gpt2_real = GPT(**gpt2_config)
gpt2_actual = count_parameters(gpt2_real)
print(f"\nActual PyTorch model parameter count: {gpt2_actual:,}")
print(f"In millions: {gpt2_actual / 1e6:.1f}M")

del gpt2_real


# --- Other GPT-2 variants ---
print("\n--- GPT-2 Model Family ---")
variants = [
    ("GPT-2 Small",  50257, 768,  12, 12, 3072,  1024),
    ("GPT-2 Medium", 50257, 1024, 16, 24, 4096,  1024),
    ("GPT-2 Large",  50257, 1280, 20, 36, 5120,  1024),
    ("GPT-2 XL",     50257, 1600, 25, 48, 6400,  1024),
]

print(f"{'Model':18s} {'d_model':>8s} {'n_heads':>8s} {'n_layers':>9s} {'d_ff':>6s} {'Params':>12s}")
print("-" * 65)
for name, vocab, dm, nh, nl, dff, msl in variants:
    counts = count_parameters_detailed(vocab, dm, nh, nl, dff, msl)
    print(f"{name:18s} {dm:>8d} {nh:>8d} {nl:>9d} {dff:>6d} {counts['total']:>12,}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — GENERATING TEXT
# ═══════════════════════════════════════════════════════════════════════════
#
# Generation is autoregressive: we feed tokens in, get logits out,
# pick the next token, append it, and repeat.
#
# Greedy decoding: always pick the token with the highest probability.
# (More advanced sampling — temperature, top-k, top-p — is in Lesson 10.)
#
# The loop:
#   1. Start with a prompt: [token_0, token_1, ...]
#   2. Forward pass → logits for every position
#   3. Take logits at the LAST position → next-token prediction
#   4. Pick the argmax (greedy) → new_token
#   5. Append new_token to the sequence
#   6. Repeat from step 2
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 6: GENERATING TEXT")
print("=" * 60)

torch.manual_seed(42)


@torch.no_grad()
def generate_greedy(
    model: GPT,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 20,
) -> torch.Tensor:
    """
    Greedy autoregressive text generation.

    At each step:
      1. Forward pass on the current sequence
      2. Take logits at the last position
      3. Argmax → next token
      4. Append and repeat
    """
    model.eval()
    generated = prompt_ids.clone()

    for step in range(max_new_tokens):
        logits = model(generated)
        next_logits = logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

    return generated


small_model = GPT(
    vocab_size=100,
    d_model=64,
    n_heads=4,
    n_layers=2,
    d_ff=256,
    max_seq_len=64,
)

prompt = torch.tensor([[1, 2, 3, 4, 5]])
print(f"\nPrompt token IDs:  {prompt[0].tolist()}")

generated = generate_greedy(small_model, prompt, max_new_tokens=15)
print(f"Generated sequence: {generated[0].tolist()}")
print(f"New tokens:         {generated[0, 5:].tolist()}")

print(f"""
Note: the output is random-looking because the model has RANDOM WEIGHTS.
After training on text data (Lesson 8), the same generation loop
produces coherent text. The architecture is the same — only the
weights change through training.

Generation step by step:
  Step 0: [1, 2, 3, 4, 5] → model → logits → argmax → token {generated[0, 5].item()}
  Step 1: [1, 2, 3, 4, 5, {generated[0, 5].item()}] → model → logits → argmax → token {generated[0, 6].item()}
  Step 2: [1, 2, 3, 4, 5, {generated[0, 5].item()}, {generated[0, 6].item()}] → ... → token {generated[0, 7].item()}
  ... and so on for each new token.
""")

print(f"Sequence length grows: {prompt.shape[1]} → {generated.shape[1]} tokens")
print(f"Forward passes needed: {generated.shape[1] - prompt.shape[1]}")
print(f"This is why generation is slow — O(n²) attention at each step!")
print(f"(KV caching, covered in Lesson 10, reduces redundant computation)")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
Exercise 1: Implement a Transformer Block
-------------------------------------------
Without looking at the code above, implement TransformerBlock from scratch:
  a) Create CausalSelfAttention with QKV projection and causal mask
  b) Create FeedForward with two linear layers and GELU
  c) Combine them with pre-norm (LayerNorm before each sub-layer)
  d) Add residual connections
  e) Verify: input shape == output shape for (batch=2, seq=10, d_model=64)

Exercise 2: Implement RoPE
----------------------------
Implement Rotary Position Embeddings from scratch:
  a) Write precompute_rope_freqs(d_head, max_len, theta=10000)
  b) Write apply_rope(x, cos, sin) that rotates pairs of dimensions
  c) Verify that the attention dot product changes with RoPE
  d) Test: for the same token at different positions, the rotated
     vectors should be different (check with torch.allclose)
  Bonus: Modify CausalSelfAttention to use RoPE instead of position embedding

Exercise 3: Build a Full GPT Model
-------------------------------------
Build a complete GPT class:
  a) Token embedding + position embedding
  b) Stack N=4 transformer blocks
  c) Final LayerNorm + LM head
  d) Forward pass: input_ids (B, T) → logits (B, T, vocab_size)
  e) Test with vocab_size=500, d_model=128, n_heads=8, n_layers=4, d_ff=512
  f) Verify the output shape is (batch, seq_len, vocab_size)

Exercise 4: Count Parameters to Match GPT-2 124M
---------------------------------------------------
  a) Using the analytical formula, compute the parameter count for:
     vocab_size=50257, d_model=768, n_heads=12, n_layers=12, d_ff=3072
  b) Break down: how many params in embeddings? In transformer blocks?
     In the LM head?
  c) What percentage of parameters are in the transformer blocks vs
     embeddings? (Hint: most params are in the blocks)
  d) If you double n_layers from 12 to 24, how many more parameters?
  e) If you double d_model from 768 to 1536, how many more parameters?
     (Hint: it's more than 2× — why?)

Exercise 5: Generate Text from Random Weights
-----------------------------------------------
  a) Create a GPT model with vocab_size=26 (one per letter), d_model=32,
     n_heads=4, n_layers=2, d_ff=128, max_seq_len=64
  b) Create a simple mapping: 0='a', 1='b', ..., 25='z'
  c) Start with prompt [0] (the letter 'a')
  d) Generate 50 tokens using greedy decoding
  e) Decode the generated token IDs back to letters
  f) The output will be gibberish — that's expected with random weights!
  g) Bonus: try generating from different prompts. Does the output change?
""")
