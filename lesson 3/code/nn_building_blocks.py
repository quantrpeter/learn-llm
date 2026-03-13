"""
============================================================================
 LESSON 3: Neural Network Building Blocks
============================================================================

 Goal: Understand every component that will later appear inside a
       transformer — so when you build one in Lesson 6, nothing is magic.

 Topics covered:
   1. Linear Layers        — weights, biases, y = Wx + b, shape transforms
   2. Activation Functions  — ReLU, GELU, SiLU/Swish, why non-linearity
   3. Layer Normalization   — LayerNorm from scratch, RMSNorm, pre/post-norm
   4. Dropout              — random zeroing, train vs eval mode
   5. Residual Connections  — skip connections, gradient flow across depth
   6. Weight Initialization — Xavier, Kaiming, impact on stability

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites: Lesson 2 (PyTorch tensors, autograd, nn.Module, training loop)
============================================================================
"""

import torch
import torch.nn as nn
import math

torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — LINEAR LAYERS (the fundamental building block)
# ═══════════════════════════════════════════════════════════════════════════
#
# nn.Linear computes y = xW^T + b. It's a matrix multiply + bias.
# In a transformer, LINEAR LAYERS are everywhere:
#   - Q, K, V projections in attention:  nn.Linear(d_model, d_model)
#   - Output projection after attention: nn.Linear(d_model, d_model)
#   - FFN up-projection:                 nn.Linear(d_model, 4*d_model)
#   - FFN down-projection:               nn.Linear(4*d_model, d_model)
#   - Final LM head:                     nn.Linear(d_model, vocab_size)
#
# A GPT-2 small (124M params) has ~50 nn.Linear layers.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: LINEAR LAYERS")
print("=" * 60)


# --- 1.1 Basic nn.Linear -------------------------------------------------

"""
WHY THIS MATTERS:
nn.Linear IS the weight matrix multiply from Lesson 1's "tiny forward pass."
Every projection in a transformer — Q, K, V, output, FFN up, FFN down, LM
head — is just an nn.Linear with different (in, out) sizes.
"""

linear = nn.Linear(in_features=8, out_features=4)

print(f"\nnn.Linear(8, 4)")
print(f"Weight shape: {linear.weight.shape}")   # (out, in) = (4, 8)
print(f"Bias shape:   {linear.bias.shape}")      # (out,) = (4,)

x = torch.randn(2, 8)   # batch of 2 vectors, each dim 8
y = linear(x)            # (2, 8) @ (8, 4) + (4,) → (2, 4)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")


# --- 1.2 Verifying y = xW^T + b manually ---------------------------------

y_manual = x @ linear.weight.T + linear.bias
print(f"\nManual y = xW^T + b matches nn.Linear: {torch.allclose(y, y_manual)}")


# --- 1.3 Shape transformations in a transformer --------------------------

"""
WHY THIS MATTERS:
Understanding how shapes flow through linear layers is critical.
In a transformer with d_model=768 and batch_size=4, seq_len=128:
  Input:                (4, 128, 768)
  After Q projection:   (4, 128, 768)  — same shape, different values
  After FFN up:         (4, 128, 3072) — expanded 4x
  After FFN down:       (4, 128, 768)  — compressed back
  After LM head:        (4, 128, 50257) — one logit per vocab token
"""

batch_size, seq_len, d_model = 4, 128, 768

q_proj = nn.Linear(d_model, d_model)       # attention Q projection
ffn_up = nn.Linear(d_model, 4 * d_model)   # FFN expansion
ffn_down = nn.Linear(4 * d_model, d_model)  # FFN compression
lm_head = nn.Linear(d_model, 50257)         # vocabulary projection

x = torch.randn(batch_size, seq_len, d_model)
print(f"\n--- Shape flow through transformer linear layers ---")
print(f"Input:         {x.shape}")

q = q_proj(x)
print(f"After Q proj:  {q.shape}")

up = ffn_up(x)
print(f"After FFN up:  {up.shape}")

down = ffn_down(up)
print(f"After FFN down:{down.shape}")

logits = lm_head(x)
print(f"After LM head: {logits.shape}")


# --- 1.4 Linear without bias ---------------------------------------------

linear_no_bias = nn.Linear(64, 64, bias=False)
print(f"\nLinear without bias — used in some LLM architectures (LLaMA)")
print(f"Has weight: {linear_no_bias.weight is not None}")
print(f"Has bias:   {linear_no_bias.bias is not None}")

total_params = sum(p.numel() for p in linear_no_bias.parameters())
print(f"Parameters:  {total_params:,} (no bias saves {64} params)")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — ACTIVATION FUNCTIONS (introducing non-linearity)
# ═══════════════════════════════════════════════════════════════════════════
#
# Without activation functions, stacking linear layers is pointless:
#   Linear2(Linear1(x)) = (W2 @ W1) @ x = W_combined @ x
# It collapses into a single linear transformation. Non-linearity breaks
# this, letting networks approximate ANY function.
#
# Modern LLMs use GELU (GPT-2, BERT) or SiLU/Swish (LLaMA, Mistral).
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: ACTIVATION FUNCTIONS")
print("=" * 60)

torch.manual_seed(42)


# --- 2.1 ReLU: the classic -----------------------------------------------

"""
WHY THIS MATTERS:
ReLU (Rectified Linear Unit) = max(0, x). Simple, fast, and was the default
for years. But it has a problem: negative inputs produce EXACTLY zero, which
means those neurons are "dead" — they stop learning entirely.
"""

x = torch.linspace(-3, 3, 13)
relu = nn.ReLU()
print(f"\n--- ReLU: max(0, x) ---")
print(f"x:       {[f'{v:.1f}' for v in x.tolist()]}")
print(f"ReLU(x): {[f'{v:.1f}' for v in relu(x).tolist()]}")


# --- 2.2 GELU: the LLM standard ------------------------------------------

"""
WHY THIS MATTERS:
GELU (Gaussian Error Linear Unit) is used in GPT-2, GPT-3, BERT, and most
modern LLMs. Unlike ReLU's hard cutoff at 0, GELU is smooth — it lets small
negative values through with small probability, preventing dead neurons.

GELU(x) = x * Phi(x), where Phi is the standard normal CDF.
Approximation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
"""

gelu = nn.GELU()
print(f"\n--- GELU: x * Phi(x) ---")
print(f"x:       {[f'{v:.1f}' for v in x.tolist()]}")
print(f"GELU(x): {[f'{v:.4f}' for v in gelu(x).tolist()]}")


# --- 2.3 Implementing GELU from scratch ----------------------------------

def gelu_manual(x):
    """GELU using the tanh approximation (same as GPT-2's implementation)."""
    return 0.5 * x * (1.0 + torch.tanh(
        math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)
    ))


test_x = torch.randn(5)
pytorch_gelu = gelu(test_x)
manual_gelu = gelu_manual(test_x)
print(f"\n--- GELU from scratch ---")
print(f"Input:         {test_x.tolist()}")
print(f"PyTorch GELU:  {[f'{v:.6f}' for v in pytorch_gelu.tolist()]}")
print(f"Manual GELU:   {[f'{v:.6f}' for v in manual_gelu.tolist()]}")
print(f"Match (atol=1e-3): {torch.allclose(pytorch_gelu, manual_gelu, atol=1e-3)}")
print(f"  (small diff: PyTorch uses exact erf, we use tanh approximation)")


# --- 2.4 SiLU / Swish: used in LLaMA and Mistral -------------------------

"""
WHY THIS MATTERS:
SiLU(x) = x * sigmoid(x), also called Swish. It's used in LLaMA, Mistral,
and other recent LLMs. Like GELU, it's smooth and allows small negative
values through. LLaMA specifically uses SiLU in its "SwiGLU" FFN variant.
"""

silu = nn.SiLU()
print(f"\n--- SiLU / Swish: x * sigmoid(x) ---")
print(f"x:       {[f'{v:.1f}' for v in x.tolist()]}")
print(f"SiLU(x): {[f'{v:.4f}' for v in silu(x).tolist()]}")


# --- 2.5 Why non-linearity matters (demonstration) -----------------------

print(f"\n--- Why non-linearity matters ---")

torch.manual_seed(42)
linear1 = nn.Linear(8, 8, bias=False)
linear2 = nn.Linear(8, 8, bias=False)

x = torch.randn(1, 8)

without_activation = linear2(linear1(x))

collapsed_weight = linear2.weight @ linear1.weight
collapsed_output = x @ collapsed_weight.T

print(f"Two linears without activation — collapses to one:")
print(f"  Linear2(Linear1(x)):    {[f'{v:.4f}' for v in without_activation[0].tolist()]}")
print(f"  x @ (W2 @ W1)^T:       {[f'{v:.4f}' for v in collapsed_output[0].tolist()]}")
print(f"  Identical? {torch.allclose(without_activation, collapsed_output, atol=1e-5)}")

with_activation = linear2(gelu(linear1(x)))
print(f"  With GELU in between:   {[f'{v:.4f}' for v in with_activation[0].tolist()]}")
print(f"  Different from collapsed? {not torch.allclose(with_activation, collapsed_output, atol=1e-2)}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — LAYER NORMALIZATION (stabilizing activations)
# ═══════════════════════════════════════════════════════════════════════════
#
# As tensors flow through dozens of layers, their values can drift — getting
# huge or tiny. LayerNorm recenters each vector to mean=0, variance=1, then
# applies a learned scale (gamma) and shift (beta).
#
# Why LayerNorm and not BatchNorm?
#   - BatchNorm normalizes across the BATCH dimension — problematic when
#     batch sizes vary or during autoregressive generation (batch=1).
#   - LayerNorm normalizes across the FEATURE dimension of each individual
#     sample — no batch dependency, perfect for transformers.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3: LAYER NORMALIZATION")
print("=" * 60)

torch.manual_seed(42)


# --- 3.1 The problem: activation drift -----------------------------------

"""
WHY THIS MATTERS:
Without normalization, deep networks suffer from "internal covariate shift" —
activations drift as they pass through layers, making training unstable.
LayerNorm fixes this by normalizing every hidden state to mean=0, var=1.
"""

x = torch.randn(2, 4) * 10 + 5   # deliberately shifted/scaled
print(f"\nBefore normalization:")
print(f"  x[0] mean={x[0].mean():.2f}, std={x[0].std():.2f}")
print(f"  x[1] mean={x[1].mean():.2f}, std={x[1].std():.2f}")

ln = nn.LayerNorm(4)
x_normed = ln(x)
print(f"After LayerNorm:")
print(f"  x[0] mean={x_normed[0].mean():.4f}, std={x_normed[0].std(correction=0):.4f}")
print(f"  x[1] mean={x_normed[1].mean():.4f}, std={x_normed[1].std(correction=0):.4f}")


# --- 3.2 Implementing LayerNorm from scratch ------------------------------

class ManualLayerNorm(nn.Module):
    """
    LayerNorm from scratch.

    For each sample independently:
      1. Compute mean and variance across features
      2. Normalize: x_hat = (x - mean) / sqrt(var + eps)
      3. Scale and shift: y = gamma * x_hat + beta

    gamma (weight) and beta (bias) are learnable parameters.
    """

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, correction=0)
        x_hat = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_hat + self.bias


torch.manual_seed(42)
x = torch.randn(2, 5, 64)   # (batch, seq_len, d_model)

pytorch_ln = nn.LayerNorm(64)
manual_ln = ManualLayerNorm(64)

manual_ln.weight.data.copy_(pytorch_ln.weight.data)
manual_ln.bias.data.copy_(pytorch_ln.bias.data)

out_pytorch = pytorch_ln(x)
out_manual = manual_ln(x)

print(f"\n--- LayerNorm from scratch ---")
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out_manual.shape}")
print(f"Matches nn.LayerNorm: {torch.allclose(out_pytorch, out_manual, atol=1e-5)}")


# --- 3.3 RMSNorm (Root Mean Square Normalization) ------------------------

"""
WHY THIS MATTERS:
RMSNorm is used in LLaMA, Mistral, and most modern LLMs. It's simpler and
~10-15% faster than LayerNorm because it skips the mean subtraction step.

RMSNorm(x) = x / RMS(x) * gamma
where RMS(x) = sqrt(mean(x^2) + eps)

The insight: re-centering (subtracting mean) isn't necessary — just
re-scaling is enough for training stability.
"""

class RMSNorm(nn.Module):
    """RMSNorm as used in LLaMA / Mistral."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


torch.manual_seed(42)
x = torch.randn(2, 5, 64)
rms_norm = RMSNorm(64)
out_rms = rms_norm(x)

print(f"\n--- RMSNorm (LLaMA-style) ---")
print(f"Input shape:  {x.shape}")
print(f"Output shape: {out_rms.shape}")
print(f"Before RMSNorm — mean: {x[0, 0, :5].tolist()}")
print(f"After RMSNorm  — mean: {out_rms[0, 0, :5].tolist()}")
print(f"RMSNorm params: {sum(p.numel() for p in rms_norm.parameters())} (no bias!)")
print(f"LayerNorm params: {sum(p.numel() for p in pytorch_ln.parameters())} (weight + bias)")


# --- 3.4 Pre-Norm vs Post-Norm -------------------------------------------

"""
WHY THIS MATTERS:
Where you place LayerNorm changes training dynamics significantly.

Post-Norm (original transformer):  x + Sublayer(LayerNorm(x))  -- NO
  Actually: LayerNorm(x + Sublayer(x))
  Harder to train, but slightly better final performance.

Pre-Norm (GPT-2, LLaMA, most modern LLMs):  x + Sublayer(LayerNorm(x))
  Normalize BEFORE the sublayer. Much more stable training.
  Almost all modern LLMs use pre-norm.
"""

class PostNormBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(x + self.ffn(x))   # normalize AFTER residual


class PreNormBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return x + self.ffn(self.norm(x))   # normalize BEFORE sublayer


print(f"\n--- Pre-Norm vs Post-Norm ---")
print(f"Post-Norm: LayerNorm(x + Sublayer(x))  — original transformer")
print(f"Pre-Norm:  x + Sublayer(LayerNorm(x))  — GPT-2, LLaMA, most modern LLMs")

torch.manual_seed(42)
x = torch.randn(1, 1, 64)
pre = PreNormBlock(64)
post = PostNormBlock(64)
print(f"Pre-Norm output range:  [{pre(x).min():.3f}, {pre(x).max():.3f}]")
print(f"Post-Norm output range: [{post(x).min():.3f}, {post(x).max():.3f}]")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — DROPOUT (regularization by random zeroing)
# ═══════════════════════════════════════════════════════════════════════════
#
# Dropout randomly zeros elements during training with probability p,
# then scales survivors by 1/(1-p) to keep expected values the same.
# At inference time, dropout is disabled — all neurons are active.
#
# In transformers, dropout is applied:
#   - After attention weights (attention dropout)
#   - After the FFN sublayer (residual dropout)
#   - On embeddings (embedding dropout, in GPT-2)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 4: DROPOUT")
print("=" * 60)

torch.manual_seed(42)


# --- 4.1 How dropout works -----------------------------------------------

"""
WHY THIS MATTERS:
Dropout prevents the network from relying too heavily on any single neuron.
By randomly disabling neurons during training, it forces the network to
learn redundant representations — a form of ensemble learning.

GPT-2 uses dropout rate 0.1 on attention and residuals.
GPT-3 uses 0.1 as well. Some very large models use 0.0 (no dropout).
"""

x = torch.ones(1, 10)
dropout = nn.Dropout(p=0.3)

dropout.train()
print(f"\n--- Dropout in training mode (p=0.3) ---")
for i in range(3):
    out = dropout(x)
    zeros = (out == 0).sum().item()
    print(f"  Trial {i+1}: {out[0].tolist()}  (zeros: {zeros}/10)")
    print(f"           Non-zero values scaled by 1/(1-0.3) = {1/0.7:.4f}")


# --- 4.2 Training vs eval mode -------------------------------------------

"""
WHY THIS MATTERS:
This is a common bug — forgetting to call model.eval() during inference.
If you generate text with dropout still active, you'll get noisy garbage.
Always: model.train() for training, model.eval() for inference/generation.
"""

print(f"\n--- Training vs Eval mode ---")
x = torch.ones(1, 8)

dropout.train()
out_train = dropout(x)
print(f"Train mode: {out_train[0].tolist()}")

dropout.eval()
out_eval = dropout(x)
print(f"Eval mode:  {out_eval[0].tolist()}")
print(f"Eval output == input: {torch.equal(out_eval, x)}")


# --- 4.3 Dropout in a real model -----------------------------------------

class FFNWithDropout(nn.Module):
    """Feed-forward network with dropout, as used in GPT-2."""

    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


torch.manual_seed(42)
ffn = FFNWithDropout(d_model=64, d_ff=256, dropout_rate=0.1)
x = torch.randn(2, 10, 64)

ffn.train()
out_train1 = ffn(x)
out_train2 = ffn(x)

ffn.eval()
out_eval1 = ffn(x)
out_eval2 = ffn(x)

print(f"\n--- Dropout in a real model ---")
print(f"Train: two forward passes differ: {not torch.equal(out_train1, out_train2)}")
print(f"Eval:  two forward passes identical: {torch.equal(out_eval1, out_eval2)}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — RESIDUAL CONNECTIONS (skip connections)
# ═══════════════════════════════════════════════════════════════════════════
#
# The single most important architectural idea for training deep networks.
# Instead of:  y = F(x)
# We compute:  y = x + F(x)
#
# This "skip connection" means the gradient always has a direct path back
# through the addition — it doesn't have to survive multiplying through
# dozens of weight matrices. Without residuals, networks deeper than ~10
# layers are essentially untrainable.
#
# Every transformer block uses residual connections:
#   x = x + Attention(LayerNorm(x))
#   x = x + FFN(LayerNorm(x))
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 5: RESIDUAL CONNECTIONS")
print("=" * 60)

torch.manual_seed(42)


# --- 5.1 Basic residual connection ----------------------------------------

"""
WHY THIS MATTERS:
Without residual connections, transformers simply don't work. The original
ResNet paper (2015) showed that residuals enable training of 100+ layer
networks. Transformers like GPT-3 have 96 layers — impossible without
skip connections.
"""

class PlainBlock(nn.Module):
    """No residual — output = F(x)."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """With residual — output = x + F(x)."""
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, x):
        return x + self.net(x)


print(f"\n--- Residual vs Plain ---")
x = torch.randn(1, 32)

plain = PlainBlock(32)
residual = ResidualBlock(32)

print(f"Input norm:           {x.norm():.4f}")
print(f"PlainBlock output:    {plain(x).norm():.4f}")
print(f"ResidualBlock output: {residual(x).norm():.4f}")


# --- 5.2 Gradient flow comparison: 20+ layers ----------------------------

"""
WHY THIS MATTERS:
This is the key experiment. Watch how gradients vanish in a plain network
but stay healthy with residual connections. In a real LLM, this is the
difference between a model that learns and one that's dead.
"""

def measure_gradient_flow(block_class, depth, dim=32):
    """Stack `depth` blocks and measure gradient magnitude at each layer."""
    torch.manual_seed(42)
    blocks = nn.ModuleList([block_class(dim) for _ in range(depth)])

    x = torch.randn(1, dim, requires_grad=True)
    h = x
    for block in blocks:
        h = block(h)

    loss = h.sum()
    loss.backward()

    grad_norms = []
    for i, block in enumerate(blocks):
        total_grad = 0.0
        for p in block.parameters():
            if p.grad is not None:
                total_grad += p.grad.norm().item()
        grad_norms.append(total_grad)

    return grad_norms


depth = 24
plain_grads = measure_gradient_flow(PlainBlock, depth)
residual_grads = measure_gradient_flow(ResidualBlock, depth)

print(f"\n--- Gradient flow across {depth} layers ---")
print(f"{'Layer':>6s}  {'Plain':>12s}  {'Residual':>12s}  {'Ratio':>10s}")
print(f"{'─' * 6}  {'─' * 12}  {'─' * 12}  {'─' * 10}")
for i in range(0, depth, 3):
    p = plain_grads[i]
    r = residual_grads[i]
    ratio = r / p if p > 1e-15 else float('inf')
    print(f"{i+1:>6d}  {p:>12.6f}  {r:>12.6f}  {ratio:>10.1f}x")

print(f"\nFirst layer gradient:")
print(f"  Plain:    {plain_grads[0]:.8f}")
print(f"  Residual: {residual_grads[0]:.8f}")
print(f"Last layer gradient:")
print(f"  Plain:    {plain_grads[-1]:.8f}")
print(f"  Residual: {residual_grads[-1]:.8f}")
print(f"Ratio (first/last):")
if plain_grads[-1] > 1e-15:
    print(f"  Plain:    {plain_grads[0] / plain_grads[-1]:.1f}x difference")
else:
    print(f"  Plain:    gradient vanished to ~0")
if residual_grads[-1] > 1e-15:
    print(f"  Residual: {residual_grads[0] / residual_grads[-1]:.1f}x difference")
else:
    print(f"  Residual: gradient vanished to ~0")


# --- 5.3 Transformer-style residual block --------------------------------

class TransformerFFN(nn.Module):
    """Pre-norm FFN with residual, exactly as in GPT-2 / LLaMA."""

    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.drop(self.fc2(self.act(self.fc1(self.norm(x)))))


print(f"\n--- Transformer-style FFN block ---")
torch.manual_seed(42)
ffn_block = TransformerFFN(d_model=64)
x = torch.randn(2, 10, 64)
y = ffn_block(x)
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")
print(f"Parameters: {sum(p.numel() for p in ffn_block.parameters()):,}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — WEIGHT INITIALIZATION (how you start matters)
# ═══════════════════════════════════════════════════════════════════════════
#
# If weights start too large → activations explode → gradients explode.
# If weights start too small → activations shrink to zero → gradients vanish.
# Good initialization keeps activations in a healthy range across all layers.
#
# Two main strategies:
#   Xavier/Glorot — balances input and output dimensions
#   Kaiming/He   — designed for ReLU-family activations
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 6: WEIGHT INITIALIZATION")
print("=" * 60)

torch.manual_seed(42)


# --- 6.1 The problem: bad initialization ---------------------------------

"""
WHY THIS MATTERS:
GPT-2 uses a special scaled initialization: weights for residual layers
are scaled by 1/sqrt(2*n_layers). Without this, the variance of the
residual stream grows with depth, causing training instability.
"""

print(f"\n--- Effect of initialization on activations ---")

x = torch.randn(1, 512)

# Too large: stddev=2.0
torch.manual_seed(42)
large_net = nn.Sequential(*[nn.Linear(512, 512) for _ in range(10)])
for m in large_net:
    nn.init.normal_(m.weight, std=2.0)
    nn.init.zeros_(m.bias)

h = x
for i, layer in enumerate(large_net):
    h = layer(h)
    if i % 3 == 0:
        print(f"  Layer {i+1} (std=2.0): activation norm = {h.norm():.2e}")

# Too small: stddev=0.001
torch.manual_seed(42)
small_net = nn.Sequential(*[nn.Linear(512, 512) for _ in range(10)])
for m in small_net:
    nn.init.normal_(m.weight, std=0.001)
    nn.init.zeros_(m.bias)

h = x
for i, layer in enumerate(small_net):
    h = layer(h)
    if i % 3 == 0:
        print(f"  Layer {i+1} (std=0.001): activation norm = {h.norm():.2e}")


# --- 6.2 Xavier / Glorot initialization ----------------------------------

"""
WHY THIS MATTERS:
Xavier init sets weight std = sqrt(2 / (fan_in + fan_out)).
This keeps the variance of activations roughly constant across layers.
Used in the original transformer (Vaswani et al., 2017).
"""

print(f"\n--- Xavier / Glorot initialization ---")
torch.manual_seed(42)
xavier_net = nn.Sequential(*[nn.Linear(512, 512) for _ in range(10)])
for m in xavier_net:
    nn.init.xavier_normal_(m.weight)
    nn.init.zeros_(m.bias)

h = x
for i, layer in enumerate(xavier_net):
    h = layer(h)
    if i % 3 == 0:
        print(f"  Layer {i+1} (Xavier): activation norm = {h.norm():.4f}")


# --- 6.3 Kaiming / He initialization -------------------------------------

"""
WHY THIS MATTERS:
Kaiming init accounts for ReLU/GELU zeroing out ~half of inputs.
std = sqrt(2 / fan_in). Used when your network has ReLU-family activations.
"""

print(f"\n--- Kaiming / He initialization ---")
torch.manual_seed(42)
kaiming_net = nn.Sequential()
for i in range(10):
    kaiming_net.add_module(f"linear_{i}", nn.Linear(512, 512))
    kaiming_net.add_module(f"relu_{i}", nn.ReLU())

for m in kaiming_net:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)

h = x
layer_idx = 0
for m in kaiming_net:
    h = m(h)
    if isinstance(m, nn.ReLU):
        if layer_idx % 3 == 0:
            print(f"  After layer {layer_idx+1} ReLU (Kaiming): activation norm = {h.norm():.4f}")
        layer_idx += 1


# --- 6.4 GPT-2 style initialization with residual scaling ----------------

"""
WHY THIS MATTERS:
GPT-2 scales the output projection of each residual block by 1/sqrt(2*N)
where N is the number of layers. This prevents the residual stream variance
from growing linearly with depth.
"""

print(f"\n--- GPT-2 residual scaling ---")
n_layers = 12

class GPT2Block(nn.Module):
    def __init__(self, d_model, n_layers):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)

        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02 / math.sqrt(2 * n_layers))

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


torch.manual_seed(42)
gpt2_blocks = nn.ModuleList([GPT2Block(256, n_layers) for _ in range(n_layers)])
x = torch.randn(1, 10, 256)
h = x
for i, block in enumerate(gpt2_blocks):
    h = block(h)

print(f"Input norm:  {x.norm():.4f}")
print(f"After {n_layers} GPT-2 blocks: {h.norm():.4f}")
print(f"Ratio: {h.norm() / x.norm():.2f}x (should be ~1-3x, not exploding)")

# Compare with naive init
torch.manual_seed(42)
class NaiveBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


naive_blocks = nn.ModuleList([NaiveBlock(256) for _ in range(n_layers)])
h_naive = x
for block in naive_blocks:
    h_naive = block(h_naive)

print(f"After {n_layers} naive blocks:  {h_naive.norm():.4f}")
print(f"GPT-2 init is {h_naive.norm() / h.norm():.1f}x more controlled")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
Exercise 1: LayerNorm from Scratch
------------------------------------
Implement LayerNorm without looking at the ManualLayerNorm class above.
  a) Write a function layer_norm(x, weight, bias, eps=1e-5) that:
     - Computes mean and variance along the last dimension
     - Normalizes: x_hat = (x - mean) / sqrt(var + eps)
     - Returns weight * x_hat + bias
  b) Test on x = torch.randn(2, 5, 64) with weight=ones, bias=zeros
  c) Verify your output matches nn.LayerNorm(64) to at least 5 decimal places
  d) Check: output mean should be ~0, std should be ~1 for each sample

Exercise 2: RMSNorm from Scratch
-----------------------------------
Implement RMSNorm without looking at the class above.
  a) Write rms_norm(x, weight, eps=1e-6) that:
     - Computes RMS = sqrt(mean(x^2) + eps)
     - Returns (x / RMS) * weight
  b) Test on the same input as Exercise 1
  c) Compare the number of operations vs LayerNorm (no mean subtraction!)
  d) Time both on a large tensor (1000, 128, 768) — how much faster is RMSNorm?
     Hint: use torch.cuda.Event or time.perf_counter

Exercise 3: FFN with Residual Connection
------------------------------------------
Build a complete transformer FFN block (nn.Module) with:
  - Pre-norm (LayerNorm before the sublayer)
  - Two linear layers: d_model -> 4*d_model -> d_model
  - GELU activation between them
  - Dropout after the second linear layer
  - Residual connection around the whole sublayer
Test with d_model=64, dropout=0.1:
  a) Verify output shape matches input shape
  b) Verify that in eval mode, output != input (the FFN changes values)
  c) Verify that output is CLOSER to input than a plain FFN (the residual
     keeps the original information flowing)

Exercise 4: Gradient Flow Experiment
---------------------------------------
Extend the gradient flow comparison from Part 5:
  a) Test with depths: 8, 16, 32, 48 layers
  b) For each depth, record the ratio: grad_norm[first_layer] / grad_norm[last_layer]
  c) Show that for plain networks, this ratio grows exponentially with depth
  d) Show that for residual networks, this ratio stays bounded
  e) Bonus: add LayerNorm to the residual blocks — does it help further?

Exercise 5: Initialization Experiment
----------------------------------------
Create a deep network (20 layers, width 512) and test three init strategies:
  a) Default PyTorch init (nn.Linear default)
  b) Xavier/Glorot normal
  c) Kaiming normal + GPT-2 residual scaling
For each, pass a random input through and record:
  - Activation norm at layers 1, 5, 10, 15, 20
  - Which strategy keeps norms most stable?
  - Train each on a simple regression task for 100 steps — which converges fastest?
""")
