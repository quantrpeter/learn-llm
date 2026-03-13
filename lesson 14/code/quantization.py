"""
============================================================================
 LESSON 14: Quantization & Inference Optimization
============================================================================

 Goal: Make your model smaller and faster for real-world deployment.
       Understand weight quantization, post-training compression, and
       inference tricks that enable running LLMs on consumer hardware.

 Topics covered:
   1. Why Quantize        — FP32→FP16→INT8→INT4 memory savings
   2. Weight Quantization — Per-tensor INT8 from scratch (scale + zero_point)
   3. Per-Channel Quant.  — Better accuracy than per-tensor
   4. GPTQ Concept        — Post-training quantization with calibration data
   5. Speculative Decoding— Draft model + verify for faster generation
   6. KV Cache Opt.       — GQA vs MQA vs MHA memory comparison

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites:
   - Lesson 2 (PyTorch fundamentals, tensors, dtypes)
   - Lesson 5 (attention mechanism, KV cache basics)
   - Lesson 10 (text generation, decoding strategies)
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — WHY QUANTIZE?
# ═══════════════════════════════════════════════════════════════════════════
#
# A 7-billion parameter model in float32 needs 28 GB of memory — just for
# the weights. That doesn't fit on most consumer GPUs. Quantization shrinks
# models by using fewer bits per weight:
#
#   FP32 (32 bits) → FP16 (16 bits) → INT8 (8 bits) → INT4 (4 bits)
#
# Each step roughly halves the memory. A 7B model at INT4 fits in ~3.5 GB —
# that runs on a laptop.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: WHY QUANTIZE?")
print("=" * 60)


# --- 1.1 Memory math for a 7B parameter model ----------------------------

"""
WHY THIS MATTERS:
Every parameter is a number stored in memory. The number of BITS per
parameter directly determines how much RAM/VRAM you need. Quantization
trades precision for memory — and in practice the quality loss is tiny.
"""

model_params = 7e9  # 7 billion

dtypes = {
    "FP32": 4,   # 32 bits = 4 bytes
    "FP16": 2,   # 16 bits = 2 bytes
    "INT8": 1,   # 8 bits  = 1 byte
    "INT4": 0.5, # 4 bits  = 0.5 bytes
}

print(f"\n--- Memory Requirements for a 7B Parameter Model ---")
print(f"  {'Dtype':>6} | {'Bits':>4} | {'Bytes/Param':>11} | {'Total Memory':>12} | {'vs FP32':>8}")
print(f"  {'-'*6}-+-{'-'*4}-+-{'-'*11}-+-{'-'*12}-+-{'-'*8}")
for dtype_name, bytes_per_param in dtypes.items():
    total_gb = model_params * bytes_per_param / 1e9
    ratio = bytes_per_param / 4.0
    print(f"  {dtype_name:>6} | {int(bytes_per_param * 8):>4} | {bytes_per_param:>11} | {total_gb:>9.1f} GB | {ratio:>7.1%}")

print(f"\n  FP32 → INT4 = 8x memory reduction: 28.0 GB → 3.5 GB")
print(f"  A 7B INT4 model fits on a 6GB GPU or even runs on a laptop CPU!")


# --- 1.2 What each precision level looks like ----------------------------

"""
WHY THIS MATTERS:
Quantization introduces rounding error. FP32 has ~7 decimal digits of
precision, FP16 has ~3.3, and INT8 maps 256 discrete levels across the
entire weight range. The question is: how much error can the model
tolerate before output quality drops?
"""

print(f"\n--- Precision Comparison ---")
torch.manual_seed(42)
weights = torch.randn(8)  # sample weights

print(f"\n  Original FP32:  {weights.tolist()}")
print(f"  As FP16:        {weights.to(torch.float16).tolist()}")
print(f"  As BFloat16:    {weights.to(torch.bfloat16).tolist()}")

fp16_err = (weights - weights.to(torch.float16).to(torch.float32)).abs().max()
bf16_err = (weights - weights.to(torch.bfloat16).to(torch.float32)).abs().max()
print(f"\n  Max FP16 error:    {fp16_err.item():.8f}")
print(f"  Max BFloat16 error:{bf16_err.item():.8f}")
print(f"  (FP16 has higher precision, BFloat16 has wider range)")


# --- 1.3 The quantization quality spectrum --------------------------------

print(f"\n--- Quantization Quality Spectrum ---")
quality_data = [
    ("FP32",  "Baseline",            "100%",   "28.0 GB"),
    ("FP16",  "Negligible loss",     "~100%",  "14.0 GB"),
    ("INT8",  "Tiny loss (<0.1 ppl)","~99.5%", " 7.0 GB"),
    ("INT4",  "Small loss (<1 ppl)", "~98%",   " 3.5 GB"),
    ("INT3",  "Noticeable loss",     "~95%",   " 2.6 GB"),
    ("INT2",  "Significant loss",    "~85%",   " 1.8 GB"),
]
print(f"  {'Dtype':>6} | {'Quality Impact':>22} | {'Relative':>8} | {'7B Size':>8}")
print(f"  {'-'*6}-+-{'-'*22}-+-{'-'*8}-+-{'-'*8}")
for dtype_name, impact, quality, size in quality_data:
    print(f"  {dtype_name:>6} | {impact:>22} | {quality:>8} | {size:>8}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — WEIGHT QUANTIZATION (Per-Tensor INT8 From Scratch)
# ═══════════════════════════════════════════════════════════════════════════
#
# The simplest quantization: map the entire weight tensor's floating-point
# values to 256 integer levels (INT8 range: -128 to 127).
#
#   quantized = round((float_val - zero_point) / scale)
#   dequantized = quantized * scale + zero_point
#
# scale and zero_point are chosen to map [min, max] of the tensor to
# [-128, 127].
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 2: WEIGHT QUANTIZATION (Per-Tensor INT8)")
print("=" * 60)

torch.manual_seed(42)

"""
WHY THIS MATTERS:
This is the foundation of all quantization methods. GPTQ, AWQ, and
GGUF all build on this basic idea — they just choose scale and
zero_point more cleverly. Understanding naive quantization first
makes the advanced methods easy to grasp.
"""


# --- 2.1 Implementing INT8 quantization from scratch ----------------------

def quantize_tensor_int8(tensor: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """
    Quantize a float tensor to INT8 using affine (asymmetric) quantization.

    Maps [tensor.min(), tensor.max()] → [-128, 127].

    Returns:
        quantized: INT8 tensor
        scale: float — step size between quantization levels
        zero_point: float — the float value that maps to integer 0
    """
    t_min = tensor.min().item()
    t_max = tensor.max().item()

    scale = (t_max - t_min) / 255.0  # 256 levels: -128 to 127
    zero_point = t_min - (-128) * scale

    quantized = torch.clamp(
        torch.round((tensor - zero_point) / scale),
        -128, 127
    ).to(torch.int8)

    return quantized, scale, zero_point


def dequantize_tensor_int8(quantized: torch.Tensor, scale: float, zero_point: float) -> torch.Tensor:
    """Convert INT8 tensor back to float using scale and zero_point."""
    return quantized.float() * scale + zero_point


# --- 2.2 Quantize a weight matrix and measure error ----------------------

print(f"\n--- Quantizing a Weight Matrix ---")
weight = torch.randn(4, 8)  # simulate a small weight tensor

quantized, scale, zero_point = quantize_tensor_int8(weight)
dequantized = dequantize_tensor_int8(quantized, scale, zero_point)

print(f"  Original shape: {weight.shape}")
print(f"  Original dtype: {weight.dtype}")
print(f"  Quantized dtype: {quantized.dtype}")
print(f"  Scale: {scale:.6f}")
print(f"  Zero point: {zero_point:.6f}")
print(f"\n  Original weights (first row):")
print(f"    {weight[0].tolist()}")
print(f"  Quantized values (first row):")
print(f"    {quantized[0].tolist()}")
print(f"  Dequantized (first row):")
print(f"    {dequantized[0].tolist()}")


# --- 2.3 Measuring quantization error ------------------------------------

"""
WHY THIS MATTERS:
Quantization error is the difference between the original float weights
and the dequantized (reconstructed) weights. If this error is large,
the model's outputs will be wrong. Good quantization methods minimize
this error — especially on the weights that matter most.
"""

error = (weight - dequantized).abs()
mse = ((weight - dequantized) ** 2).mean()
max_err = error.max()
relative_err = error.mean() / weight.abs().mean()

print(f"\n--- Quantization Error Analysis ---")
print(f"  Mean Squared Error (MSE):   {mse.item():.8f}")
print(f"  Max Absolute Error:         {max_err.item():.6f}")
print(f"  Mean Relative Error:        {relative_err.item():.4%}")

print(f"\n  Memory savings:")
original_bytes = weight.numel() * 4  # float32
quantized_bytes = quantized.numel() * 1 + 8  # int8 + scale/zp overhead
print(f"    Original:   {original_bytes} bytes (float32)")
print(f"    Quantized:  {quantized_bytes} bytes (int8 + scale/zp)")
print(f"    Compression: {original_bytes / quantized_bytes:.1f}x")


# --- 2.4 Effect on matrix multiply output --------------------------------

print(f"\n--- Effect on Matrix Multiply ---")
x = torch.randn(2, 8)

y_original = x @ weight.T
y_quantized = x @ dequantized.T

output_error = (y_original - y_quantized).abs()
print(f"  Input shape:    {x.shape}")
print(f"  Output (FP32):  {y_original[0].tolist()}")
print(f"  Output (INT8):  {y_quantized[0].tolist()}")
print(f"  Output max err: {output_error.max().item():.6f}")
print(f"  Output MSE:     {((y_original - y_quantized) ** 2).mean().item():.8f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — PER-CHANNEL QUANTIZATION
# ═══════════════════════════════════════════════════════════════════════════
#
# Per-tensor quantization uses ONE scale/zero_point for the entire matrix.
# If some rows have small weights and others have large weights, the shared
# scale wastes bits on the small rows.
#
# Per-channel quantization: compute separate scale/zero_point for each
# OUTPUT CHANNEL (row). This is much more accurate.
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 3: PER-CHANNEL QUANTIZATION")
print("=" * 60)

torch.manual_seed(42)

"""
WHY THIS MATTERS:
Real weight matrices have different value ranges per row/column.
Per-channel quantization is the standard in production — GPTQ, AWQ,
and llama.cpp all use per-channel (or per-group) quantization.
The improvement over per-tensor is substantial.
"""


# --- 3.1 Implementing per-channel INT8 quantization ----------------------

def quantize_per_channel_int8(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize each row (output channel) independently.

    Returns:
        quantized: INT8 tensor (same shape)
        scales: (num_rows,) — one scale per channel
        zero_points: (num_rows,) — one zero_point per channel
    """
    num_channels = tensor.shape[0]
    scales = torch.zeros(num_channels)
    zero_points = torch.zeros(num_channels)
    quantized = torch.zeros_like(tensor, dtype=torch.int8)

    for c in range(num_channels):
        row = tensor[c]
        r_min = row.min().item()
        r_max = row.max().item()

        s = (r_max - r_min) / 255.0
        zp = r_min - (-128) * s

        scales[c] = s
        zero_points[c] = zp
        quantized[c] = torch.clamp(
            torch.round((row - zp) / s), -128, 127
        ).to(torch.int8)

    return quantized, scales, zero_points


def dequantize_per_channel_int8(
    quantized: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor,
) -> torch.Tensor:
    """Dequantize with per-channel parameters."""
    result = torch.zeros_like(quantized, dtype=torch.float32)
    for c in range(quantized.shape[0]):
        result[c] = quantized[c].float() * scales[c] + zero_points[c]
    return result


# --- 3.2 Compare per-tensor vs per-channel -------------------------------

print(f"\n--- Per-Tensor vs Per-Channel Comparison ---")

# Create a weight matrix with varying row magnitudes (realistic scenario)
weight_varied = torch.randn(8, 16)
weight_varied[0] *= 10.0  # one row with large weights
weight_varied[7] *= 0.01  # one row with tiny weights

# Per-tensor
qt, st, zpt = quantize_tensor_int8(weight_varied)
dqt = dequantize_tensor_int8(qt, st, zpt)
mse_tensor = ((weight_varied - dqt) ** 2).mean()

# Per-channel
qc, sc, zpc = quantize_per_channel_int8(weight_varied)
dqc = dequantize_per_channel_int8(qc, sc, zpc)
mse_channel = ((weight_varied - dqc) ** 2).mean()

print(f"  Weight matrix shape: {weight_varied.shape}")
print(f"  Row 0 range: [{weight_varied[0].min():.2f}, {weight_varied[0].max():.2f}] (scaled 10x)")
print(f"  Row 7 range: [{weight_varied[7].min():.4f}, {weight_varied[7].max():.4f}] (scaled 0.01x)")
print(f"\n  Per-tensor MSE:  {mse_tensor.item():.8f}")
print(f"  Per-channel MSE: {mse_channel.item():.8f}")
print(f"  Improvement:     {mse_tensor.item() / mse_channel.item():.1f}x lower error")


# --- 3.3 Per-row error breakdown -----------------------------------------

print(f"\n--- Per-Row Error Breakdown ---")
print(f"  {'Row':>4} | {'Range':>18} | {'Per-Tensor MSE':>16} | {'Per-Channel MSE':>16} | {'Winner':>10}")
print(f"  {'-'*4}-+-{'-'*18}-+-{'-'*16}-+-{'-'*16}-+-{'-'*10}")
for r in range(8):
    rng = f"[{weight_varied[r].min():.2f}, {weight_varied[r].max():.2f}]"
    mse_t = ((weight_varied[r] - dqt[r]) ** 2).mean().item()
    mse_c = ((weight_varied[r] - dqc[r]) ** 2).mean().item()
    winner = "per-ch" if mse_c < mse_t else "per-tensor"
    print(f"  {r:4d} | {rng:>18} | {mse_t:16.8f} | {mse_c:16.8f} | {winner:>10}")

print(f"\n  Per-channel wins especially on rows with extreme ranges.")
print(f"  The tiny-weight row 7 gets its own tight scale, preserving detail.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — GPTQ CONCEPT (Post-Training Quantization)
# ═══════════════════════════════════════════════════════════════════════════
#
# Naive quantization treats all weights equally. But some weights matter
# MORE than others for model output quality. GPTQ (Frantar et al., 2022)
# uses a small calibration dataset to figure out which weights are
# sensitive, then quantizes carefully to minimize output error.
#
# Key idea: use the Hessian (second derivatives of the loss) to measure
# weight sensitivity, then redistribute quantization error from sensitive
# weights to less sensitive ones.
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 4: GPTQ CONCEPT (Post-Training Quantization)")
print("=" * 60)

torch.manual_seed(42)

"""
WHY THIS MATTERS:
GPTQ is the most widely used post-training quantization method. It's how
you get "GPTQ" models on HuggingFace. Understanding its core idea —
Hessian-guided error compensation — explains why it produces better
quality than naive round-to-nearest quantization.
"""


# --- 4.1 Why naive quantization isn't optimal -----------------------------

print(f"\n--- The Problem with Naive Quantization ---")

layer = nn.Linear(16, 8, bias=False)
torch.manual_seed(42)

# Calibration data: representative inputs from a real dataset
calibration_data = torch.randn(32, 16)

# Compute outputs with the original weights
with torch.no_grad():
    original_output = layer(calibration_data)

# Quantize naively and measure output degradation
weight_q, s, zp = quantize_tensor_int8(layer.weight.data)
weight_dq = dequantize_tensor_int8(weight_q, s, zp)
naive_output = calibration_data @ weight_dq.T

naive_mse = ((original_output - naive_output) ** 2).mean()
print(f"  Layer: Linear(16, 8)")
print(f"  Calibration samples: {calibration_data.shape[0]}")
print(f"  Naive quantization output MSE: {naive_mse.item():.6f}")


# --- 4.2 Hessian-based weight sensitivity --------------------------------

"""
WHY THIS MATTERS:
The Hessian matrix H = 2 * X^T @ X (for a linear layer) tells us how
sensitive the output is to each weight. If H[i,i] is large, weight i
is important — quantization error on that weight hurts output quality.
GPTQ uses this to be careful with important weights and more aggressive
with unimportant ones.
"""

print(f"\n--- Weight Sensitivity via Hessian ---")

# Approximate Hessian for a linear layer: H = X^T @ X / n_samples
H = (calibration_data.T @ calibration_data) / calibration_data.shape[0]
sensitivity = H.diag()

print(f"  Hessian shape: {H.shape}")
print(f"  Diagonal (sensitivity per input dim):")
print(f"    {sensitivity.tolist()}")

most_sensitive = sensitivity.argmax().item()
least_sensitive = sensitivity.argmin().item()
print(f"\n  Most sensitive input dim:  {most_sensitive} (H_diag = {sensitivity[most_sensitive]:.4f})")
print(f"  Least sensitive input dim: {least_sensitive} (H_diag = {sensitivity[least_sensitive]:.4f})")
print(f"  Ratio: {sensitivity[most_sensitive] / sensitivity[least_sensitive]:.1f}x")
print(f"  → GPTQ is more careful when quantizing weights connected to dim {most_sensitive}")


# --- 4.3 Simulating GPTQ's error compensation ----------------------------

"""
WHY THIS MATTERS:
GPTQ processes weights one column at a time. After quantizing a column,
it computes the quantization error and COMPENSATES by adjusting the
remaining (not yet quantized) columns. This redistributes error from
sensitive weights to less sensitive ones.
"""

print(f"\n--- GPTQ Error Compensation (Simplified) ---")

W = layer.weight.data.clone()
W_quant = torch.zeros_like(W)

# Process columns left to right (simplified GPTQ)
H_inv_diag = 1.0 / (H.diag() + 1e-6)  # inverse Hessian diagonal

for col in range(W.shape[1]):
    # Quantize this column
    col_data = W[:, col]
    q_col, s_col, zp_col = quantize_tensor_int8(col_data.unsqueeze(1))
    dq_col = dequantize_tensor_int8(q_col, s_col, zp_col).squeeze(1)
    W_quant[:, col] = dq_col

    # Compute quantization error for this column
    quant_error = col_data - dq_col

    # Compensate: distribute error to remaining columns
    # (proportional to Hessian inverse — less sensitive columns absorb more)
    if col < W.shape[1] - 1:
        remaining = W[:, col + 1:]
        for j in range(remaining.shape[1]):
            compensation = quant_error * H_inv_diag[col + 1 + j] / H_inv_diag[col]
            W[:, col + 1 + j] += compensation * 0.1

# Measure GPTQ-style output vs naive output
gptq_output = calibration_data @ W_quant.T
gptq_mse = ((original_output - gptq_output) ** 2).mean()

print(f"  Naive quantization MSE:   {naive_mse.item():.6f}")
print(f"  GPTQ-style quant MSE:     {gptq_mse.item():.6f}")
improvement = naive_mse.item() / max(gptq_mse.item(), 1e-10)
print(f"  Improvement:              {improvement:.1f}x")


# --- 4.4 The full GPTQ pipeline ------------------------------------------

print(f"\n--- Full GPTQ Pipeline (Conceptual) ---")
print("""
  ┌─────────────────────────────────────────────────────────────┐
  │                  GPTQ QUANTIZATION PIPELINE                  │
  │                                                             │
  │  1. COLLECT CALIBRATION DATA                                │
  │     128-256 samples from a representative dataset           │
  │     (e.g., C4, WikiText, or your target domain)             │
  │                                                             │
  │  2. COMPUTE HESSIAN (per layer)                             │
  │     H = X^T @ X   (input activations)                      │
  │     Tells us which weights are sensitive                    │
  │                                                             │
  │  3. QUANTIZE COLUMN-BY-COLUMN                               │
  │     For each column in the weight matrix:                   │
  │       a) Round weights to nearest INT value                 │
  │       b) Compute quantization error                         │
  │       c) Compensate remaining columns using H^{-1}          │
  │                                                             │
  │  4. SAVE QUANTIZED WEIGHTS                                  │
  │     Store: INT4/INT8 weights + scales + zero_points         │
  │     Format: GPTQ, GGUF, or SafeTensors                     │
  └─────────────────────────────────────────────────────────────┘
""")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — SPECULATIVE DECODING
# ═══════════════════════════════════════════════════════════════════════════
#
# LLM inference is memory-bandwidth bound, not compute bound. A small
# "draft" model is nearly as fast as a large "target" model per token
# because both are bottlenecked by memory reads.
#
# Speculative decoding: use a fast draft model to propose K tokens, then
# verify them all at once with the big model. If the draft model is decent,
# most tokens get accepted — you get K tokens for the cost of ~1 big step.
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 5: SPECULATIVE DECODING")
print("=" * 60)

torch.manual_seed(42)

"""
WHY THIS MATTERS:
Speculative decoding can give 2-3x speedup WITHOUT changing model
quality at all — the output distribution is mathematically identical
to the target model. It's used in production systems (vLLM, TGI)
and is one of the most impactful inference optimizations.
"""


# --- 5.1 The draft-verify paradigm ---------------------------------------

print(f"\n--- Speculative Decoding: Draft + Verify ---")
print("""
  STANDARD AUTOREGRESSIVE GENERATION (slow):
  ┌────────┐   ┌────────┐   ┌────────┐   ┌────────┐
  │ Token 1│ → │ Token 2│ → │ Token 3│ → │ Token 4│  (4 big model calls)
  └────────┘   └────────┘   └────────┘   └────────┘

  SPECULATIVE DECODING (fast):
  ┌──────────────────────────────────────┐
  │ Draft model proposes: T1, T2, T3, T4│  (4 small model calls — cheap)
  └──────────────┬───────────────────────┘
                 │
  ┌──────────────▼───────────────────────┐
  │ Target model verifies ALL at once    │  (1 big model call — parallel)
  │ Accept: T1 ✓  T2 ✓  T3 ✗           │
  └──────────────────────────────────────┘
  Result: got 2 tokens for 1 big model call (+ cheap draft calls)
""")


# --- 5.2 Simulating speculative decoding ---------------------------------

class DraftModel:
    """Simulates a small fast model (70% match rate with target)."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def predict_probs(self, context_len: int) -> torch.Tensor:
        torch.manual_seed(context_len * 7 + 3)
        logits = torch.randn(self.vocab_size)
        return F.softmax(logits, dim=-1)


class TargetModel:
    """Simulates a large accurate model."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def predict_probs(self, context_len: int) -> torch.Tensor:
        torch.manual_seed(context_len * 7 + 3)
        logits = torch.randn(self.vocab_size) * 1.2 + 0.3
        return F.softmax(logits, dim=-1)


def speculative_decode(
    draft: DraftModel,
    target: TargetModel,
    num_draft_tokens: int = 4,
    context_len: int = 10,
) -> dict:
    """
    Simulate one round of speculative decoding.

    1. Draft model proposes num_draft_tokens
    2. Target model verifies all proposals in parallel
    3. Accept tokens where target agrees with draft (probabilistically)
    """
    draft_tokens = []
    draft_probs = []

    # Step 1: Draft model generates K tokens autoregressively
    for i in range(num_draft_tokens):
        probs = draft.predict_probs(context_len + i)
        token = torch.multinomial(probs, 1).item()
        draft_tokens.append(token)
        draft_probs.append(probs[token].item())

    # Step 2: Target model scores ALL draft tokens in parallel
    target_probs = []
    for i in range(num_draft_tokens):
        probs = target.predict_probs(context_len + i)
        target_probs.append(probs[draft_tokens[i]].item())

    # Step 3: Accept/reject using probability ratio
    accepted = 0
    for i in range(num_draft_tokens):
        acceptance_prob = min(1.0, target_probs[i] / max(draft_probs[i], 1e-10))
        if torch.rand(1).item() < acceptance_prob:
            accepted += 1
        else:
            break  # reject this and all subsequent tokens

    return {
        "draft_tokens": draft_tokens,
        "draft_probs": draft_probs,
        "target_probs": target_probs,
        "accepted": accepted,
        "proposed": num_draft_tokens,
    }


vocab_size = 100
draft = DraftModel(vocab_size)
target = TargetModel(vocab_size)

print(f"\n--- Running Speculative Decoding Simulation ---")
print(f"  Vocab size: {vocab_size}")
print(f"  Draft tokens per round: 4\n")

total_accepted = 0
total_proposed = 0
num_rounds = 20

for round_idx in range(num_rounds):
    result = speculative_decode(draft, target, num_draft_tokens=4, context_len=10 + round_idx)
    total_accepted += result["accepted"]
    total_proposed += result["proposed"]

    if round_idx < 5:
        status = ["✓" if i < result["accepted"] else "✗"
                  for i in range(result["proposed"])]
        print(f"  Round {round_idx + 1}: proposed={result['proposed']}, "
              f"accepted={result['accepted']}  [{' '.join(status)}]")

print(f"  ...")
acceptance_rate = total_accepted / total_proposed
print(f"\n  Total: {total_accepted}/{total_proposed} tokens accepted "
      f"({acceptance_rate:.1%} acceptance rate)")
print(f"  Effective speedup: ~{1.0 / (1.0 - acceptance_rate * 0.7):.1f}x "
      f"(depends on draft/target speed ratio)")


# --- 5.3 When speculative decoding helps most ----------------------------

print(f"\n--- When Speculative Decoding Works Best ---")
scenarios = [
    ("Code completion",   "HIGH (predictable patterns)",    "~2.5x"),
    ("Translation",       "HIGH (structured output)",       "~2.3x"),
    ("Creative writing",  "MEDIUM (less predictable)",      "~1.8x"),
    ("Brainstorming",     "LOW (high entropy outputs)",     "~1.3x"),
]
print(f"  {'Task':>20} | {'Draft Accuracy':>30} | {'Speedup':>8}")
print(f"  {'-'*20}-+-{'-'*30}-+-{'-'*8}")
for task, accuracy, speedup in scenarios:
    print(f"  {task:>20} | {accuracy:>30} | {speedup:>8}")
print(f"\n  Key insight: speculative decoding helps most when the output is")
print(f"  predictable (low entropy). The draft model can guess correctly more often.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — KV CACHE OPTIMIZATION (GQA vs MQA vs MHA)
# ═══════════════════════════════════════════════════════════════════════════
#
# During generation, the KV cache stores keys and values for all past
# tokens. With Multi-Head Attention (MHA), each head has its own K and V —
# the cache grows linearly with num_heads × seq_len × d_head.
#
# Grouped Query Attention (GQA) and Multi-Query Attention (MQA) reduce
# cache size by sharing K/V across groups of query heads.
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("PART 6: KV CACHE OPTIMIZATION (GQA vs MQA vs MHA)")
print("=" * 60)

torch.manual_seed(42)

"""
WHY THIS MATTERS:
For long sequences, the KV cache dominates GPU memory — even more than
the model weights. A 7B model generating 8K tokens can need 8+ GB just
for the KV cache. GQA (used in LLaMA-2 70B) reduces this by 4-8x
with minimal quality loss.
"""


# --- 6.1 KV cache size calculation ----------------------------------------

def kv_cache_size_gb(
    num_layers: int,
    num_kv_heads: int,
    d_head: int,
    seq_len: int,
    batch_size: int = 1,
    dtype_bytes: int = 2,  # float16
) -> float:
    """Calculate KV cache size in GB."""
    # K and V each: (batch, num_kv_heads, seq_len, d_head)
    per_layer = 2 * batch_size * num_kv_heads * seq_len * d_head * dtype_bytes
    total = per_layer * num_layers
    return total / 1e9


print(f"\n--- KV Cache Size for 7B Model (32 layers, d_model=4096) ---")
print(f"  (FP16, batch_size=1)\n")

configs = {
    "MHA (32 KV heads)": {"num_kv_heads": 32, "d_head": 128},
    "GQA (8 KV heads)":  {"num_kv_heads": 8,  "d_head": 128},
    "GQA (4 KV heads)":  {"num_kv_heads": 4,  "d_head": 128},
    "MQA (1 KV head)":   {"num_kv_heads": 1,  "d_head": 128},
}

seq_lengths = [2048, 4096, 8192, 16384]

print(f"  {'Config':>22} |", end="")
for sl in seq_lengths:
    print(f" {sl:>7} tok |", end="")
print()
print(f"  {'-'*22}-+", end="")
for _ in seq_lengths:
    print(f"-{'-'*10}-+", end="")
print()

for config_name, params in configs.items():
    print(f"  {config_name:>22} |", end="")
    for sl in seq_lengths:
        size = kv_cache_size_gb(
            num_layers=32, seq_len=sl, **params
        )
        print(f" {size:>8.2f} GB |", end="")
    print()


# --- 6.2 Implementing MHA, GQA, and MQA side by side ---------------------

print(f"\n--- Attention Variants Implementation ---")

class MultiHeadAttention(nn.Module):
    """Standard MHA: each query head has its own K and V head."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class GroupedQueryAttention(nn.Module):
    """GQA: multiple query heads share each KV head."""

    def __init__(self, d_model: int, n_q_heads: int, n_kv_heads: int):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_q_heads // n_kv_heads
        self.d_head = d_model // n_q_heads

        self.q_proj = nn.Linear(d_model, n_q_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_q_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Repeat KV heads to match query head count
        k = k.repeat_interleave(self.group_size, dim=1)
        v = v.repeat_interleave(self.group_size, dim=1)

        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class MultiQueryAttention(nn.Module):
    """MQA: all query heads share a single KV head."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_head, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, T, 1, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, T, 1, self.d_head).transpose(1, 2)

        # Broadcast single KV head across all query heads
        attn = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


# --- 6.3 Compare parameter counts and outputs ----------------------------

d_model = 256
n_heads = 8

mha = MultiHeadAttention(d_model, n_heads)
gqa = GroupedQueryAttention(d_model, n_q_heads=n_heads, n_kv_heads=2)
mqa = MultiQueryAttention(d_model, n_heads)

def count_params(m): return sum(p.numel() for p in m.parameters())

x = torch.randn(2, 16, d_model)

print(f"  d_model={d_model}, n_query_heads={n_heads}, seq_len=16, batch=2\n")
print(f"  {'Variant':>8} | {'KV Heads':>9} | {'Params':>8} | {'KV Params':>10} | {'Output Shape':>14}")
print(f"  {'-'*8}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}-+-{'-'*14}")

with torch.no_grad():
    for name, module, kv_heads in [
        ("MHA", mha, n_heads),
        ("GQA", gqa, 2),
        ("MQA", mqa, 1),
    ]:
        out = module(x)
        total_p = count_params(module)
        kv_p = sum(p.numel() for n, p in module.named_parameters() if "k_proj" in n or "v_proj" in n)
        print(f"  {name:>8} | {kv_heads:>9} | {total_p:>8,} | {kv_p:>10,} | {str(out.shape):>14}")


# --- 6.4 Memory savings summary ------------------------------------------

print(f"\n--- KV Cache Savings at Scale (LLaMA-2 7B config) ---")
print(f"  32 layers, d_model=4096, 32 query heads, d_head=128")
print(f"  Sequence length: 4096, FP16\n")

for name, kv_heads in [("MHA", 32), ("GQA-8", 8), ("GQA-4", 4), ("MQA", 1)]:
    cache_gb = kv_cache_size_gb(num_layers=32, num_kv_heads=kv_heads, d_head=128, seq_len=4096)
    savings = 1 - (kv_heads / 32)
    print(f"  {name:>6}: {cache_gb:.2f} GB KV cache  ({savings:.0%} reduction vs MHA)")

print(f"\n  LLaMA-2 70B uses GQA with 8 KV heads (vs 64 query heads)")
print(f"  → 8x smaller KV cache, nearly the same quality as MHA")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
Exercise 1: INT4 Quantization from Scratch
--------------------------------------------
  a) Modify quantize_tensor_int8() to create quantize_tensor_int4().
     Map values to the range [-8, 7] (16 levels instead of 256).
  b) Quantize a (64, 64) random weight matrix to INT4.
  c) Compare MSE against INT8 quantization of the same matrix.
  d) What is the compression ratio vs FP32? vs INT8?
  Hint: INT4 uses 4 bits per weight = 0.5 bytes.

Exercise 2: Quantize a Full MLP
----------------------------------
  a) Build a 3-layer MLP: Linear(256, 512) -> GELU -> Linear(512, 256)
  b) Quantize ALL weight matrices to INT8 (per-channel).
  c) Run the same input through both the original and quantized MLPs.
  d) Measure output MSE. Also measure total memory savings.
  e) Repeat with per-tensor quantization and compare error.

Exercise 3: Speculative Decoding Acceptance Rate
---------------------------------------------------
  a) Modify DraftModel to have a configurable "accuracy" parameter
     (0.0 = random, 1.0 = perfect match with target).
  b) Run speculative decoding with accuracy = [0.2, 0.4, 0.6, 0.8, 0.95].
  c) Plot (text-based) acceptance rate vs draft accuracy.
  d) At what accuracy does speculative decoding break even?
     (i.e., speedup > 1.0, considering draft model overhead)

Exercise 4: Group Size Sweep for GQA
---------------------------------------
  a) Create GQA modules with group sizes: 1 (MHA), 2, 4, 8, 32 (MQA).
     (For n_q_heads=32, that means n_kv_heads = 32, 16, 8, 4, 1)
  b) Count KV projection parameters for each.
  c) Run the same input through each and compare outputs using cosine
     similarity with MHA output as the reference.
  d) What's the sweet spot: biggest memory saving with smallest quality loss?

Exercise 5: Calibration Data Impact on GPTQ
----------------------------------------------
  a) Create a Linear(64, 32) layer.
  b) Implement the simplified GPTQ from Part 4.3.
  c) Run GPTQ with calibration sets of size: [8, 32, 128, 512].
  d) For each, measure output MSE on a fixed test set of 100 samples.
  e) Does more calibration data always help? Is there a point of
     diminishing returns?
""")
