"""
============================================================================
 LESSON 9: Distributed Training & Scaling
============================================================================

 Goal: Learn the techniques that make it possible to train models too large
       for a single GPU — and how to choose the right model size.

 Topics covered:
   1. Data Parallelism (DDP)    — replicate model, split data, all-reduce
   2. FSDP                      — shard parameters, gradients, optimizer states
   3. Tensor Parallelism        — split individual layers across GPUs
   4. Mixed Precision           — FP16/BF16 training with torch.amp
   5. Flash Attention           — IO-aware memory-efficient attention
   6. Scaling Laws              — Chinchilla-optimal compute allocation

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites:
   - Lesson 6 (transformer architecture)
   - Lesson 8 (pre-training your LLM)
============================================================================
"""

import torch
import torch.nn as nn
import math

torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — DATA PARALLELISM (DDP)
# ═══════════════════════════════════════════════════════════════════════════
#
# The simplest multi-GPU strategy: put a COPY of the full model on every
# GPU, split the training data, and synchronize gradients after each step.
#
# How DDP works:
#   1. Replicate — copy the full model to each GPU
#   2. Scatter   — split each mini-batch (each GPU gets 1/N of the data)
#   3. Forward   — each GPU computes loss independently on its shard
#   4. All-Reduce— average the gradients across ALL GPUs
#   5. Update    — each GPU applies the same optimizer step → weights stay in sync
#
# The "all-reduce" is the critical operation — it sums tensors across all
# GPUs in a single communication round (ring all-reduce is O(N) efficient).
#
# DDP is the default choice when the model fits in a single GPU's memory.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: DATA PARALLELISM (DDP)")
print("=" * 60)

torch.manual_seed(42)


class SimpleModel(nn.Module):
    """Tiny model for demonstrating parallel training concepts."""

    def __init__(self, d_model=256, vocab_size=1000):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.emb(x)
        h = self.ffn(h)
        return self.head(h)


model = SimpleModel(d_model=256, vocab_size=1000)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")

# --- DDP wrapping pattern (conceptual — requires distributed init) --------
#
#   import torch.distributed as dist
#   from torch.nn.parallel import DistributedDataParallel as DDP
#
#   dist.init_process_group("nccl")                    # NCCL backend for GPUs
#   local_rank = int(os.environ["LOCAL_RANK"])
#   model = model.to(local_rank)
#   model = DDP(model, device_ids=[local_rank])         # wraps the model
#
#   # Training loop is IDENTICAL to single-GPU — DDP handles gradient sync.
#   # Launch with: torchrun --nproc_per_node=4 train.py

print("""
DDP wrapping pattern:
  model = DDP(model, device_ids=[local_rank])

  The training loop stays IDENTICAL to single-GPU.
  DDP intercepts loss.backward() and automatically
  all-reduces gradients before optimizer.step().
""")


# --- Simulate all-reduce: averaging gradients across GPUs -----------------

def simulate_all_reduce(gpu_gradients: list[torch.Tensor]) -> torch.Tensor:
    """
    All-reduce averages gradients across all GPUs.
    In real DDP, this happens over NCCL with ring all-reduce.
    Here we simulate it on CPU.
    """
    stacked = torch.stack(gpu_gradients)
    return stacked.mean(dim=0)


n_gpus = 4
param = torch.randn(8)

gpu_grads = []
for gpu_id in range(n_gpus):
    torch.manual_seed(gpu_id)
    data = torch.randn(32, 8)
    grad = data.mean(dim=0)
    gpu_grads.append(grad)

print(f"Simulating {n_gpus}-GPU all-reduce on a parameter of shape {param.shape}:")
for i, g in enumerate(gpu_grads):
    print(f"  GPU {i} gradient: [{g[0]:.4f}, {g[1]:.4f}, {g[2]:.4f}, ...]")

avg_grad = simulate_all_reduce(gpu_grads)
print(f"  All-reduced:    [{avg_grad[0]:.4f}, {avg_grad[1]:.4f}, {avg_grad[2]:.4f}, ...]")
print(f"  After all-reduce, every GPU has the SAME averaged gradient.")

# --- Effective batch size -------------------------------------------------

per_gpu_batch = 32
n_gpus = 8
effective_batch = per_gpu_batch * n_gpus
print(f"\nEffective batch size:")
print(f"  Per-GPU batch:  {per_gpu_batch}")
print(f"  Number of GPUs: {n_gpus}")
print(f"  Effective batch: {per_gpu_batch} × {n_gpus} = {effective_batch}")
print(f"  With grad accumulation (4 steps): {effective_batch * 4} = {effective_batch * 4}")
print(f"  LLaMA-2 used effective batch size of ~4M tokens")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — FULLY SHARDED DATA PARALLELISM (FSDP)
# ═══════════════════════════════════════════════════════════════════════════
#
# Problem: DDP requires each GPU to hold the FULL model.  For a 7B model:
#   - Parameters:       7B × 4 bytes (FP32)            =  28 GB
#   - Gradients:        7B × 4 bytes                   =  28 GB
#   - Optimizer states: 7B × 8 bytes (Adam has 2 states) = 56 GB
#   - Total per GPU:    ~112 GB  ← doesn't fit on an 80GB A100!
#
# FSDP solves this by SHARDING across GPUs.
# Each GPU holds only 1/N of the parameters, gradients, and optimizer states.
# Before a forward/backward on a layer, FSDP gathers that layer's params
# from all GPUs, computes, then discards the non-local shard.
#
# Based on Microsoft's ZeRO (Zero Redundancy Optimizer):
#   ZeRO Stage 1: Shard optimizer states only
#   ZeRO Stage 2: + shard gradients
#   ZeRO Stage 3: + shard parameters (= FSDP)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: FSDP (Fully Sharded Data Parallelism)")
print("=" * 60)


def memory_per_gpu(num_params_B: float, num_gpus: int, precision: str = "fp32"):
    """
    Calculate per-GPU memory under different sharding strategies.

    For Adam optimizer in FP32:
      Parameters:       N × bytes_per_param
      Gradients:        N × bytes_per_param
      Optimizer states: N × 8 bytes (momentum + variance, both FP32)
    """
    bytes_per_param = 4 if precision == "fp32" else 2

    param_mem = num_params_B * 1e9 * bytes_per_param
    grad_mem = num_params_B * 1e9 * bytes_per_param
    opt_mem = num_params_B * 1e9 * 8   # Adam always FP32 master

    results = {}

    # DDP: full replica on every GPU
    results["DDP"] = (param_mem + grad_mem + opt_mem) / 1e9

    # ZeRO Stage 1: shard optimizer only
    results["ZeRO-1"] = (param_mem + grad_mem + opt_mem / num_gpus) / 1e9

    # ZeRO Stage 2: shard optimizer + gradients
    results["ZeRO-2"] = (param_mem + (grad_mem + opt_mem) / num_gpus) / 1e9

    # ZeRO Stage 3 / FSDP: shard everything
    results["FSDP"] = (param_mem + grad_mem + opt_mem) / num_gpus / 1e9

    return results


print("\n--- Memory per GPU for a 7B parameter model (FP32) ---")
mem = memory_per_gpu(num_params_B=7.0, num_gpus=4)
print(f"{'Strategy':10s} {'Per-GPU Memory':>15s}")
print("-" * 28)
for strategy, gb in mem.items():
    fits = "fits 80GB" if gb <= 80 else "EXCEEDS 80GB"
    print(f"{strategy:10s} {gb:>12.1f} GB   ({fits})")

print(f"\nFSDP saves {mem['DDP'] / mem['FSDP']:.1f}x memory vs DDP!")
print(f"A 7B model that needs {mem['DDP']:.0f} GB/GPU with DDP")
print(f"only needs {mem['FSDP']:.0f} GB/GPU with FSDP across 4 GPUs.")

# --- FSDP wrapping pattern (conceptual) ---
print("""
FSDP wrapping pattern:
  from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

  model = FSDP(
      model,
      sharding_strategy=ShardingStrategy.FULL_SHARD,  # = ZeRO Stage 3
      mixed_precision=MixedPrecision(
          param_dtype=torch.bfloat16,
          reduce_dtype=torch.bfloat16,
      ),
  )
""")

# --- Mixed precision with FSDP saves even more ---
print("--- With mixed precision (BF16 params) + FSDP ---")
mem_bf16 = memory_per_gpu(num_params_B=7.0, num_gpus=4, precision="bf16")
print(f"{'Strategy':10s} {'FP32':>10s} {'BF16':>10s} {'Savings':>10s}")
print("-" * 42)
for strat in mem.keys():
    savings = (1 - mem_bf16[strat] / mem[strat]) * 100
    print(f"{strat:10s} {mem[strat]:>8.1f} GB {mem_bf16[strat]:>8.1f} GB   {savings:>6.0f}%")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — TENSOR PARALLELISM
# ═══════════════════════════════════════════════════════════════════════════
#
# Tensor Parallelism splits INDIVIDUAL LAYERS across GPUs.
# Unlike DDP (which replicates the whole model), TP partitions the weight
# matrices themselves.
#
# Example: a Linear(4096, 4096) has a 4096×4096 weight matrix.
# With 2-way TP, each GPU holds a 4096×2048 slice.
#
# Two strategies for nn.Linear(in, out):
#   Column Parallel: split output dimension → each GPU has (in, out/N)
#   Row Parallel:    split input dimension  → each GPU has (in/N, out)
#
# In practice, TP is used WITHIN a single node (fast NVLink interconnect),
# while FSDP is used ACROSS nodes (slower network).
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3: TENSOR PARALLELISM")
print("=" * 60)

torch.manual_seed(42)

in_features = 4096
out_features = 4096
n_gpus = 2

# --- Full (unsplit) linear layer ---
full_weight = torch.randn(out_features, in_features) * 0.01
full_bias = torch.randn(out_features) * 0.01
x = torch.randn(2, in_features)

full_output = x @ full_weight.T + full_bias
print(f"\nFull Linear({in_features}, {out_features})")
print(f"  Weight shape: {full_weight.shape}")
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {full_output.shape}")
print(f"  Weight memory: {full_weight.numel() * 4 / 1e6:.1f} MB (FP32)")


# --- Column Parallel: split output dimension ---
print(f"\n--- Column Parallel (split output features) ---")
shard_size = out_features // n_gpus
gpu0_weight = full_weight[:shard_size, :]      # (2048, 4096) — first half of rows
gpu1_weight = full_weight[shard_size:, :]      # (2048, 4096) — second half

gpu0_bias = full_bias[:shard_size]
gpu1_bias = full_bias[shard_size:]

gpu0_out = x @ gpu0_weight.T + gpu0_bias       # (2, 2048)
gpu1_out = x @ gpu1_weight.T + gpu1_bias       # (2, 2048)

# Concatenate outputs from both GPUs
reconstructed = torch.cat([gpu0_out, gpu1_out], dim=-1)

print(f"  GPU 0: Linear({in_features}, {shard_size})")
print(f"    Weight: {gpu0_weight.shape}, Output: {gpu0_out.shape}")
print(f"  GPU 1: Linear({in_features}, {shard_size})")
print(f"    Weight: {gpu1_weight.shape}, Output: {gpu1_out.shape}")
print(f"  Concatenated: {reconstructed.shape}")
print(f"  Matches full output: {torch.allclose(full_output, reconstructed, atol=1e-5)}")
print(f"  Memory per GPU: {gpu0_weight.numel() * 4 / 1e6:.1f} MB (half of {full_weight.numel() * 4 / 1e6:.1f} MB)")


# --- Row Parallel: split input dimension ---
print(f"\n--- Row Parallel (split input features) ---")
in_shard = in_features // n_gpus
gpu0_weight_rp = full_weight[:, :in_shard]     # (4096, 2048)
gpu1_weight_rp = full_weight[:, in_shard:]     # (4096, 2048)

x_gpu0 = x[:, :in_shard]                       # (2, 2048) — first half of input
x_gpu1 = x[:, in_shard:]                       # (2, 2048) — second half

gpu0_partial = x_gpu0 @ gpu0_weight_rp.T       # (2, 4096)
gpu1_partial = x_gpu1 @ gpu1_weight_rp.T       # (2, 4096)

# All-reduce (sum partial results) + add bias
reconstructed_rp = gpu0_partial + gpu1_partial + full_bias

print(f"  GPU 0: input shard {x_gpu0.shape} × weight shard {gpu0_weight_rp.shape}")
print(f"  GPU 1: input shard {x_gpu1.shape} × weight shard {gpu1_weight_rp.shape}")
print(f"  Sum partial results (all-reduce): {reconstructed_rp.shape}")
print(f"  Matches full output: {torch.allclose(full_output, reconstructed_rp, atol=1e-5)}")

print(f"""
Summary: Tensor Parallelism for Linear({in_features}, {out_features}) on {n_gpus} GPUs
  Column Parallel: each GPU has ({in_features}, {shard_size}) — concat outputs
  Row Parallel:    each GPU has ({in_shard}, {out_features}) — sum outputs (all-reduce)
  Memory per GPU:  {gpu0_weight.numel() * 4 / 1e6:.1f} MB instead of {full_weight.numel() * 4 / 1e6:.1f} MB
""")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — MIXED PRECISION TRAINING
# ═══════════════════════════════════════════════════════════════════════════
#
# Mixed precision uses lower-precision dtypes (FP16 or BF16) for most
# computation, while keeping FP32 "master weights" for the optimizer.
#
# Benefits:
#   - 2× less memory for activations and gradients
#   - 2× faster matrix multiplies on modern GPUs (Tensor Cores)
#   - No accuracy loss when done correctly
#
# torch.amp.autocast automatically chooses the right dtype per operation:
#   - Matrix multiplies → BF16/FP16 (safe and fast)
#   - Reductions, norms → FP32 (needs precision)
#
# GradScaler: only needed for FP16 (not BF16) to prevent gradient underflow.
# BF16 has the same exponent range as FP32, so it doesn't underflow.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 4: MIXED PRECISION TRAINING")
print("=" * 60)

torch.manual_seed(42)


# --- Memory comparison: FP32 vs FP16/BF16 --------------------------------

d_model = 4096
n_layers = 32
batch_size = 8
seq_len = 2048

activations_elements = batch_size * seq_len * d_model * n_layers
fp32_mem = activations_elements * 4 / 1e9
fp16_mem = activations_elements * 2 / 1e9

print(f"\nActivation memory for a {d_model}-dim, {n_layers}-layer model:")
print(f"  Batch={batch_size}, Seq={seq_len}")
print(f"  FP32: {fp32_mem:.1f} GB")
print(f"  FP16: {fp16_mem:.1f} GB")
print(f"  Savings: {fp32_mem - fp16_mem:.1f} GB ({(1 - fp16_mem/fp32_mem)*100:.0f}%)")


# --- torch.amp.autocast in action ----------------------------------------

model_amp = nn.Sequential(
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Linear(2048, 512),
)
x_amp = torch.randn(4, 128, 512)

print(f"\n--- autocast dtype behavior ---")
print(f"Weight dtype (always FP32 master copy): {model_amp[0].weight.dtype}")

with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
    y_amp = model_amp(x_amp)
    print(f"Output dtype inside autocast:  {y_amp.dtype}")
    print(f"Weight dtype inside autocast:  {model_amp[0].weight.dtype}")

print(f"Weight dtype outside autocast: {model_amp[0].weight.dtype}")

# --- Full mixed-precision training loop pattern ---
print("""
Mixed-precision training loop:

  scaler = torch.amp.GradScaler()           # FP16 only, not needed for BF16

  for batch in dataloader:
      with torch.amp.autocast("cuda", dtype=torch.bfloat16):
          logits = model(batch["input_ids"])
          loss = criterion(logits, batch["labels"])

      # BF16 path (modern GPUs):
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      # FP16 path (older GPUs — needs loss scaling):
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
      optimizer.zero_grad()
""")


# --- Dtype comparison table -----------------------------------------------

print("--- Precision formats comparison ---")
dtypes = [
    ("FP32", torch.float32, 4),
    ("FP16", torch.float16, 2),
    ("BF16", torch.bfloat16, 2),
]

print(f"{'Format':6s} {'Bytes':>6s} {'Max Value':>14s} {'Min Positive':>15s}")
print("-" * 45)
for name, dt, size in dtypes:
    info = torch.finfo(dt)
    print(f"{name:6s} {size:>6d} {info.max:>14.0f} {info.tiny:>15.2e}")

print(f"\nBF16 has the same range as FP32 (no overflow/underflow risk)")
print(f"but less precision (7-bit mantissa vs 23-bit).")
print(f"This is why BF16 is preferred for LLM training — safe AND fast.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — FLASH ATTENTION
# ═══════════════════════════════════════════════════════════════════════════
#
# Standard attention computes the FULL n×n attention matrix:
#   scores = Q @ K^T / sqrt(d_k)     → (n, n) matrix stored in HBM
#   weights = softmax(scores)          → another (n, n) matrix
#   output = weights @ V               → read (n, n) from HBM again
#
# For n=8192 (common context length), the attention matrix has 67M entries.
# For n=128K (GPT-4), that's 16 BILLION entries — 64 GB in FP32.
#
# Flash Attention (Dao et al., 2022) avoids materializing the full n×n
# matrix by computing attention in TILES that fit in GPU SRAM (fast cache).
#
# Key insight: attention is IO-bound, not compute-bound.
# The bottleneck is reading/writing to HBM, not the actual math.
# By keeping tiles in SRAM and fusing operations, Flash Attention:
#   - Reduces HBM reads from O(n²) to O(n)
#   - Uses O(n) memory instead of O(n²)
#   - Is 2-4× faster in practice
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 5: FLASH ATTENTION")
print("=" * 60)

torch.manual_seed(42)


# --- Standard attention: O(n²) memory ------------------------------------

def standard_attention(Q, K, V):
    """
    Standard scaled dot-product attention.
    Materializes the FULL n×n attention matrix in memory.
    Memory: O(n² + n*d) — dominated by the n×n matrix for large n.
    """
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return weights @ V, scores.shape


def attention_memory_bytes(seq_len, d_model, n_heads, dtype_bytes=2):
    """Calculate memory for standard attention (per head, per batch)."""
    d_head = d_model // n_heads
    qkv_mem = 3 * seq_len * d_head * dtype_bytes
    score_matrix = seq_len * seq_len * dtype_bytes
    weight_matrix = seq_len * seq_len * dtype_bytes
    output_mem = seq_len * d_head * dtype_bytes
    return {
        "Q+K+V": qkv_mem,
        "score_matrix": score_matrix,
        "weight_matrix": weight_matrix,
        "output": output_mem,
        "total": qkv_mem + score_matrix + weight_matrix + output_mem,
    }


# Compare memory at different sequence lengths
print(f"\n--- Attention memory per head (BF16) ---")
print(f"  d_model=4096, n_heads=32, d_head=128\n")
print(f"{'Seq Len':>10s} {'Score Matrix':>14s} {'Total':>10s} {'Bottleneck':>12s}")
print("-" * 50)
for seq_len in [512, 2048, 8192, 32768, 131072]:
    mem = attention_memory_bytes(seq_len, 4096, 32, dtype_bytes=2)
    score_mb = mem["score_matrix"] / 1e6
    total_mb = mem["total"] / 1e6
    pct = mem["score_matrix"] / mem["total"] * 100
    print(f"{seq_len:>10,} {score_mb:>12.1f} MB {total_mb:>8.1f} MB   {pct:>5.0f}% score")

print(f"\nThe n×n score matrix dominates at long sequences!")
print(f"At 128K tokens: {131072**2 * 2 / 1e9:.1f} GB just for the score matrix (per head).")


# --- Flash Attention: tiled computation, O(n) memory ---------------------

print(f"""
Flash Attention key ideas:

  Standard Attention:
    1. Compute FULL score matrix Q@K^T         → O(n²) memory
    2. Apply softmax to FULL matrix            → O(n²) memory
    3. Multiply weights × V                    → reads O(n²) from HBM

  Flash Attention:
    1. Split Q, K, V into TILES (blocks)
    2. For each tile pair, compute attention IN SRAM (on-chip)
    3. Accumulate results with online softmax (no full matrix needed)
    4. Write only the final output to HBM       → O(n) memory

  Memory complexity:
    Standard: O(n²) — must store full attention matrix
    Flash:    O(n)  — only stores Q/K/V tiles + running accumulators

  Speed improvement: 2-4× faster (IO-bound → reduced HBM traffic)
""")

# --- Demonstrate the memory savings with a concrete example ---------------

n = 8192
d = 128
standard_mem = n * n * 2            # full n×n matrix in BF16
flash_mem = n * d * 2 * 3 + 1024   # Q+K+V tiles + small accumulators

print(f"Concrete example: seq_len={n}, d_head={d} (BF16)")
print(f"  Standard attention memory: {standard_mem / 1e6:.1f} MB  (O(n²))")
print(f"  Flash attention memory:    {flash_mem / 1e6:.1f} MB  (O(n))")
print(f"  Reduction: {standard_mem / flash_mem:.0f}×")

# --- PyTorch's built-in Flash Attention ---

print(f"""
Using Flash Attention in PyTorch:

  # PyTorch 2.0+ includes Flash Attention via SDPA:
  from torch.nn.functional import scaled_dot_product_attention

  output = scaled_dot_product_attention(
      query, key, value,
      is_causal=True,      # causal mask for autoregressive LM
  )

  # PyTorch automatically selects the best backend:
  #   - FlashAttention-2 (Dao, 2023)
  #   - Memory-Efficient Attention (xformers)
  #   - Math fallback (standard implementation)
""")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — SCALING LAWS
# ═══════════════════════════════════════════════════════════════════════════
#
# The Chinchilla paper (Hoffmann et al., 2022) answered a fundamental
# question: given a fixed compute budget, what's the optimal split between
# model size and training data?
#
# Key findings:
#   - Previous models (GPT-3, Gopher) were UNDERTRAINED — too large for
#     the amount of data they saw.
#   - Optimal: training tokens ≈ 20 × model parameters
#   - Compute scales equally with model size and data:
#       C ≈ 6 × N × D  (FLOPs ≈ 6 × params × tokens)
#
# This changed how the field trains models:
#   Before Chinchilla: make the model as big as possible
#   After Chinchilla:  balance model size with enough training data
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 6: SCALING LAWS")
print("=" * 60)


def compute_flops(num_params: float, num_tokens: float) -> float:
    """
    Estimate training compute in FLOPs.
    Rule of thumb: C ≈ 6 × N × D
      - 6 comes from: 2 (multiply-add) × 3 (forward + backward is ~3× forward)
    """
    return 6 * num_params * num_tokens


def chinchilla_optimal(compute_flops: float) -> tuple[float, float]:
    """
    Given a compute budget (FLOPs), return optimal (params, tokens).

    Chinchilla finding: D_opt ≈ 20 × N_opt
    Combined with C = 6 × N × D:
      C = 6 × N × 20N = 120 × N²
      N_opt = sqrt(C / 120)
      D_opt = 20 × N_opt
    """
    N_opt = math.sqrt(compute_flops / 120)
    D_opt = 20 * N_opt
    return N_opt, D_opt


def format_params(n):
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    return f"{n/1e3:.0f}K"


def format_tokens(n):
    if n >= 1e12:
        return f"{n/1e12:.1f}T"
    if n >= 1e9:
        return f"{n/1e9:.0f}B"
    if n >= 1e6:
        return f"{n/1e6:.0f}M"
    return f"{n:.0f}"


# --- Chinchilla optimal for different compute budgets ---------------------

print(f"\n--- Chinchilla-Optimal Model Sizes ---")
print(f"  Rule: Tokens ≈ 20 × Parameters")
print(f"  Formula: C = 6 × N × D, so N_opt = sqrt(C / 120)\n")
print(f"{'Compute (FLOPs)':>20s} {'Opt. Params':>12s} {'Opt. Tokens':>12s} {'Ratio D/N':>10s}")
print("-" * 58)

compute_budgets = [
    1e18,    # small experiment
    1e19,    # medium
    1e20,    # GPT-3 scale
    1e21,    # Chinchilla/LLaMA scale
    1e23,    # frontier models
    1e24,    # GPT-4 rumored scale
]

for C in compute_budgets:
    N, D = chinchilla_optimal(C)
    ratio = D / N
    print(f"{C:>20.0e} {format_params(N):>12s} {format_tokens(D):>12s} {ratio:>10.0f}×")


# --- Compare real models against Chinchilla optimality --------------------

print(f"\n--- Real Models vs Chinchilla Optimal ---")
real_models = [
    ("GPT-3",        175e9,   300e9,  3.7e23),
    ("Chinchilla",    70e9,  1.4e12,  5.8e23),
    ("LLaMA-1 7B",     7e9,  1.0e12,  4.2e22),
    ("LLaMA-1 65B",   65e9,  1.4e12,  5.4e23),
    ("LLaMA-2 70B",   70e9,  2.0e12,  8.4e23),
]

print(f"{'Model':17s} {'Params':>8s} {'Tokens':>8s} {'D/N':>6s} {'Status':>20s}")
print("-" * 65)
for name, N, D, C in real_models:
    ratio = D / N
    N_opt, D_opt = chinchilla_optimal(C)
    status = "under-trained" if D < D_opt * 0.5 else "near-optimal" if D < D_opt * 1.5 else "over-trained"
    print(f"{name:17s} {format_params(N):>8s} {format_tokens(D):>8s} {ratio:>5.0f}× {status:>20s}")

print(f"""
Key takeaways:
  - GPT-3 was massively under-trained (300B tokens for 175B params)
  - Chinchilla (70B params, 1.4T tokens) matched GPT-3 with 4× less compute
  - LLaMA pushed beyond Chinchilla ratios — "over-training" smaller models
    is compute-inefficient but gives smaller, cheaper-to-serve models
""")


# --- Loss prediction with scaling laws ------------------------------------

def predicted_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    """
    Chinchilla loss prediction formula:
      L(N, D) = A/N^alpha + B/D^beta + E

    Where:
      N = number of parameters
      D = number of training tokens
      A, B = fitted constants
      alpha, beta = scaling exponents
      E = irreducible entropy of natural language
    """
    return A / (N ** alpha) + B / (D ** beta) + E


print("--- Predicted loss for different configurations ---")
configs = [
    ("400M, 8B tokens",   400e6,  8e9),
    ("1B, 20B tokens",    1e9,    20e9),
    ("7B, 140B tokens",   7e9,    140e9),
    ("7B, 1T tokens",     7e9,    1e12),
    ("70B, 1.4T tokens",  70e9,   1.4e12),
]

print(f"{'Config':22s} {'FLOPs':>12s} {'Pred. Loss':>12s}")
print("-" * 50)
for name, N, D in configs:
    C = compute_flops(N, D)
    L = predicted_loss(N, D)
    print(f"{name:22s} {C:>12.2e} {L:>12.3f}")

print(f"\nIrreducible loss (E ≈ 1.69) is the theoretical minimum —")
print(f"even a perfect model can't predict natural language exactly.")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
Exercise 1: All-Reduce Simulation
-----------------------------------
  a) Create 8 "GPU" tensors, each of shape (1024,), with different
     random seeds (simulating different data shards).
  b) Implement all-reduce by averaging them.
  c) Verify that the result equals the gradient you'd get from
     processing all 8 shards on a single GPU (concatenate all data,
     compute mean).
  d) What is the effective batch size if each GPU processes 64 samples?

Exercise 2: FSDP Memory Calculator
------------------------------------
  a) Write a function that takes model_size_B (billions of params),
     num_gpus, and precision ("fp32" or "bf16") and returns per-GPU
     memory for DDP, ZeRO-1, ZeRO-2, and FSDP.
  b) Calculate memory requirements for a 13B parameter model on 8 GPUs.
  c) What is the minimum number of A100-80GB GPUs needed to train a
     70B model with FSDP in BF16?
  d) At what model size does DDP stop fitting on a single 80GB GPU?
     (Hint: solve for N in the DDP formula)

Exercise 3: Tensor Parallelism for Attention
----------------------------------------------
  a) Given a multi-head attention layer with d_model=4096 and n_heads=32,
     split it across 4 GPUs using column parallelism on the QKV projection.
  b) Each GPU should compute attention for its subset of heads (8 heads each).
  c) Verify that concatenating the outputs matches the full-model output.
  d) How much memory does each GPU save on the QKV weight matrices?

Exercise 4: Mixed Precision Memory Savings
--------------------------------------------
  a) Create a 4-layer MLP: Linear(2048, 8192) -> GELU -> Linear(8192, 2048),
     repeated twice. Count the parameters.
  b) Calculate total memory for: weights (FP32 master), gradients (BF16),
     optimizer states (FP32 momentum + variance), and activations (BF16)
     with batch_size=16, seq_len=2048.
  c) Compare total memory vs. a pure FP32 training setup.
  d) Run a forward pass with and without torch.amp.autocast on CPU.
     Verify the output dtypes match expectations.

Exercise 5: Chinchilla Optimal Budget Planner
-----------------------------------------------
  a) You have a compute budget of 1e21 FLOPs (about 1000 A100-hours).
     What is the Chinchilla-optimal model size and token count?
  b) Compare the predicted loss for:
     - The Chinchilla-optimal configuration
     - 2× larger model with half the tokens (same compute)
     - 2× more tokens with half the model size (same compute)
  c) Which is better: a 7B model trained on 2T tokens, or a 13B model
     trained on 1T tokens? (Compute the FLOPs and predicted loss for each.)
  d) At what compute budget does the optimal model size reach 1 trillion
     parameters? How many tokens would it need?
""")
