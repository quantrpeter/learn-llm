"""
============================================================================
 LESSON 2: Python & Deep Learning Tooling — PyTorch Fundamentals
============================================================================

 Goal: Get fluent with PyTorch — the library you'll use to build every
       component of your LLM from scratch.

 Topics covered:
   1. Tensors          — creating, shaping, indexing, dtypes, devices
   2. Tensor Operations — element-wise, matmul, broadcasting, reductions
   3. Autograd          — automatic differentiation and gradient computation
   4. torch.nn          — building neural networks from modules
   5. Training Loop     — forward → loss → backward → step → zero_grad
   6. Practical Tools   — mixed precision, gradient clipping, checkpoints

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites: Lesson 1 (math foundations — softmax, cross-entropy, gradients)
============================================================================
"""

import torch
import torch.nn as nn

torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — TENSORS (the building block)
# ═══════════════════════════════════════════════════════════════════════════
#
# A tensor is an n-dimensional array — the universal data structure in deep
# learning. Everything in a transformer is a tensor:
#   - Token embeddings: shape (batch, seq_len, d_model)
#   - Attention weights: shape (batch, n_heads, seq_len, seq_len)
#   - The entire vocabulary output: shape (batch, seq_len, vocab_size)
#
# If you understood lists-of-lists in Lesson 1, tensors are the same idea
# but with GPU acceleration and automatic differentiation built in.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: TENSORS")
print("=" * 60)


# --- 1.1 Creating tensors ------------------------------------------------

t_from_list = torch.tensor([1.0, 2.0, 3.0])
t_zeros = torch.zeros(3, 4)
t_ones = torch.ones(2, 3)
t_randn = torch.randn(2, 3)    # standard normal — this is how weights are initialized

print(f"\nFrom list:  {t_from_list}")
print(f"Zeros(3,4):\n{t_zeros}")
print(f"Ones(2,3):\n{t_ones}")
print(f"Randn(2,3):\n{t_randn}")


# --- 1.2 Tensor shapes ---------------------------------------------------
# The number of dimensions determines what the tensor represents.

vector = torch.randn(768)           # 1D: a single token embedding (like GPT-2's d_model=768)
matrix = torch.randn(10, 768)       # 2D: a sequence of 10 token embeddings
batch = torch.randn(4, 10, 768)     # 3D: a batch of 4 sequences

print(f"\n1D vector shape:  {vector.shape}")
print(f"2D matrix shape:  {matrix.shape}")
print(f"3D batch shape:   {batch.shape}")
print(f"Total elements in batch: {batch.numel()}")  # 4 * 10 * 768 = 30,720


# --- 1.3 Reshaping -------------------------------------------------------
# Reshaping is constant — you'll reshape tensors between every layer.
# view() and reshape() change the shape without copying data.

x = torch.arange(24)  # [0, 1, 2, ..., 23]
print(f"\nOriginal: {x.shape}")

x_2d = x.view(4, 6)
print(f"view(4, 6):\n{x_2d}")

x_3d = x.view(2, 3, 4)
print(f"view(2, 3, 4):\n{x_3d}")

# unsqueeze adds a dimension of size 1
# squeeze removes dimensions of size 1
v = torch.randn(5)
print(f"\nOriginal shape:              {v.shape}")
print(f"unsqueeze(0) → row vector:   {v.unsqueeze(0).shape}")
print(f"unsqueeze(1) → column vector:{v.unsqueeze(1).shape}")
print(f"After squeeze:               {v.unsqueeze(0).squeeze(0).shape}")


# --- 1.4 Indexing and slicing --------------------------------------------
# Same as NumPy — muscle memory you'll use constantly.

m = torch.arange(12).view(3, 4).float()
print(f"\nMatrix:\n{m}")
print(f"Row 0:          {m[0]}")
print(f"Col 2:          {m[:, 2]}")
print(f"Top-left 2×2:\n{m[:2, :2]}")
print(f"Last row:       {m[-1]}")


# --- 1.5 Device: CPU vs GPU ----------------------------------------------
# Tensors live on a specific device. GPU training requires moving data there.

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nAvailable device: {device}")

t = torch.randn(3, 3)
t_device = t.to(device)
print(f"Tensor device: {t_device.device}")


# --- 1.6 Dtypes ----------------------------------------------------------
# Precision directly affects memory, speed, and training stability.
# - float32: default, most precise
# - float16: half memory, faster on GPUs, but can overflow
# - bfloat16: same range as float32, less precision — the standard for LLM training

t32 = torch.randn(1000, 1000, dtype=torch.float32)
t16 = t32.to(torch.float16)
tbf = t32.to(torch.bfloat16)

print(f"\nfloat32 memory: {t32.element_size() * t32.numel() / 1e6:.1f} MB")
print(f"float16 memory: {t16.element_size() * t16.numel() / 1e6:.1f} MB")
print(f"bfloat16 memory:{tbf.element_size() * tbf.numel() / 1e6:.1f} MB")

# bfloat16 keeps the exponent range of float32 (no overflow), trades off mantissa bits
print(f"\nfloat16 max:  {torch.finfo(torch.float16).max}")
print(f"bfloat16 max: {torch.finfo(torch.bfloat16).max}")
print(f"float32 max:  {torch.finfo(torch.float32).max}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — TENSOR OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════
#
# In Lesson 1 we wrote matrix multiply by hand. PyTorch does it all on GPU,
# fused and optimized. The @ operator in PyTorch is the same Q @ K^T you'll
# write in the attention mechanism.
#
# The key insight: think in terms of shapes, not loops.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: TENSOR OPERATIONS")
print("=" * 60)

torch.manual_seed(42)


# --- 2.1 Element-wise operations -----------------------------------------

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(f"\na = {a}")
print(f"b = {b}")
print(f"a + b = {a + b}")
print(f"a * b = {a * b}")
print(f"exp(a) = {torch.exp(a)}")
print(f"log(a) = {torch.log(a)}")


# --- 2.2 Matrix multiply: the @ operator ---------------------------------

"""
WHY THIS MATTERS:
Every linear layer in a transformer is a matrix multiply.
- nn.Linear(768, 3072) computes: output = input @ W.T + bias
- Attention scores: scores = Q @ K.T / sqrt(d_k)

The @ operator is syntactic sugar for torch.matmul.
"""

Q = torch.randn(4, 8)     # 4 tokens, d_k=8
K = torch.randn(4, 8)     # 4 tokens, d_k=8

scores = Q @ K.T           # (4, 8) @ (8, 4) → (4, 4) attention score matrix
print(f"\nQ shape:      {Q.shape}")
print(f"K shape:      {K.shape}")
print(f"Q @ K^T shape: {scores.shape}")
print(f"Attention scores:\n{scores}")


# --- 2.3 Broadcasting rules ----------------------------------------------
# Broadcasting lets you operate on tensors with different shapes.
# PyTorch aligns dimensions from the right and expands size-1 dims.

"""
WHY THIS MATTERS:
- Adding bias to every token: (batch, seq_len, d_model) + (d_model,) broadcasts
- Masking attention: (batch, 1, 1, seq_len) mask broadcasts to (batch, heads, seq, seq)
"""

x = torch.randn(2, 3, 4)
bias = torch.randn(4)         # broadcasts across batch and seq dimensions
result = x + bias

print(f"\nx shape:      {x.shape}")
print(f"bias shape:   {bias.shape}")
print(f"result shape: {result.shape}")

row = torch.tensor([[1.0], [2.0], [3.0]])  # (3, 1)
col = torch.tensor([[10.0, 20.0, 30.0]])   # (1, 3)
print(f"\n(3,1) + (1,3) broadcasts to (3,3):\n{row + col}")


# --- 2.4 Reductions: sum, mean, max along dimensions ---------------------

x = torch.tensor([[1.0, 2.0, 3.0],
                   [4.0, 5.0, 6.0]])

print(f"\nx:\n{x}")
print(f"sum(all):      {x.sum()}")
print(f"sum(dim=0):    {x.sum(dim=0)}")    # sum across rows → shape (3,)
print(f"sum(dim=1):    {x.sum(dim=1)}")    # sum across cols → shape (2,)
print(f"mean(dim=1):   {x.mean(dim=1)}")
print(f"max(dim=1):    {x.max(dim=1)}")    # returns (values, indices)


# --- 2.5 Softmax and cross-entropy (connecting to Lesson 1) ---------------

"""
WHY THIS MATTERS:
In Lesson 1 we coded softmax from scratch. Now we use PyTorch's fused,
numerically stable versions. nn.CrossEntropyLoss takes raw logits — it
applies log_softmax + NLL internally (exactly the log-sum-exp trick from
Lesson 1, Part 5).
"""

logits = torch.tensor([[2.0, 1.0, 0.1, -1.0, 3.0]])  # (1, vocab_size=5)
target = torch.tensor([2])                              # correct token index

probs = torch.softmax(logits, dim=-1)
loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(logits, target)

print(f"\nLogits:   {logits}")
print(f"Softmax:  {probs}")
print(f"Target:   {target.item()}")
print(f"CE Loss:  {loss.item():.4f}")
print(f"Perplexity: {torch.exp(loss).item():.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — AUTOGRAD (automatic differentiation)
# ═══════════════════════════════════════════════════════════════════════════
#
# In Lesson 1 we computed gradients numerically (f(x+h) - f(x-h)) / 2h.
# PyTorch autograd does this EXACTLY via the chain rule — no approximation.
#
# The deal: set requires_grad=True, do math, call .backward(), read .grad.
# That's the entire training loop's gradient computation.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3: AUTOGRAD")
print("=" * 60)

torch.manual_seed(42)


# --- 3.1 Basic gradient computation --------------------------------------

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2 + 2 * x + 1     # dy/dx = 2x + 2 = 8 at x=3

y.backward()
print(f"\ny = x² + 2x + 1")
print(f"x = {x.item()}")
print(f"dy/dx (analytical) = 2*3 + 2 = 8")
print(f"dy/dx (autograd)   = {x.grad.item()}")


# --- 3.2 Forward → loss → backward → grad --------------------------------

"""
WHY THIS MATTERS:
This is the exact pattern that trains every neural network:
  1. Forward pass: compute prediction
  2. Loss: measure how wrong
  3. Backward: compute gradients of loss w.r.t. all weights
  4. Update: nudge weights by -lr * grad
"""

w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

x_data = torch.tensor(2.0)
y_true = torch.tensor(5.0)

y_pred = w * x_data + b             # forward pass
loss = (y_pred - y_true) ** 2       # MSE loss

loss.backward()                      # compute gradients

print(f"\nSimple linear model: y = w*x + b")
print(f"w={w.item()}, b={b.item()}, x={x_data.item()}, y_true={y_true.item()}")
print(f"y_pred = {y_pred.item()}, loss = {loss.item()}")
print(f"∂loss/∂w = {w.grad.item()}")
print(f"∂loss/∂b = {b.grad.item()}")


# --- 3.3 Computational graph and detach -----------------------------------

"""
WHY THIS MATTERS:
- torch.no_grad() is used during inference (generation) — no need to track
  gradients, saves memory and compute.
- .detach() breaks the graph — used when you need a tensor's value but don't
  want gradients to flow through it (e.g., target networks in RL, KL penalty
  reference in RLHF).
"""

x = torch.tensor(2.0, requires_grad=True)
y = x * 3
z = y.detach() * 2    # z depends on y's VALUE but not its gradient path

print(f"\nx = 2.0 (requires_grad=True)")
print(f"y = x * 3 = {y.item()}")
print(f"z = detach(y) * 2 = {z.item()}")
print(f"z.requires_grad = {z.requires_grad}")

with torch.no_grad():
    fast = x * 100
    print(f"\nInside no_grad: x * 100 = {fast.item()}")
    print(f"fast.requires_grad = {fast.requires_grad}")


# --- 3.4 Gradient accumulation --------------------------------------------

"""
WHY THIS MATTERS:
When training a 7B parameter LLM, you can't fit a large batch in GPU memory.
Instead, you process small mini-batches and ACCUMULATE gradients before
stepping. This simulates a larger effective batch size.

Key: PyTorch ADDS gradients by default on .backward(). You must call
zero_grad() when you're ready to start fresh.
"""

w = torch.tensor(1.0, requires_grad=True)

# Simulate 4 micro-batches that accumulate into one big gradient
print(f"\nGradient accumulation over 4 micro-batches:")
accumulation_steps = 4
for step in range(accumulation_steps):
    x_i = torch.tensor(float(step + 1))
    loss_i = (w * x_i - 5.0) ** 2
    loss_i.backward()                    # gradients ADD to w.grad
    print(f"  Step {step}: loss={loss_i.item():.2f}, accumulated grad={w.grad.item():.2f}")

# In practice you'd now do: optimizer.step() then optimizer.zero_grad()
print(f"Final accumulated gradient: {w.grad.item():.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — BUILDING A NEURAL NETWORK WITH torch.nn
# ═══════════════════════════════════════════════════════════════════════════
#
# torch.nn provides the building blocks you'll compose into a transformer:
#   - nn.Linear  → the "linear projection" in attention, FFN, LM head
#   - nn.GELU    → the activation function in modern LLMs (GPT-2 onwards)
#   - nn.Module  → the base class for every model you'll build
#
# A full transformer is just ~15 nn.Module classes composed together.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 4: BUILDING NEURAL NETWORKS (torch.nn)")
print("=" * 60)

torch.manual_seed(42)


# --- 4.1 nn.Linear -------------------------------------------------------

"""
WHY THIS MATTERS:
nn.Linear IS the weight matrix multiply + bias from Lesson 1.
  output = input @ W.T + b
A GPT-style transformer uses nn.Linear for:
  - Q, K, V projections in attention
  - Output projection in attention
  - Both layers of the feed-forward network (FFN)
  - The final "language model head" that predicts the next token
"""

linear = nn.Linear(in_features=4, out_features=3)

x = torch.randn(2, 4)     # batch of 2 vectors, each dim 4
y = linear(x)              # (2, 4) @ (4, 3).T → err, internally it's (2, 4) @ (3, 4).T = (2, 3)

print(f"\nnn.Linear(4, 3)")
print(f"Weight shape: {linear.weight.shape}")    # (out, in) — PyTorch convention
print(f"Bias shape:   {linear.bias.shape}")
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")


# --- 4.2 Activation functions: ReLU vs GELU -------------------------------

"""
WHY THIS MATTERS:
ReLU: max(0, x) — simple, fast, used in older models.
GELU: x * Φ(x) — smooth approximation, used in GPT-2, BERT, LLaMA.
GELU lets small negative values through (unlike ReLU's hard cutoff),
which improves training stability.
"""

x = torch.linspace(-3, 3, 7)
relu = nn.ReLU()
gelu = nn.GELU()

print(f"\nx:       {x.tolist()}")
print(f"ReLU(x): {relu(x).tolist()}")
print(f"GELU(x): {[f'{v:.4f}' for v in gelu(x).tolist()]}")


# --- 4.3 nn.Sequential ---------------------------------------------------

simple_ffn = nn.Sequential(
    nn.Linear(8, 32),
    nn.GELU(),
    nn.Linear(32, 8),
)

x = torch.randn(2, 8)
y = simple_ffn(x)
print(f"\nSequential FFN: Linear(8→32) → GELU → Linear(32→8)")
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")


# --- 4.4 nn.Module: building a custom model class -------------------------

class TwoLayerMLP(nn.Module):
    """
    A simple 2-layer MLP (multi-layer perceptron).

    WHY THIS MATTERS:
    The feed-forward network inside every transformer block IS a 2-layer MLP:
      FFN(x) = Linear2(GELU(Linear1(x)))
    This exact pattern appears in GPT, BERT, LLaMA, and every transformer.
    """

    def __init__(self, d_model: int, d_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


mlp = TwoLayerMLP(d_model=64, d_hidden=256)
x = torch.randn(2, 10, 64)   # (batch=2, seq_len=10, d_model=64)
y = mlp(x)

print(f"\nTwoLayerMLP(d_model=64, d_hidden=256)")
print(f"Input shape:  {x.shape}")
print(f"Output shape: {y.shape}")


# --- 4.5 Parameter counting ----------------------------------------------

"""
WHY THIS MATTERS:
"GPT-3 has 175 billion parameters" — that's the sum of all weight tensors.
Knowing how to count parameters tells you memory requirements:
  float32: 4 bytes/param → 175B × 4 = 700 GB
  float16: 2 bytes/param → 175B × 2 = 350 GB
"""

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


print(f"\nParameter count for TwoLayerMLP(64, 256):")
for name, param in mlp.named_parameters():
    print(f"  {name:10s} shape={str(param.shape):15s} params={param.numel():,}")
print(f"  {'TOTAL':10s} {' ':15s} params={count_parameters(mlp):,}")

# fc1: 64*256 + 256 = 16,640
# fc2: 256*64 + 64  = 16,448
# total: 33,088


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — THE TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════
#
# Every model — from a 1-parameter linear regression to a 175B-parameter
# GPT-3 — trains with the same loop:
#
#   for each batch:
#       prediction = model(input)          # forward pass
#       loss = loss_fn(prediction, target)  # measure error
#       loss.backward()                     # compute gradients
#       optimizer.step()                    # update weights
#       optimizer.zero_grad()               # reset gradients
#
# That's it. Everything else is optimization (learning rate schedules,
# gradient clipping, mixed precision, distributed training).
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 5: THE TRAINING LOOP")
print("=" * 60)

torch.manual_seed(42)


# --- 5.1 Synthetic dataset: y = 2x + 3 + noise ---------------------------

N = 200
x_train = torch.randn(N, 1)
y_train = 2.0 * x_train + 3.0 + 0.1 * torch.randn(N, 1)

print(f"\nDataset: {N} samples of y = 2x + 3 + noise")
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")


# --- 5.2 Model, loss function, optimizer ----------------------------------

model = nn.Linear(1, 1)

"""
WHY THIS MATTERS:
AdamW is the standard optimizer for LLM training (GPT-2, GPT-3, LLaMA).
It combines:
  - Momentum (smooths out noisy gradients)
  - Adaptive learning rates per-parameter
  - Decoupled weight decay (regularization to prevent overfitting)
"""

optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
loss_fn = nn.MSELoss()


# --- 5.3 Training loop ----------------------------------------------------

print(f"\nTraining linear model on y = 2x + 3:")
num_epochs = 300

for epoch in range(num_epochs):
    y_pred = model(x_train)             # forward
    loss = loss_fn(y_pred, y_train)     # loss
    loss.backward()                      # backward
    optimizer.step()                     # update weights
    optimizer.zero_grad()                # reset gradients

    if epoch % 50 == 0 or epoch == num_epochs - 1:
        w = model.weight.item()
        b = model.bias.item()
        print(f"  Epoch {epoch:3d} | Loss: {loss.item():.6f} | w={w:.4f}, b={b:.4f}")

w_final = model.weight.item()
b_final = model.bias.item()
print(f"\nLearned: y = {w_final:.4f}x + {b_final:.4f}")
print(f"True:    y = 2.0000x + 3.0000")


# --- 5.4 Loss curve (text-based) -----------------------------------------

torch.manual_seed(42)
model2 = nn.Linear(1, 1)
optimizer2 = torch.optim.AdamW(model2.parameters(), lr=0.05)
losses = []

for epoch in range(300):
    y_pred = model2(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer2.step()
    optimizer2.zero_grad()
    losses.append(loss.item())

print(f"\nLoss curve (text plot):")
num_bars = 20
for i in range(0, 300, 30):
    bar_len = int(min(losses[i], 10.0) / 10.0 * num_bars)
    bar = "█" * bar_len + "░" * (num_bars - bar_len)
    print(f"  Epoch {i:3d} | {bar} | {losses[i]:.4f}")
bar_len = int(min(losses[-1], 10.0) / 10.0 * num_bars)
bar = "█" * bar_len + "░" * (num_bars - bar_len)
print(f"  Epoch 299 | {bar} | {losses[-1]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — PRACTICAL TOOLS
# ═══════════════════════════════════════════════════════════════════════════
#
# These are the engineering tricks that make large-scale training possible.
# You won't need them for toy models, but they're essential for LLMs:
#   - Mixed precision: 2× faster, half the memory
#   - Gradient clipping: prevents exploding gradients
#   - Checkpointing: resume training after crashes
#   - Reproducibility: same seed → same results
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 6: PRACTICAL TOOLS")
print("=" * 60)

torch.manual_seed(42)


# --- 6.1 Mixed precision: torch.amp --------------------------------------

"""
WHY THIS MATTERS:
LLM training would be 2× slower and use 2× more memory without mixed
precision. The idea: do most computation in float16/bfloat16, but keep
a float32 "master copy" of weights for the optimizer update.

torch.amp.autocast chooses the right dtype per-operation automatically.
GradScaler prevents underflow in float16 (not needed for bfloat16).
"""

print(f"\n--- Mixed Precision ---")
model_amp = nn.Linear(256, 256)
x_amp = torch.randn(8, 256)

# autocast: automatically runs ops in lower precision where safe
with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
    y_amp = model_amp(x_amp)
    print(f"Input dtype:  {x_amp.dtype}")
    print(f"Output dtype: {y_amp.dtype}")
    print(f"Weight dtype inside autocast: {model_amp.weight.dtype}")
    # weights stay float32 — only the COMPUTATION is in bfloat16

print(f"Weight dtype outside autocast: {model_amp.weight.dtype}")

# GradScaler — only needed for float16 on GPU, not bfloat16
# scaler = torch.amp.GradScaler()
# scaled_loss = scaler.scale(loss)
# scaled_loss.backward()
# scaler.step(optimizer)
# scaler.update()
print(f"\nGradScaler: used with float16 to prevent gradient underflow")
print(f"Not needed for bfloat16 (which most modern hardware supports)")


# --- 6.2 Gradient clipping -----------------------------------------------

"""
WHY THIS MATTERS:
During LLM training, gradients can occasionally spike (especially early on
or with long sequences). Gradient clipping caps the total gradient norm,
preventing catastrophic weight updates.

Every LLM training run uses this — typically max_norm=1.0.
"""

print(f"\n--- Gradient Clipping ---")
model_clip = nn.Linear(10, 10)
x_clip = torch.randn(1, 10) * 100   # large input → large gradients
y_clip = model_clip(x_clip)
loss_clip = y_clip.sum()
loss_clip.backward()

grad_norm_before = torch.nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm=float('inf'))
print(f"Gradient norm before clipping: {grad_norm_before:.4f}")

# Reset and redo to actually clip
model_clip.zero_grad()
y_clip = model_clip(x_clip)
loss_clip = y_clip.sum()
loss_clip.backward()

grad_norm_after = torch.nn.utils.clip_grad_norm_(model_clip.parameters(), max_norm=1.0)
print(f"Gradient norm after clipping to 1.0: {grad_norm_after:.4f}")

# Verify gradients were actually scaled down
total_norm = sum(p.grad.norm() ** 2 for p in model_clip.parameters()) ** 0.5
print(f"Actual gradient norm now: {total_norm:.4f}")


# --- 6.3 Saving and loading checkpoints ----------------------------------

"""
WHY THIS MATTERS:
Training a large LLM takes weeks. If it crashes at step 100,000 you need to
resume from a checkpoint, not start over. Checkpoints save:
  - model weights (state_dict)
  - optimizer state (momentum buffers, adaptive learning rates)
  - training step / epoch
"""

print(f"\n--- Checkpoints ---")
model_save = nn.Linear(4, 2)

checkpoint = {
    "model_state_dict": model_save.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 42,
    "loss": 0.0012,
}

import tempfile, os
with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
    ckpt_path = f.name
    torch.save(checkpoint, ckpt_path)

loaded = torch.load(ckpt_path, weights_only=False)
print(f"Saved checkpoint to: {os.path.basename(ckpt_path)}")
print(f"Loaded epoch: {loaded['epoch']}")
print(f"Loaded loss:  {loaded['loss']}")

model_loaded = nn.Linear(4, 2)
model_loaded.load_state_dict(loaded["model_state_dict"])

# Verify weights match
match = torch.equal(model_save.weight, model_loaded.weight)
print(f"Weights match after load: {match}")

os.unlink(ckpt_path)


# --- 6.4 Reproducibility -------------------------------------------------

"""
WHY THIS MATTERS:
Debugging training issues requires reproducibility. If you can't reproduce
a crash, you can't fix it. Setting all seeds ensures identical behavior
across runs.
"""

print(f"\n--- Reproducibility ---")

torch.manual_seed(123)
a = torch.randn(3)

torch.manual_seed(123)
b = torch.randn(3)

print(f"Same seed → same tensor: {torch.equal(a, b)}")
print(f"  a = {a}")
print(f"  b = {b}")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
Exercise 1: 3D Tensor Reductions
---------------------------------
Create a tensor of shape (2, 3, 4) using torch.randn.
  a) Compute the mean along dim=1 — what shape do you get?
  b) Compute the sum along dim=-1 — what shape do you get?
  c) Compute max along dim=0 — what shape and what does it represent?
Hint: think about what each dimension means if this were
      (batch=2, seq_len=3, d_model=4).

Exercise 2: Manual Softmax vs F.softmax
-----------------------------------------
Implement softmax manually using PyTorch ops:
  def my_softmax(logits):
      # Subtract max for numerical stability (Lesson 1!)
      # Compute exp, then divide by sum
      ...
Compare your output against torch.nn.functional.softmax(logits, dim=-1).
They should match to at least 6 decimal places.
Test with: logits = torch.tensor([2.0, 1.0, 0.1, -1.0, 3.0])

Exercise 3: Custom MLP with Parameter Counting
------------------------------------------------
Build a 3-layer MLP class (nn.Module) with:
  - Linear(784, 256) → GELU → Linear(256, 128) → GELU → Linear(128, 10)
  a) Count total parameters. Verify by hand:
     Layer 1: 784*256 + 256 = ?
     Layer 2: 256*128 + 128 = ?
     Layer 3: 128*10  + 10  = ?
  b) How much memory (in MB) would this model use in float32? float16?

Exercise 4: Train on y = sin(x)
---------------------------------
  a) Generate 500 training points: x in [-π, π], y = sin(x) + noise
  b) Build a small MLP: Linear(1, 64) → GELU → Linear(64, 64) → GELU → Linear(64, 1)
  c) Train for 500 steps with AdamW (lr=0.01)
  d) Print loss every 100 steps
  e) After training, test on x = [0, π/2, π] — does it predict [0, 1, 0]?

Exercise 5: Gradient Accumulation
-----------------------------------
Implement gradient accumulation over 4 mini-batches:
  a) Create a dataset of 100 points: y = 3x - 1 + noise
  b) Build a simple nn.Linear(1, 1) model
  c) Split data into mini-batches of 25
  d) Accumulate gradients over 4 mini-batches, THEN step
  e) Train for 50 "effective" steps, print loss every 10 steps
  f) Verify: the result should be similar to training with batch_size=100
     (gradient accumulation simulates larger batches)
""")
