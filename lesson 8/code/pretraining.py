"""
============================================================================
 LESSON 8: Pre-Training Your LLM
============================================================================

 Goal: Train a language model from scratch — understand every component
       of the pre-training process from objective to checkpoint.

 Topics covered:
   1. Training Objective  — next-token prediction (causal language modeling)
   2. Loss Function       — cross-entropy over vocabulary, perplexity
   3. Optimizer           — AdamW internals, comparison with SGD and Adam
   4. Learning Rate Sched — warmup + cosine decay, implemented from scratch
   5. Training Loop       — complete loop with a tiny GPT on synthetic data
   6. Checkpointing       — save/load model + optimizer + step, resume

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites:
   - Lesson 2 (PyTorch fundamentals, autograd, training loop basics)
   - Lesson 6 (transformer architecture, GPT model)
   - Lesson 7 (data pipeline concepts)
============================================================================
"""

import torch
import torch.nn as nn
import math
import os
import tempfile

torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — TRAINING OBJECTIVE (Next-Token Prediction)
# ═══════════════════════════════════════════════════════════════════════════
#
# The entire pre-training process boils down to one task:
#   Given tokens [t0, t1, t2, ..., t_{n-1}], predict [t1, t2, t3, ..., t_n].
#
# This is called CAUSAL LANGUAGE MODELING (CLM). At every position, the
# model predicts the next token. The causal mask (Lesson 5) ensures each
# position can only see tokens before it — no peeking at the future.
#
# This single objective teaches the model grammar, facts, reasoning,
# and even code — all from predicting the next token.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: TRAINING OBJECTIVE — Next-Token Prediction")
print("=" * 60)


# --- 1.1 The input→target shift -------------------------------------------

"""
WHY THIS MATTERS:
A sequence of N tokens gives us N-1 training examples for free.
The model learns from EVERY position simultaneously, not just the last one.
This is far more data-efficient than training on (prompt, answer) pairs.
"""

vocab = ["<pad>", "the", "cat", "sat", "on", "mat", ".", "dog", "ran", "in", "park"]
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for i, w in enumerate(vocab)}

sentence = "the cat sat on the mat ."
token_ids = torch.tensor([word2id[w] for w in sentence.split()])

print(f"\nSentence: \"{sentence}\"")
print(f"Token IDs: {token_ids.tolist()}")

# The key insight: shift by one position
inputs = token_ids[:-1]    # [the, cat, sat, on, the, mat]
targets = token_ids[1:]    # [cat, sat, on, the, mat, .]

print(f"\nInput tokens:  {[id2word[i.item()] for i in inputs]}")
print(f"Target tokens: {[id2word[i.item()] for i in targets]}")

print(f"\nTraining pairs (next-token prediction):")
for i in range(len(inputs)):
    context = [id2word[inputs[j].item()] for j in range(i + 1)]
    target = id2word[targets[i].item()]
    print(f"  Position {i}: {str(context):40s} → predict \"{target}\"")


# --- 1.2 Batched sequences ------------------------------------------------

"""
WHY THIS MATTERS:
In practice, we don't train on one sentence at a time. We pack many
sentences into fixed-length blocks (from Lesson 7's data pipeline)
and train on entire blocks. This maximizes GPU utilization.
"""

print(f"\n--- Batched training data ---")
corpus_text = "the cat sat on the mat . the dog ran in the park ."
corpus_ids = torch.tensor([word2id[w] for w in corpus_text.split()])

seq_len = 6
batch_size = 2

torch.manual_seed(42)
starts = torch.randint(0, len(corpus_ids) - seq_len, (batch_size,))
x_batch = torch.stack([corpus_ids[s:s + seq_len] for s in starts])
y_batch = torch.stack([corpus_ids[s + 1:s + seq_len + 1] for s in starts])

print(f"Corpus: {corpus_text}")
print(f"Corpus IDs: {corpus_ids.tolist()}")
print(f"\nBatch inputs  (shape {list(x_batch.shape)}):")
for i in range(batch_size):
    words = [id2word[t.item()] for t in x_batch[i]]
    print(f"  Sample {i}: {x_batch[i].tolist()} → {words}")
print(f"Batch targets (shape {list(y_batch.shape)}):")
for i in range(batch_size):
    words = [id2word[t.item()] for t in y_batch[i]]
    print(f"  Sample {i}: {y_batch[i].tolist()} → {words}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — LOSS FUNCTION (Cross-Entropy over Vocabulary)
# ═══════════════════════════════════════════════════════════════════════════
#
# In Lesson 1 we implemented cross-entropy from scratch:
#   loss = -log(p_correct)
#
# For LLM training, the model outputs logits of shape (B, T, V) where
# V is the vocabulary size. At each of the B*T positions, we compute
# cross-entropy between the predicted distribution and the true token.
#
# PyTorch's nn.CrossEntropyLoss handles all of this, including the
# log-sum-exp trick for numerical stability (Lesson 1, Part 5).
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: LOSS FUNCTION — Cross-Entropy")
print("=" * 60)

torch.manual_seed(42)


# --- 2.1 Loss on a single position ----------------------------------------

vocab_size = len(vocab)
logits_single = torch.randn(1, vocab_size)     # model output for one position
target_single = torch.tensor([2])               # correct token is "sat" (index 2)

probs = torch.softmax(logits_single, dim=-1)
manual_loss = -torch.log(probs[0, target_single[0]])

loss_fn = nn.CrossEntropyLoss()
pytorch_loss = loss_fn(logits_single, target_single)

print(f"\n--- Single position loss ---")
print(f"Logits: {logits_single[0].tolist()}")
print(f"Softmax probs: {[f'{p:.4f}' for p in probs[0].tolist()]}")
print(f"Target token: \"{id2word[2]}\" (index 2)")
print(f"P(correct): {probs[0, 2].item():.4f}")
print(f"Manual loss -log(P): {manual_loss.item():.4f}")
print(f"PyTorch CE loss:     {pytorch_loss.item():.4f}")
print(f"Match: {torch.allclose(manual_loss, pytorch_loss)}")


# --- 2.2 Loss over a full sequence ----------------------------------------

"""
WHY THIS MATTERS:
The training loss is averaged over ALL positions in ALL sequences.
nn.CrossEntropyLoss expects:
  - input:  (N, C) where N = batch*seq_len, C = vocab_size
  - target: (N,) — the correct token index at each position

We reshape (B, T, V) → (B*T, V) and (B, T) → (B*T) before passing to the loss.
"""

B, T, V = 2, 6, vocab_size
logits_seq = torch.randn(B, T, V)
targets_seq = torch.randint(0, V, (B, T))

# Reshape for CrossEntropyLoss: flatten batch and sequence dimensions
loss_seq = loss_fn(logits_seq.view(B * T, V), targets_seq.view(B * T))
perplexity = torch.exp(loss_seq)

print(f"\n--- Sequence loss ---")
print(f"Logits shape:  {list(logits_seq.shape)} → reshaped to ({B*T}, {V})")
print(f"Targets shape: {list(targets_seq.shape)} → reshaped to ({B*T},)")
print(f"Average CE loss: {loss_seq.item():.4f}")
print(f"Perplexity:      {perplexity.item():.2f}")
print(f"\nInterpretation: random logits on vocab_size={V} →")
print(f"  expected loss ≈ ln({V}) = {math.log(V):.4f}")
print(f"  expected perplexity ≈ {V}")


# --- 2.3 What perplexity means --------------------------------------------

print(f"\n--- Perplexity benchmarks ---")
for ppl, desc in [
    (50000, "Untrained (random, vocab=50k)"),
    (100, "After a few hours of training"),
    (30, "Decent small model"),
    (10, "Good model (GPT-2 level on WikiText)"),
    (5, "Strong model"),
]:
    loss_val = math.log(ppl)
    print(f"  PPL={ppl:>6d} → loss={loss_val:.2f}  ({desc})")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — OPTIMIZER (AdamW)
# ═══════════════════════════════════════════════════════════════════════════
#
# AdamW is the standard optimizer for LLM pre-training.
# Used by: GPT-2, GPT-3, LLaMA, Mistral — every major LLM.
#
# AdamW combines three ideas:
#   1. Momentum (β1): smooth out noisy gradients by averaging
#   2. Adaptive LR (β2): scale learning rate per-parameter by gradient variance
#   3. Decoupled weight decay (λ): shrink weights directly (not through gradient)
#
# The "W" in AdamW means weight decay is DECOUPLED from the gradient update.
# Original Adam with L2 regularization ≠ AdamW. The distinction matters
# for training stability at scale.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3: OPTIMIZER — AdamW")
print("=" * 60)

torch.manual_seed(42)


# --- 3.1 AdamW update rule (manual implementation) -------------------------

"""
WHY THIS MATTERS:
Understanding the optimizer internals helps you debug training instabilities.
If loss spikes, you need to know whether the issue is momentum, adaptive LR,
or weight decay. Here we implement one AdamW step from scratch.

AdamW update for parameter θ at step t:
  g = ∂loss/∂θ                                    (gradient)
  m = β1 * m + (1 - β1) * g                       (first moment = mean)
  v = β2 * v + (1 - β2) * g²                      (second moment = variance)
  m̂ = m / (1 - β1^t)                              (bias correction)
  v̂ = v / (1 - β2^t)                              (bias correction)
  θ = θ - lr * m̂ / (√v̂ + ε)                      (Adam step)
  θ = θ - lr * λ * θ                               (weight decay, DECOUPLED)
"""

print(f"\n--- Manual AdamW step ---")

theta = torch.tensor([1.5, -0.8, 2.3])
grad = torch.tensor([0.3, -0.1, 0.5])

m = torch.zeros_like(theta)   # first moment
v = torch.zeros_like(theta)   # second moment
beta1, beta2, eps = 0.9, 0.999, 1e-8
lr, weight_decay = 1e-3, 0.01
t = 1

# Step 1: Update moments
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad ** 2
print(f"After moment update:")
print(f"  m (first moment):  {m.tolist()}")
print(f"  v (second moment): {v.tolist()}")

# Step 2: Bias correction
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
print(f"After bias correction:")
print(f"  m_hat: {m_hat.tolist()}")
print(f"  v_hat: {v_hat.tolist()}")

# Step 3: Adam update + decoupled weight decay
theta_before = theta.clone()
theta = theta - lr * (m_hat / (v_hat.sqrt() + eps))   # Adam step
theta = theta - lr * weight_decay * theta_before       # weight decay (decoupled!)

print(f"\nParameter before: {theta_before.tolist()}")
print(f"Parameter after:  {[f'{x:.6f}' for x in theta.tolist()]}")
print(f"Change:           {[f'{(a-b):.6f}' for a, b in zip(theta, theta_before)]}")


# --- 3.2 Key hyperparameters -----------------------------------------------

print(f"\n--- Standard AdamW hyperparameters ---")
print(f"""
  β1 = 0.9      First moment decay (momentum).
                 Higher → smoother updates, slower to respond.
                 GPT-3 used β1=0.9.

  β2 = 0.95-0.999  Second moment decay (adaptive LR).
                    Higher → more stable but slower adaptation.
                    GPT-3 used β2=0.95. LLaMA used β2=0.95.

  ε  = 1e-8     Prevents division by zero. Rarely changed.

  lr = 1e-4 to 6e-4  Peak learning rate. Depends on model size.
                      Larger models need SMALLER learning rates.

  weight_decay = 0.01-0.1  Shrinks weights to prevent overfitting.
                            Applied to all params EXCEPT biases and norms.
""")


# --- 3.3 Comparison: SGD vs Adam vs AdamW ---------------------------------

"""
WHY THIS MATTERS:
SGD is simple but slow on complex loss landscapes.
Adam adapts per-parameter — much faster convergence.
AdamW adds proper weight decay — better generalization.
"""

print(f"--- Optimizer comparison on y = 2x + 3 ---")

torch.manual_seed(42)
N = 200
x_data = torch.randn(N, 1)
y_data = 2.0 * x_data + 3.0 + 0.1 * torch.randn(N, 1)
mse = nn.MSELoss()

class SmallMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)


x_cmp = torch.linspace(-3, 3, N).unsqueeze(1)
y_cmp = torch.sin(x_cmp) + 0.1 * torch.randn_like(x_cmp)

results = {}
for name, OptimizerClass, kwargs in [
    ("SGD",   torch.optim.SGD,   {"lr": 0.01}),
    ("Adam",  torch.optim.Adam,  {"lr": 0.01}),
    ("AdamW", torch.optim.AdamW, {"lr": 0.01, "weight_decay": 0.01}),
]:
    torch.manual_seed(42)
    model_cmp = SmallMLP()
    opt = OptimizerClass(model_cmp.parameters(), **kwargs)
    losses = []
    for step in range(200):
        pred = model_cmp(x_cmp)
        loss = mse(pred, y_cmp)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    results[name] = losses
    print(f"  {name:5s}: final loss={losses[-1]:.6f}")

print(f"\nLoss at step 50:")
for name in ["SGD", "Adam", "AdamW"]:
    print(f"  {name:5s}: {results[name][49]:.6f}")
print(f"(Adam/AdamW converge faster on non-convex MLP loss landscapes)")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — LEARNING RATE SCHEDULE (Warmup + Cosine Decay)
# ═══════════════════════════════════════════════════════════════════════════
#
# Every LLM uses a learning rate schedule:
#   Phase 1 — WARMUP: linearly increase LR from 0 to max_lr
#   Phase 2 — COSINE DECAY: smoothly decrease LR from max_lr to min_lr
#
# Why warmup? At the start, model weights are random and gradients are
# noisy/large. A high learning rate would cause instability. Warmup lets
# the optimizer's moment estimates "warm up" to reasonable values.
#
# Why cosine? It provides a smooth, gradual decay — no abrupt jumps.
# Empirically works better than step-decay or exponential schedules.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 4: LEARNING RATE SCHEDULE")
print("=" * 60)


# --- 4.1 Warmup + cosine decay from scratch --------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int,
           max_lr: float, min_lr: float) -> float:
    """
    Compute learning rate at a given step using warmup + cosine decay.

    This exact schedule is used by GPT-3, LLaMA, and most modern LLMs.
    """
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


warmup_steps = 100
max_steps = 1000
max_lr = 3e-4
min_lr = 3e-5

print(f"\nSchedule config:")
print(f"  warmup_steps = {warmup_steps}")
print(f"  max_steps    = {max_steps}")
print(f"  max_lr       = {max_lr}")
print(f"  min_lr       = {min_lr}")

# Show LR at key points
print(f"\nLR at key steps:")
for s in [0, 25, 50, 100, 200, 400, 600, 800, 1000]:
    lr_val = get_lr(s, warmup_steps, max_steps, max_lr, min_lr)
    phase = "warmup" if s < warmup_steps else "cosine"
    print(f"  Step {s:4d}: lr = {lr_val:.6f}  ({phase})")


# --- 4.2 ASCII plot of the schedule ----------------------------------------

print(f"\nLearning Rate Schedule (ASCII plot):")
print(f"  max_lr={max_lr:.1e} |")

plot_width = 50
plot_height = 15
schedule = [get_lr(s, warmup_steps, max_steps, max_lr, min_lr) for s in range(max_steps)]

for row in range(plot_height, 0, -1):
    threshold = min_lr + (max_lr - min_lr) * row / plot_height
    line = ""
    for col in range(plot_width):
        step_idx = int(col * max_steps / plot_width)
        if schedule[step_idx] >= threshold:
            line += "█"
        else:
            line += " "
    lr_label = f"{threshold:.1e}" if row in [plot_height, plot_height // 2, 1] else "         "
    print(f"  {lr_label} |{line}|")

print(f"  {'':9s} +{'─' * plot_width}+")
step_labels = "".join(f"{int(i * max_steps / 5):<10d}" for i in range(6))
print(f"  {'':10s}{step_labels}")
print(f"  {'':10s}{'step →':^{plot_width}s}")

warmup_pos = int(warmup_steps / max_steps * plot_width)
print(f"  {'':10s}{' ' * warmup_pos}↑ warmup ends (step {warmup_steps})")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — TRAINING LOOP (Complete Pre-Training)
# ═══════════════════════════════════════════════════════════════════════════
#
# Here we put everything together: build a tiny GPT model and train it
# on synthetic text data. This is the exact same process used to train
# GPT-2, GPT-3, and LLaMA — just at a much smaller scale.
#
# The loop:
#   for step in range(max_steps):
#       x, y = get_batch(data)                  # from data pipeline
#       logits = model(x)                        # forward pass
#       loss = F.cross_entropy(logits, y)        # loss computation
#       loss.backward()                          # backward pass
#       clip_grad_norm_(model.parameters(), 1.0) # gradient clipping
#       optimizer.step()                         # weight update
#       optimizer.zero_grad()                    # reset gradients
#       update_lr(scheduler)                     # LR schedule
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 5: TRAINING LOOP")
print("=" * 60)

torch.manual_seed(42)


# --- 5.1 Tiny GPT model (same architecture as Lesson 6) -------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len))
                  .unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_head)
        scores = scores.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        out = weights @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_seq_len):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """
    Minimal GPT using the architecture from Lesson 6.
    Token embedding + positional embedding → N transformer blocks →
    LayerNorm → LM head → logits.
    """

    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.lm_head(x)


# --- 5.2 Synthetic training corpus ----------------------------------------

raw_text = (
    "the cat sat on the mat . "
    "the dog ran in the park . "
    "the bird flew over the tree . "
    "a fish swam in the lake . "
    "the cat chased the dog . "
    "the dog barked at the bird . "
)
raw_text = raw_text * 80  # repeat to create a larger corpus

words = raw_text.split()
train_vocab = sorted(set(words))
train_w2id = {w: i for i, w in enumerate(train_vocab)}
train_id2w = {i: w for i, w in enumerate(train_vocab)}
corpus_tokens = torch.tensor([train_w2id[w] for w in words])

VOCAB_SIZE = len(train_vocab)
print(f"\n--- Training corpus ---")
print(f"Vocabulary ({VOCAB_SIZE} tokens): {train_vocab}")
print(f"Corpus length: {len(corpus_tokens)} tokens")
print(f"First 20 tokens: {[train_id2w[t.item()] for t in corpus_tokens[:20]]}")


# --- 5.3 Data loading function --------------------------------------------

SEQ_LEN = 32
BATCH_SIZE = 8


def get_batch(data, batch_size, seq_len):
    """Sample random (input, target) pairs from the token corpus."""
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i + seq_len] for i in ix])
    y = torch.stack([data[i + 1:i + seq_len + 1] for i in ix])
    return x, y


# Verify batch shapes
x_demo, y_demo = get_batch(corpus_tokens, BATCH_SIZE, SEQ_LEN)
print(f"\nBatch shapes: input={list(x_demo.shape)}, target={list(y_demo.shape)}")
print(f"Sample input:  {[train_id2w[t.item()] for t in x_demo[0, :8]]}...")
print(f"Sample target: {[train_id2w[t.item()] for t in y_demo[0, :8]]}...")


# --- 5.4 Initialize model -------------------------------------------------

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 4
D_FF = 256

model = TinyGPT(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    max_seq_len=SEQ_LEN,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"\n--- Model config ---")
print(f"  vocab_size  = {VOCAB_SIZE}")
print(f"  d_model     = {D_MODEL}")
print(f"  n_heads     = {N_HEADS}")
print(f"  n_layers    = {N_LAYERS}")
print(f"  d_ff        = {D_FF}")
print(f"  max_seq_len = {SEQ_LEN}")
print(f"  parameters  = {n_params:,}")


# --- 5.5 Training -----------------------------------------------------------

TRAIN_STEPS = 300
WARMUP = 30
MAX_LR = 1e-3
MIN_LR = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# Separate weight-decay groups: decay all except biases and layer norms
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
param_groups = [
    {"params": decay_params, "weight_decay": WEIGHT_DECAY},
    {"params": no_decay_params, "weight_decay": 0.0},
]

optimizer = torch.optim.AdamW(param_groups, lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8)
loss_fn = nn.CrossEntropyLoss()

print(f"\n--- Training config ---")
print(f"  steps        = {TRAIN_STEPS}")
print(f"  warmup       = {WARMUP}")
print(f"  max_lr       = {MAX_LR}")
print(f"  min_lr       = {MIN_LR}")
print(f"  weight_decay = {WEIGHT_DECAY}")
print(f"  grad_clip    = {GRAD_CLIP}")
print(f"  decay params:    {sum(p.numel() for p in decay_params):,}")
print(f"  no-decay params: {sum(p.numel() for p in no_decay_params):,}")

print(f"\n--- Training ---")
print(f"{'Step':>5s} | {'Loss':>8s} | {'PPL':>8s} | {'Grad Norm':>10s} | {'LR':>10s}")
print("-" * 56)

torch.manual_seed(42)
train_losses = []
train_ppls = []

for step in range(TRAIN_STEPS):
    # Learning rate schedule
    lr = get_lr(step, WARMUP, TRAIN_STEPS, MAX_LR, MIN_LR)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    # Get batch
    xb, yb = get_batch(corpus_tokens, BATCH_SIZE, SEQ_LEN)

    # Forward pass
    logits = model(xb)                                  # (B, T, V)
    loss = loss_fn(logits.view(-1, VOCAB_SIZE), yb.view(-1))

    # Backward pass
    loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    # Track metrics
    loss_val = loss.item()
    ppl_val = math.exp(min(loss_val, 20))  # cap to avoid overflow
    train_losses.append(loss_val)
    train_ppls.append(ppl_val)

    if step % 50 == 0 or step == TRAIN_STEPS - 1:
        print(f"{step:5d} | {loss_val:8.4f} | {ppl_val:8.2f} | {grad_norm:10.4f} | {lr:10.6f}")

print(f"\nTraining complete!")
print(f"  Initial loss: {train_losses[0]:.4f} (PPL={train_ppls[0]:.1f})")
print(f"  Final loss:   {train_losses[-1]:.4f} (PPL={train_ppls[-1]:.1f})")
print(f"  Random baseline: loss={math.log(VOCAB_SIZE):.4f} (PPL={VOCAB_SIZE})")


# --- 5.6 Loss curve (ASCII) -----------------------------------------------

print(f"\nLoss curve:")
num_bars = 30
for i in range(0, TRAIN_STEPS, 50):
    bar_len = int(min(train_losses[i], 4.0) / 4.0 * num_bars)
    bar = "█" * bar_len + "░" * (num_bars - bar_len)
    print(f"  Step {i:3d} | {bar} | {train_losses[i]:.4f}")
i = TRAIN_STEPS - 1
bar_len = int(min(train_losses[i], 4.0) / 4.0 * num_bars)
bar = "█" * bar_len + "░" * (num_bars - bar_len)
print(f"  Step {i:3d} | {bar} | {train_losses[i]:.4f}")


# --- 5.7 Quick generation test --------------------------------------------

print(f"\n--- Generation test (greedy, from trained model) ---")
model.eval()

prompt_text = "the cat"
prompt_ids = torch.tensor([[train_w2id[w] for w in prompt_text.split()]])

generated = prompt_ids.clone()
with torch.no_grad():
    for _ in range(10):
        logits = model(generated[:, -SEQ_LEN:])
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)

gen_words = [train_id2w[t.item()] for t in generated[0]]
print(f"Prompt: \"{prompt_text}\"")
print(f"Generated: {' '.join(gen_words)}")
print(f"(A tiny model on a tiny corpus — patterns should be recognizable)")

model.train()


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — CHECKPOINTING (Save / Load / Resume)
# ═══════════════════════════════════════════════════════════════════════════
#
# Training a large LLM takes days to weeks. Hardware fails, jobs get
# preempted, bugs surface. Without checkpointing, you'd lose everything.
#
# A checkpoint saves:
#   - model weights (state_dict)
#   - optimizer state (momentum buffers, adaptive learning rates)
#   - training step / epoch
#   - loss value (for monitoring)
#   - random state (for exact reproducibility)
#
# You MUST save optimizer state to resume properly. If you only save
# model weights, Adam's moment estimates reset to zero and training
# quality drops (the "warm-up penalty" of not saving optimizer state).
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 6: CHECKPOINTING")
print("=" * 60)


# --- 6.1 Save a checkpoint ------------------------------------------------

ckpt_dir = tempfile.mkdtemp()
ckpt_path = os.path.join(ckpt_dir, "tiny_gpt_step300.pt")

checkpoint = {
    "step": TRAIN_STEPS,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": train_losses[-1],
    "config": {
        "vocab_size": VOCAB_SIZE,
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "max_seq_len": SEQ_LEN,
    },
    "rng_state": torch.random.get_rng_state(),
}

torch.save(checkpoint, ckpt_path)
ckpt_size = os.path.getsize(ckpt_path)
print(f"\n--- Save checkpoint ---")
print(f"Saved to: {os.path.basename(ckpt_path)}")
print(f"Size: {ckpt_size / 1024:.1f} KB")
print(f"Contents:")
for key in checkpoint:
    if key == "model_state_dict":
        print(f"  {key}: {len(checkpoint[key])} tensors")
    elif key == "optimizer_state_dict":
        n_states = len(checkpoint[key].get("state", {}))
        print(f"  {key}: {n_states} parameter states")
    elif key == "rng_state":
        print(f"  {key}: tensor of shape {checkpoint[key].shape}")
    else:
        print(f"  {key}: {checkpoint[key]}")


# --- 6.2 Load checkpoint and verify ---------------------------------------

loaded = torch.load(ckpt_path, weights_only=False)

model_loaded = TinyGPT(**loaded["config"])
model_loaded.load_state_dict(loaded["model_state_dict"])

# Verify weights match
match = all(
    torch.equal(p1, p2)
    for p1, p2 in zip(model.parameters(), model_loaded.parameters())
)
print(f"\n--- Load checkpoint ---")
print(f"Loaded step: {loaded['step']}")
print(f"Loaded loss: {loaded['loss']:.4f}")
print(f"Weights match: {match}")


# --- 6.3 Resume training --------------------------------------------------

print(f"\n--- Resume training from checkpoint ---")

# Restore optimizer state (critical for good resumed training)
decay_resumed = [p for n, p in model_loaded.named_parameters() if p.dim() >= 2]
no_decay_resumed = [p for n, p in model_loaded.named_parameters() if p.dim() < 2]
optimizer_resumed = torch.optim.AdamW(
    [
        {"params": decay_resumed, "weight_decay": WEIGHT_DECAY},
        {"params": no_decay_resumed, "weight_decay": 0.0},
    ],
    lr=MAX_LR, betas=(0.9, 0.95), eps=1e-8,
)
optimizer_resumed.load_state_dict(loaded["optimizer_state_dict"])

# Restore RNG state for reproducibility
torch.random.set_rng_state(loaded["rng_state"])

start_step = loaded["step"]
extra_steps = 50
model_loaded.train()

print(f"Resuming from step {start_step}, training {extra_steps} more steps...")
print(f"{'Step':>5s} | {'Loss':>8s} | {'PPL':>8s}")
print("-" * 30)

for step in range(start_step, start_step + extra_steps):
    lr = get_lr(step, WARMUP, start_step + extra_steps, MAX_LR, MIN_LR)
    for pg in optimizer_resumed.param_groups:
        pg["lr"] = lr

    xb, yb = get_batch(corpus_tokens, BATCH_SIZE, SEQ_LEN)
    logits = model_loaded(xb)
    loss = loss_fn(logits.view(-1, VOCAB_SIZE), yb.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model_loaded.parameters(), GRAD_CLIP)
    optimizer_resumed.step()
    optimizer_resumed.zero_grad()

    if step % 25 == 0 or step == start_step + extra_steps - 1:
        ppl = math.exp(min(loss.item(), 20))
        print(f"{step:5d} | {loss.item():8.4f} | {ppl:8.2f}")

print(f"\nResumed training complete — model continued improving from checkpoint.")

# Cleanup
os.unlink(ckpt_path)
os.rmdir(ckpt_dir)


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
Exercise 1: Custom Learning Rate Schedule
-------------------------------------------
Implement a learning rate schedule with THREE phases:
  a) Warmup: linear increase from 0 to max_lr over 200 steps
  b) Constant: hold at max_lr for 300 steps
  c) Cosine decay: decrease to min_lr over the remaining 500 steps
Use max_lr=3e-4, min_lr=1e-5, total_steps=1000.
Plot it as ASCII art and compare to the standard warmup+cosine schedule.

Exercise 2: Manual AdamW vs PyTorch AdamW
-------------------------------------------
  a) Create a small model: nn.Linear(10, 1)
  b) Run 100 steps with PyTorch's AdamW (lr=0.01, betas=(0.9, 0.99))
  c) Implement AdamW from scratch (maintain m, v buffers yourself)
  d) Run 100 steps with your manual implementation
  e) Verify both produce the same final weights (within floating point tolerance)
Hint: don't forget bias correction in your manual implementation.

Exercise 3: Train on a Longer Corpus
--------------------------------------
  a) Create a richer synthetic corpus with 30+ unique words and
     more complex patterns (e.g., "if the cat is hungry the cat eats fish")
  b) Train the TinyGPT for 500 steps with seq_len=64
  c) Plot training loss and perplexity curves (ASCII or matplotlib)
  d) Generate 20 tokens from 3 different prompts — does the model
     learn the sentence structures?

Exercise 4: Hyperparameter Sweep
----------------------------------
Train the TinyGPT model with different configurations and compare:
  a) Learning rates: 1e-4 vs 5e-4 vs 1e-3 vs 5e-3
  b) Model sizes: d_model in [32, 64, 128], n_layers in [2, 4, 8]
  c) For each config, train for 200 steps and record the final loss.
  d) Which learning rate works best? Is bigger always better for the model?
Hint: too-high LR causes divergence. Too-large model overfits the tiny corpus.

Exercise 5: Checkpoint Robustness
------------------------------------
  a) Train a model for 200 steps, save a checkpoint
  b) Load the checkpoint and continue training for 100 more steps
  c) Compare the final loss to training 300 steps straight through
  d) Are they identical? (They should be, if you saved the RNG state)
  e) What happens if you DON'T restore the optimizer state?
     Load only the model weights and train 100 more steps.
     Compare the loss — it should be worse initially (no momentum history).
""")
