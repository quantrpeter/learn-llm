"""
============================================================================
 LESSON 1: Mathematical Foundations for LLMs
============================================================================

 Goal: Build the math intuition required to understand how neural networks
       and transformers work.

 Topics covered:
   1. Linear Algebra  — vectors, matrices, dot products, norms
   2. Calculus         — derivatives, chain rule, gradients
   3. Probability      — distributions, entropy, cross-entropy, KL divergence
   4. Softmax          — converting raw scores to probabilities
   5. Numerical Stability — log-sum-exp trick

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom
============================================================================
"""

import math
import random

# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — LINEAR ALGEBRA
# ═══════════════════════════════════════════════════════════════════════════
#
# In a neural network, ALL computation is matrix math.
#   - An input sentence becomes a matrix of shape (sequence_length, d_model)
#   - Every layer multiplies that matrix by a weight matrix
#   - Attention scores come from a matrix multiply: Q @ K^T
#
# You don't need a math degree — you need to be comfortable with these
# five operations: dot product, matrix multiply, transpose, norms, and
# broadcasting.
# ═══════════════════════════════════════════════════════════════════════════


# --- 1.1 Vectors ----------------------------------------------------------
# A vector is just a list of numbers. In ML, a vector represents a single
# data point — for example, the embedding of one token.

def vector_add(a: list[float], b: list[float]) -> list[float]:
    """Element-wise addition of two vectors."""
    assert len(a) == len(b), "Vectors must be same length"
    return [ai + bi for ai, bi in zip(a, b)]


def vector_scale(scalar: float, v: list[float]) -> list[float]:
    """Multiply every element of a vector by a scalar."""
    return [scalar * vi for vi in v]


def dot_product(a: list[float], b: list[float]) -> float:
    """
    Dot product: sum of element-wise products.

    WHY THIS MATTERS:
    - Attention scores are computed as dot products between query and key vectors.
    - A large dot product means two vectors point in the same direction (similar).
    - A dot product of zero means they are orthogonal (unrelated).

    Formula: a · b = Σ(ai * bi)
    """
    assert len(a) == len(b)
    return sum(ai * bi for ai, bi in zip(a, b))


def vector_norm(v: list[float]) -> float:
    """
    L2 norm (Euclidean length) of a vector.

    WHY THIS MATTERS:
    - Used in RMSNorm (a normalization layer in modern LLMs like LLaMA).
    - Also used in cosine similarity for comparing embeddings.

    Formula: ||v|| = sqrt(Σ vi²)
    """
    return math.sqrt(sum(vi ** 2 for vi in v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity: dot product of unit vectors.
    Returns a value between -1 (opposite) and 1 (identical direction).

    WHY THIS MATTERS:
    - This is how we measure whether two word embeddings are "similar."
    - "king" and "queen" have high cosine similarity in embedding space.
    """
    return dot_product(a, b) / (vector_norm(a) * vector_norm(b))


# --- 1.2 Matrices ---------------------------------------------------------
# A matrix is a 2D grid of numbers. We represent it as a list of rows.
# In LLMs:
#   - Weight matrices have shape (input_dim, output_dim)
#   - A batch of token embeddings is a matrix of shape (seq_len, d_model)

Matrix = list[list[float]]


def matrix_shape(m: Matrix) -> tuple[int, int]:
    """Return (rows, cols) of a matrix."""
    return len(m), len(m[0])


def matrix_transpose(m: Matrix) -> Matrix:
    """
    Flip rows and columns.

    WHY THIS MATTERS:
    - In attention: scores = Q @ K^T  (we transpose the key matrix)
    - Transpose swaps shape from (n, d) to (d, n)
    """
    rows, cols = matrix_shape(m)
    return [[m[r][c] for r in range(rows)] for c in range(cols)]


def matrix_multiply(a: Matrix, b: Matrix) -> Matrix:
    """
    Matrix multiplication: C[i][j] = dot product of row i of A and col j of B.

    Requirements: A is (m, n), B is (n, p) → result is (m, p).

    WHY THIS MATTERS:
    - This is THE fundamental operation in neural networks.
    - Every linear layer computes: output = input @ weights + bias
    - Attention computes: scores = (Q @ K^T) / sqrt(d_k)
    """
    a_rows, a_cols = matrix_shape(a)
    b_rows, b_cols = matrix_shape(b)
    assert a_cols == b_rows, f"Incompatible shapes: ({a_rows},{a_cols}) @ ({b_rows},{b_cols})"

    b_t = matrix_transpose(b)
    return [
        [dot_product(a[i], b_t[j]) for j in range(b_cols)]
        for i in range(a_rows)
    ]


# --- Demo: Matrix operations ---
print("=" * 60)
print("PART 1: LINEAR ALGEBRA")
print("=" * 60)

v1 = [1.0, 2.0, 3.0]
v2 = [4.0, 5.0, 6.0]
print(f"\nv1 = {v1}")
print(f"v2 = {v2}")
print(f"v1 + v2 = {vector_add(v1, v2)}")
print(f"v1 · v2 = {dot_product(v1, v2)}")
print(f"||v1|| = {vector_norm(v1):.4f}")
print(f"cosine_similarity(v1, v2) = {cosine_similarity(v1, v2):.4f}")

A = [[1, 2], [3, 4], [5, 6]]  # 3x2
B = [[7, 8, 9], [10, 11, 12]]  # 2x3
C = matrix_multiply(A, B)  # 3x3
print(f"\nA (3×2) @ B (2×3) = C (3×3):")
for row in C:
    print(f"  {row}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — CALCULUS & GRADIENTS
# ═══════════════════════════════════════════════════════════════════════════
#
# Neural networks learn by:
#   1. Computing a loss (how wrong the prediction is)
#   2. Computing the gradient of the loss w.r.t. every weight
#   3. Nudging weights in the direction that reduces the loss
#
# The chain rule lets us propagate gradients backward through many layers.
# This is what "backpropagation" is — just the chain rule applied repeatedly.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: CALCULUS & GRADIENTS")
print("=" * 60)


def numerical_derivative(f, x: float, h: float = 1e-7) -> float:
    """
    Approximate df/dx using the central difference method.

    This is how you verify your analytical gradients are correct.
    PyTorch's autograd does this symbolically, but the idea is the same.
    """
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, params: list[float], h: float = 1e-7) -> list[float]:
    """
    Compute partial derivative of f w.r.t. each parameter.
    This is the gradient vector: ∇f = [∂f/∂p0, ∂f/∂p1, ...]
    """
    grad = []
    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += h
        params_minus[i] -= h
        grad.append((f(params_plus) - f(params_minus)) / (2 * h))
    return grad


# Example: gradient of a simple function f(x) = x²
print(f"\nf(x) = x²")
print(f"Analytical derivative at x=3: 2*3 = 6")
print(f"Numerical derivative at x=3:  {numerical_derivative(lambda x: x**2, 3.0):.6f}")


# Example: gradient of a multi-variable loss
# Suppose loss = (prediction - target)² where prediction = w*x + b
# This is a tiny "neural network" with one weight and one bias.
def mse_loss(params: list[float]) -> float:
    """Mean squared error for a single data point: y_hat = w*x + b"""
    w, b = params
    x, y_true = 2.0, 5.0  # one training example
    y_pred = w * x + b
    return (y_pred - y_true) ** 2


print(f"\nMSE loss at w=1.0, b=0.0: {mse_loss([1.0, 0.0]):.4f}")
grad = numerical_gradient(mse_loss, [1.0, 0.0])
print(f"Gradient [∂L/∂w, ∂L/∂b] = [{grad[0]:.4f}, {grad[1]:.4f}]")
print("(Negative gradient points toward lower loss — that's where we step)")


# --- Chain Rule Demo: The Heart of Backpropagation -----------------------
# If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)
#
# In a neural network with layers L1 → L2 → L3 → loss:
#   ∂loss/∂L1 = ∂loss/∂L3 * ∂L3/∂L2 * ∂L2/∂L1
# This is exactly what PyTorch's autograd computes automatically.

print("\n--- Chain Rule ---")
print("y = (3x + 2)²")
print("Let u = 3x + 2, y = u²")
print("dy/dx = dy/du * du/dx = 2u * 3 = 6(3x + 2)")

x = 1.0
analytical = 6 * (3 * x + 2)
numerical = numerical_derivative(lambda x: (3 * x + 2) ** 2, x)
print(f"At x=1: analytical = {analytical:.4f}, numerical = {numerical:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — PROBABILITY, ENTROPY & CROSS-ENTROPY
# ═══════════════════════════════════════════════════════════════════════════
#
# LLMs are probability machines. At each step, the model outputs a
# probability distribution over the ENTIRE vocabulary (~32,000-128,000 tokens).
#
# The training loss (cross-entropy) measures: "how surprised was the model
# by the correct next token?"
#
# Low cross-entropy = model assigns high probability to the right answer.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3: PROBABILITY & INFORMATION THEORY")
print("=" * 60)


def entropy(probs: list[float]) -> float:
    """
    Shannon entropy: H(p) = -Σ p(x) * log(p(x))

    Measures the "uncertainty" or "information content" of a distribution.
    - A fair coin has entropy = 1 bit (maximum uncertainty for 2 outcomes)
    - A loaded coin (99% heads) has entropy ≈ 0.08 bits (very predictable)

    WHY THIS MATTERS:
    - A well-trained LLM has low entropy on natural text (it's confident
      about the next token).
    - High entropy = the model is uncertain (many tokens seem equally likely).
    """
    return -sum(p * math.log(p + 1e-12) for p in probs if p > 0)


def cross_entropy(true_probs: list[float], pred_probs: list[float]) -> float:
    """
    Cross-entropy: H(p, q) = -Σ p(x) * log(q(x))

    Measures how well distribution q (model's prediction) matches
    distribution p (true distribution, usually one-hot).

    THIS IS THE LOSS FUNCTION for training LLMs.

    When p is one-hot (only one correct answer):
      H(p, q) = -log(q(correct_token))
    So we just need the model to assign high probability to the right token.
    """
    assert len(true_probs) == len(pred_probs)
    return -sum(p * math.log(q + 1e-12) for p, q in zip(true_probs, pred_probs) if p > 0)


def kl_divergence(p: list[float], q: list[float]) -> float:
    """
    KL divergence: D_KL(p || q) = Σ p(x) * log(p(x) / q(x))

    Measures "extra surprise" from using q instead of the true distribution p.
    Always >= 0. Equals 0 only when p == q.

    WHY THIS MATTERS:
    - Used in RLHF/DPO (Lesson 13) to keep the fine-tuned model close to
      the base model. A KL penalty prevents the model from changing too much.
    - cross_entropy(p, q) = entropy(p) + kl_divergence(p, q)
    """
    assert len(p) == len(q)
    return sum(
        pi * math.log((pi + 1e-12) / (qi + 1e-12))
        for pi, qi in zip(p, q) if pi > 0
    )


# Demo: Compare distributions
fair_coin = [0.5, 0.5]
loaded_coin = [0.99, 0.01]
print(f"\nFair coin   entropy: {entropy(fair_coin):.4f} nats")
print(f"Loaded coin entropy: {entropy(loaded_coin):.4f} nats")

# Simulating LLM next-token prediction
# Vocabulary: ["the", "a", "cat", "dog", "sat"]
vocab = ["the", "a", "cat", "dog", "sat"]
true_dist = [0.0, 0.0, 1.0, 0.0, 0.0]  # correct answer is "cat"

good_model = [0.05, 0.05, 0.80, 0.05, 0.05]  # confident and correct
bad_model = [0.20, 0.20, 0.20, 0.20, 0.20]  # uniform — clueless
wrong_model = [0.05, 0.05, 0.05, 0.80, 0.05]  # confident but wrong

print(f"\nNext token prediction (correct = 'cat'):")
print(f"  Good model  CE loss: {cross_entropy(true_dist, good_model):.4f}")
print(f"  Bad model   CE loss: {cross_entropy(true_dist, bad_model):.4f}")
print(f"  Wrong model CE loss: {cross_entropy(true_dist, wrong_model):.4f}")
print("  (Lower is better — good model is most confident on 'cat')")


# Perplexity: the exponential of cross-entropy
# A perplexity of 10 means the model is "as confused as if it were choosing
# uniformly among 10 tokens" at each step.
print(f"\nPerplexity (exp of CE loss):")
print(f"  Good model:  {math.exp(cross_entropy(true_dist, good_model)):.2f}")
print(f"  Bad model:   {math.exp(cross_entropy(true_dist, bad_model)):.2f}")
print(f"  Wrong model: {math.exp(cross_entropy(true_dist, wrong_model)):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — SOFTMAX
# ═══════════════════════════════════════════════════════════════════════════
#
# The final layer of an LLM outputs raw scores called "logits" — one per
# token in the vocabulary. These can be any real number.
#
# Softmax converts logits → probabilities (positive, sum to 1).
#
# softmax(z_i) = exp(z_i) / Σ exp(z_j)
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 4: SOFTMAX")
print("=" * 60)


def softmax_naive(logits: list[float]) -> list[float]:
    """
    Naive softmax — BROKEN for large values.
    If logits contain large numbers, exp() overflows to infinity.
    """
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def softmax(logits: list[float]) -> list[float]:
    """
    Numerically stable softmax.
    Subtracting max(logits) prevents overflow without changing the result.

    Proof: softmax(z - c) = exp(z_i - c) / Σ exp(z_j - c)
         = exp(z_i)*exp(-c) / (exp(-c) * Σ exp(z_j))
         = softmax(z)
    """
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


# Demo: softmax turns logits into probabilities
logits = [2.0, 1.0, 0.1, -1.0, 3.0]
probs = softmax(logits)
print(f"\nLogits: {logits}")
print(f"Softmax: {[f'{p:.4f}' for p in probs]}")
print(f"Sum: {sum(probs):.6f}  (should be 1.0)")
print(f"Highest prob at index {probs.index(max(probs))} (matches highest logit at index {logits.index(max(logits))})")

# Temperature scaling — controls sharpness of the distribution
# Used during text generation (Lesson 10)
print(f"\nTemperature scaling (lower T = sharper, higher T = flatter):")
for temp in [0.5, 1.0, 2.0]:
    scaled = [z / temp for z in logits]
    p = softmax(scaled)
    print(f"  T={temp}: {[f'{x:.3f}' for x in p]}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — NUMERICAL STABILITY: THE LOG-SUM-EXP TRICK
# ═══════════════════════════════════════════════════════════════════════════
#
# In practice, we almost never compute softmax then take log separately.
# Instead we compute log_softmax directly using the log-sum-exp trick.
# This avoids taking log(0) which gives -infinity.
#
# log_softmax(z_i) = z_i - log(Σ exp(z_j))
# log(Σ exp(z_j)) = max(z) + log(Σ exp(z_j - max(z)))   ← stable version
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 5: NUMERICAL STABILITY")
print("=" * 60)


def log_sum_exp(logits: list[float]) -> float:
    """
    Compute log(Σ exp(z_i)) in a numerically stable way.
    Used everywhere in LLM training — inside the loss function.
    """
    max_logit = max(logits)
    return max_logit + math.log(sum(math.exp(z - max_logit) for z in logits))


def log_softmax(logits: list[float]) -> list[float]:
    """
    Log-softmax: more stable than log(softmax(z)).
    This is what PyTorch actually uses in nn.CrossEntropyLoss internally.
    """
    lse = log_sum_exp(logits)
    return [z - lse for z in logits]


def cross_entropy_from_logits(logits: list[float], target_index: int) -> float:
    """
    Cross-entropy loss directly from logits (the practical version).
    This is exactly what nn.CrossEntropyLoss does in PyTorch.

    loss = -log_softmax(logits)[target_index]
    """
    return -log_softmax(logits)[target_index]


# Demo: numerical stability
print("\n--- Overflow demo ---")
big_logits = [1000.0, 1001.0, 1002.0]
try:
    naive_result = softmax_naive(big_logits)
    print(f"Naive softmax on {big_logits}: {naive_result}")
except OverflowError:
    print(f"Naive softmax on {big_logits}: OVERFLOW (exp(1000) is too large)")

stable_result = softmax(big_logits)
print(f"Stable softmax on {big_logits}: {[f'{p:.4f}' for p in stable_result]}")

# Demo: cross-entropy from logits
print(f"\n--- Cross-entropy from logits ---")
logits = [2.0, 1.0, 0.1, -1.0, 3.0]
target = 2  # correct token is at index 2
loss = cross_entropy_from_logits(logits, target)
print(f"Logits: {logits}")
print(f"Target index: {target}")
print(f"Loss: {loss:.4f}")
print(f"Perplexity: {math.exp(loss):.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — PUTTING IT ALL TOGETHER: A TINY FORWARD PASS
# ═══════════════════════════════════════════════════════════════════════════
#
# Let's simulate one forward pass of a toy "language model":
#   input embedding → linear layer → softmax → loss
#
# This is exactly what happens (at massive scale) in a real LLM.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 6: TINY FORWARD PASS (PUTTING IT ALL TOGETHER)")
print("=" * 60)

random.seed(42)

VOCAB_SIZE = 8
EMBED_DIM = 4

# Simulate: one token embedding (what comes out of the transformer)
token_embedding = [random.gauss(0, 1) for _ in range(EMBED_DIM)]

# Weight matrix of the final "language model head" layer
# Maps from embedding space (d=4) to vocabulary space (v=8)
lm_head_weights = [
    [random.gauss(0, 0.5) for _ in range(VOCAB_SIZE)]
    for _ in range(EMBED_DIM)
]

# Step 1: Compute logits = embedding @ lm_head_weights
#   (1, 4) @ (4, 8) → (1, 8) → 8 logits, one per vocab token
logits = [
    sum(token_embedding[d] * lm_head_weights[d][v] for d in range(EMBED_DIM))
    for v in range(VOCAB_SIZE)
]

# Step 2: Convert to probabilities
probs = softmax(logits)

# Step 3: Compute loss (target token is index 3)
target_token = 3
loss = cross_entropy_from_logits(logits, target_token)

print(f"\nToken embedding (d={EMBED_DIM}): {[f'{x:.3f}' for x in token_embedding]}")
print(f"LM head weights shape: ({EMBED_DIM}, {VOCAB_SIZE})")
print(f"Logits (vocab_size={VOCAB_SIZE}): {[f'{x:.3f}' for x in logits]}")
print(f"Probabilities: {[f'{p:.4f}' for p in probs]}")
print(f"Target token index: {target_token}")
print(f"P(target): {probs[target_token]:.4f}")
print(f"Loss: {loss:.4f}")
print(f"Perplexity: {math.exp(loss):.2f}")
print(f"\nInterpretation: the model is as confused as choosing among "
      f"{math.exp(loss):.1f} tokens uniformly.")
print(f"(An untrained model with vocab_size={VOCAB_SIZE} would have "
      f"perplexity ≈ {VOCAB_SIZE})")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
Exercise 1: Matrix Multiplication
----------------------------------
Implement matrix_multiply without looking at the solution above.
Test it: A(2×3) @ B(3×2) should give a (2×2) result.
Verify: [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]

Exercise 2: Gradient Descent by Hand
-------------------------------------
Using the mse_loss function above (with x=2, y_true=5):
  a) Start with w=0.0, b=0.0
  b) Compute the gradient using numerical_gradient()
  c) Update: w = w - 0.01 * grad_w, b = b - 0.01 * grad_b
  d) Repeat 100 times. Print the loss every 10 steps.
  e) What values do w and b converge to? (Hint: y = w*x + b = 5)

Exercise 3: Softmax Properties
-------------------------------
  a) Verify that softmax([0, 0, 0]) = [1/3, 1/3, 1/3]
  b) Verify that softmax([z + c for z in logits]) == softmax(logits) for any c
  c) What happens to softmax as temperature approaches 0? (Try T=0.01)
  d) What happens as temperature approaches infinity? (Try T=100)

Exercise 4: Cross-Entropy Intuition
-------------------------------------
  a) What is the minimum possible cross-entropy loss? (Hint: when the model
     is 100% confident and correct)
  b) What is the cross-entropy loss when the model assigns probability
     1/vocab_size to every token? Express in terms of vocab_size.
  c) GPT-2 has vocab_size=50257. What perplexity corresponds to a random
     model?

Exercise 5: KL Divergence
---------------------------
  a) Compute KL(p || q) and KL(q || p) for:
     p = [0.9, 0.1]
     q = [0.5, 0.5]
  b) Are they equal? (KL divergence is NOT symmetric)
  c) When is KL divergence = 0?
""")
