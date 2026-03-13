# Lesson 1: Mathematical Foundations for LLMs

Build the math intuition required to understand how neural networks and transformers work.

## Prerequisites

- Python 3.10+
- No external dependencies (uses only `math` and `random` from the standard library)

## How to Run

```bash
python3 math_foundations.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Linear Algebra
Vectors, dot products, matrix multiplication, cosine similarity. These are the building blocks of every neural network — attention scores are dot products, every layer is a matrix multiply.

**Key functions:** `dot_product()`, `matrix_multiply()`, `cosine_similarity()`

### Part 2 — Calculus & Gradients
Numerical derivatives, multi-variable gradients, and the chain rule. Backpropagation is the chain rule applied layer by layer — this section shows exactly how that works.

**Key functions:** `numerical_derivative()`, `numerical_gradient()`

### Part 3 — Probability & Information Theory
Entropy, cross-entropy, KL divergence, and perplexity. Cross-entropy is the loss function used to train every LLM. Perplexity is the standard evaluation metric.

**Key functions:** `entropy()`, `cross_entropy()`, `kl_divergence()`

### Part 4 — Softmax
Converts raw model outputs (logits) into a probability distribution over the vocabulary. Also covers temperature scaling, which controls how "creative" vs "deterministic" text generation is.

**Key functions:** `softmax()`, `softmax_naive()`

### Part 5 — Numerical Stability
The log-sum-exp trick and log-softmax — why naive implementations overflow and how to fix it. This is what PyTorch's `nn.CrossEntropyLoss` does internally.

**Key functions:** `log_sum_exp()`, `log_softmax()`, `cross_entropy_from_logits()`

### Part 6 — Tiny Forward Pass
Combines everything into a single forward pass of a toy language model: embedding → linear layer → softmax → cross-entropy loss. This is exactly what a real LLM does at massive scale.

## Exercises

5 exercises are printed at the end of the script:

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | Matrix Multiplication | Implement matmul from scratch |
| 2 | Gradient Descent | Run 100 steps of manual gradient descent |
| 3 | Softmax Properties | Explore shift-invariance and temperature |
| 4 | Cross-Entropy Intuition | Reason about min/max loss and perplexity |
| 5 | KL Divergence | Verify asymmetry and zero conditions |

## Key Takeaways

- **Dot product** measures similarity between vectors — this is how attention works
- **Matrix multiply** is the core operation of every neural network layer
- **Cross-entropy loss** = "how surprised was the model by the correct answer?"
- **Perplexity** = exp(cross-entropy) — "choosing among N equally likely tokens"
- **Softmax** turns logits into probabilities; **temperature** controls sharpness
- **Log-sum-exp trick** prevents numerical overflow in real implementations

## What's Next

[Lesson 2: Python & Deep Learning Tooling](../../lessons/lesson_02/) — PyTorch tensors, autograd, GPU acceleration, and training loops.
