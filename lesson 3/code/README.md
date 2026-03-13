# Lesson 3: Neural Network Building Blocks

Understand every component that will later appear inside a transformer — so when you build one in Lesson 6, nothing is magic.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- Lesson 2 completed (tensors, autograd, nn.Module, training loops)

## How to Run

```bash
python3 nn_building_blocks.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Linear Layers
`nn.Linear` computes `y = xW^T + b` — the fundamental operation in every neural network. We verify the math manually, trace shape transformations through a transformer (Q projection, FFN up/down, LM head), and explore bias-free variants used in LLaMA.

**Key concepts:** A GPT-2 (124M params) has ~50 nn.Linear layers. Every projection — Q, K, V, output, FFN, LM head — is just a matrix multiply + bias.

### Part 2 — Activation Functions
ReLU, GELU, and SiLU/Swish — why non-linearity is essential (without it, stacked linears collapse into one), implementing GELU from scratch using the tanh approximation, and why modern LLMs prefer smooth activations over ReLU's hard cutoff.

**Key concepts:** GELU is the standard for GPT-2/BERT. SiLU/Swish is used in LLaMA/Mistral. Both allow small negative gradients to flow, preventing dead neurons.

### Part 3 — Layer Normalization
Implementing LayerNorm from scratch (mean subtraction, variance normalization, learned scale/shift), comparing to `nn.LayerNorm`, implementing RMSNorm (LLaMA-style, ~10-15% faster), and understanding pre-norm vs post-norm placement.

**Key concepts:** LayerNorm stabilizes activations across depth. RMSNorm skips the mean step. Pre-norm (normalize before sublayer) is used in all modern LLMs.

### Part 4 — Dropout
How dropout works (random zeroing + scaling by 1/(1-p)), the critical difference between training and eval mode, and where dropout is applied in transformers (attention weights, residual connections, embeddings).

**Key concepts:** Forgetting `model.eval()` during inference is a common bug. GPT-2/3 use dropout rate 0.1.

### Part 5 — Residual Connections
Implementing skip connections (`y = x + F(x)`), the key gradient flow experiment comparing 24 layers with and without residuals, and building a transformer-style FFN block with pre-norm + residual.

**Key concepts:** Without residuals, gradients vanish through deep networks. GPT-3's 96 layers are only possible because of skip connections.

### Part 6 — Weight Initialization
The problem (too large → explosion, too small → vanishing), Xavier/Glorot for balanced variance, Kaiming/He for ReLU-family activations, and GPT-2's residual scaling trick (1/sqrt(2*n_layers)).

**Key concepts:** Bad initialization makes training impossible. GPT-2 scales residual output projections to prevent variance growth across depth.

## Exercises

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | LayerNorm from Scratch | Implement normalize → scale → shift without peeking |
| 2 | RMSNorm from Scratch | Implement the simpler LLaMA-style norm, benchmark speed |
| 3 | FFN with Residuals | Build a complete pre-norm + residual + dropout FFN block |
| 4 | Gradient Flow | Compare gradient magnitudes at depths 8, 16, 32, 48 |
| 5 | Initialization Experiment | Test 3 init strategies on a 20-layer network |

## Key Takeaways

- **Linear layers** are the fundamental building block — every transformer projection is `y = xW^T + b`
- **Activation functions** (GELU, SiLU) add non-linearity — without them, depth is pointless
- **LayerNorm / RMSNorm** stabilize activations — pre-norm placement is the modern standard
- **Dropout** regularizes during training but must be disabled at inference (`model.eval()`)
- **Residual connections** enable deep networks by giving gradients a direct path backward
- **Weight initialization** (Xavier, Kaiming, residual scaling) prevents exploding/vanishing activations

## What's Next

[Lesson 4: Text Representation & Tokenization](../../lesson%204/) — Converting raw text into numbers with BPE, vocabulary design, and token embeddings.
