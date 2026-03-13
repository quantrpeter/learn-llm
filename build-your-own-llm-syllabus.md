# Build Your Own LLM: Step-by-Step Lesson Syllabus

---

## Lesson 1: Mathematical Foundations

**Goal:** Build the math intuition required to understand how neural networks and transformers work.

| Topic | Details |
|-------|---------|
| Linear Algebra | Vectors, matrices, matrix multiplication, transpose, dot product, norms |
| Calculus | Derivatives, partial derivatives, chain rule, gradients |
| Probability & Statistics | Distributions, Bayes' theorem, entropy, cross-entropy, KL divergence |
| Softmax & Log-Softmax | Why softmax converts logits to probabilities |
| Numerical Stability | Overflow/underflow issues, log-sum-exp trick |

**Exercises:**
- Implement matrix multiplication from scratch in Python (no NumPy)
- Compute the gradient of a simple loss function by hand
- Implement softmax and cross-entropy loss from scratch

**Resources:**
- 3Blue1Brown — "Essence of Linear Algebra" (YouTube)
- Khan Academy — Multivariable Calculus

---

## Lesson 2: Python & Deep Learning Tooling

**Goal:** Get fluent with the core tools you'll use throughout the project.

| Topic | Details |
|-------|---------|
| Python for ML | NumPy, array broadcasting, vectorized operations |
| PyTorch Fundamentals | Tensors, autograd, GPU acceleration, `torch.nn` |
| Training Loop Anatomy | Forward pass, loss computation, backward pass, optimizer step |
| Debugging Tools | Tensor shape tracking, gradient checking, `torch.autograd.gradcheck` |
| Environment Setup | CUDA/ROCm, conda/venv, mixed precision with `torch.amp` |

**Exercises:**
- Train a simple 2-layer MLP on MNIST using raw PyTorch (no high-level wrappers)
- Profile GPU memory usage during training
- Implement gradient accumulation manually

---

## Lesson 3: Neural Network Building Blocks

**Goal:** Understand every component that will later appear inside a transformer.

| Topic | Details |
|-------|---------|
| Linear Layers | Weights, biases, `y = Wx + b` |
| Activation Functions | ReLU, GELU, SiLU/Swish — why non-linearity matters |
| Layer Normalization | Why LayerNorm over BatchNorm for sequences, RMSNorm variant |
| Dropout | Regularization during training, disabled at inference |
| Residual Connections | Skip connections, gradient flow, why they enable depth |
| Weight Initialization | Xavier, Kaiming, why initialization matters for training stability |

**Exercises:**
- Implement LayerNorm and RMSNorm from scratch
- Build a feed-forward network with residual connections
- Visualize gradient magnitudes across 20+ layers with and without skip connections

---

## Lesson 4: Text Representation & Tokenization

**Goal:** Convert raw text into numbers the model can process.

| Topic | Details |
|-------|---------|
| Character vs Word vs Subword | Trade-offs of each granularity |
| Byte Pair Encoding (BPE) | The algorithm behind GPT tokenizers |
| SentencePiece & Unigram | Alternative tokenization approaches |
| Vocabulary Size | Impact on model size, rare word handling, multilingual support |
| Special Tokens | `<bos>`, `<eos>`, `<pad>`, `<unk>` — purpose of each |
| Token Embeddings | Lookup tables, embedding dimensions, weight tying |

**Exercises:**
- Implement BPE tokenizer from scratch
- Train a custom tokenizer on your own text corpus using HuggingFace `tokenizers`
- Analyze vocabulary coverage across different languages

---

## Lesson 5: Attention Mechanism — The Core Innovation

**Goal:** Deeply understand self-attention, the mechanism that makes LLMs possible.

| Topic | Details |
|-------|---------|
| Intuition | Attention as soft dictionary lookup — queries, keys, values |
| Scaled Dot-Product Attention | `Attention(Q,K,V) = softmax(QK^T / √d_k) V` |
| Multi-Head Attention | Why multiple heads, head dimension, concatenation and projection |
| Causal (Masked) Attention | Lower-triangular mask for autoregressive generation |
| Attention Complexity | O(n²d) cost, why long sequences are expensive |
| KV Cache | Caching keys/values during inference for efficiency |

**Exercises:**
- Implement single-head attention from scratch
- Extend to multi-head attention
- Implement causal masking and verify it blocks future token visibility
- Visualize attention patterns on sample sentences

---

## Lesson 6: The Transformer Architecture

**Goal:** Assemble all components into a complete transformer decoder (GPT-style).

| Topic | Details |
|-------|---------|
| Original Transformer | Encoder-decoder architecture (Vaswani et al., 2017) |
| Decoder-Only Architecture | Why GPT/Llama use decoder-only (simplicity, causal LM) |
| Block Structure | Attention → LayerNorm → FFN → LayerNorm → Residual |
| Positional Encoding | Sinusoidal, learned, Rotary Position Embeddings (RoPE) |
| Pre-Norm vs Post-Norm | Training stability trade-offs |
| Model Dimensions | `d_model`, `n_heads`, `n_layers`, `d_ff` — how to size them |

**Exercises:**
- Implement a full transformer decoder block
- Stack N blocks into a complete GPT-style model
- Implement RoPE positional embeddings
- Count parameters and verify against known model sizes (GPT-2 124M)

---

## Lesson 7: Data Pipeline for Pre-Training

**Goal:** Prepare a large-scale text dataset for training your LLM.

| Topic | Details |
|-------|---------|
| Data Sources | Common Crawl, Wikipedia, books, code, The Pile, FineWeb |
| Data Cleaning | Deduplication, filtering low-quality text, language detection |
| Data Formatting | Packing sequences, document boundaries, padding strategies |
| DataLoader Design | Streaming vs map-style, shuffling, batch construction |
| Data Mixing | Proportions of different data sources, curriculum strategies |
| Data Contamination | Benchmark leakage risks, decontamination methods |

**Exercises:**
- Download and preprocess a subset of Wikipedia
- Build a streaming DataLoader that tokenizes on-the-fly
- Implement sequence packing to minimize padding waste
- Measure tokens/second throughput of your data pipeline

---

## Lesson 8: Pre-Training Your LLM

**Goal:** Train a language model from scratch on your prepared dataset.

| Topic | Details |
|-------|---------|
| Training Objective | Next-token prediction (causal language modeling) |
| Loss Function | Cross-entropy loss over vocabulary |
| Optimizer | AdamW — weight decay, beta1, beta2, epsilon |
| Learning Rate Schedule | Warmup + cosine decay, peak LR selection |
| Hyperparameters | Batch size, sequence length, gradient clipping |
| Training Stability | Loss spikes, gradient norms, NaN detection |
| Checkpointing | Saving/resuming, checkpoint frequency |

**Exercises:**
- Train a ~10M parameter model on a small corpus (WikiText-103)
- Implement cosine learning rate schedule with warmup
- Plot training loss curves, track perplexity
- Experiment with different batch sizes and learning rates

**Recommended Scale for Learning:**
| Model | Params | GPU Memory | Training Time |
|-------|--------|------------|---------------|
| Tiny | ~10M | 4GB | Hours |
| Small | ~125M | 16GB | Days |
| Medium | ~350M | 40GB+ | Weeks |

---

## Lesson 9: Distributed Training & Scaling

**Goal:** Learn techniques to train models that don't fit on a single GPU.

| Topic | Details |
|-------|---------|
| Data Parallelism (DDP) | Replicate model, split data, all-reduce gradients |
| Fully Sharded Data Parallelism (FSDP) | Shard parameters, gradients, and optimizer states |
| Tensor Parallelism | Split individual layers across GPUs |
| Pipeline Parallelism | Split model layers across GPUs, micro-batching |
| Mixed Precision (FP16/BF16) | Halve memory, increase throughput, loss scaling |
| Flash Attention | Memory-efficient attention, IO-aware algorithm |
| Gradient Checkpointing | Trade compute for memory |
| Scaling Laws | Chinchilla-optimal compute allocation, tokens vs parameters |

**Exercises:**
- Convert single-GPU training to DDP across 2+ GPUs
- Enable mixed-precision training with `torch.amp`
- Integrate Flash Attention and measure speedup
- Calculate Chinchilla-optimal model size for your compute budget

---

## Lesson 10: Text Generation & Decoding

**Goal:** Generate text from your trained model using different strategies.

| Topic | Details |
|-------|---------|
| Greedy Decoding | Always pick the highest probability token |
| Temperature Sampling | Control randomness — low T (sharp) vs high T (creative) |
| Top-k Sampling | Sample from top k most likely tokens |
| Top-p (Nucleus) Sampling | Sample from smallest set with cumulative probability ≥ p |
| Repetition Penalty | Discourage the model from repeating itself |
| Beam Search | Maintain multiple hypotheses (less common for open-ended generation) |
| Stopping Criteria | EOS token, max length, custom stop strings |
| KV Cache Implementation | Speed up autoregressive generation |

**Exercises:**
- Implement greedy, top-k, top-p, and temperature sampling
- Build a `generate()` function with KV caching
- Compare output quality across decoding strategies
- Measure tokens/second generation speed

---

## Lesson 11: Evaluation & Benchmarking

**Goal:** Measure how good your model actually is.

| Topic | Details |
|-------|---------|
| Perplexity | Primary metric for language models, lower is better |
| Benchmark Suites | HellaSwag, ARC, MMLU, TruthfulQA, WinoGrande |
| Eval Harness | Using `lm-evaluation-harness` for standardized evaluation |
| Few-Shot Evaluation | 0-shot, 5-shot prompting for benchmarks |
| Human Evaluation | When automated metrics fail |
| Contamination Checks | Ensuring benchmarks weren't in training data |

**Exercises:**
- Compute perplexity on a held-out test set
- Run your model through `lm-evaluation-harness` on 3+ benchmarks
- Compare your model's scores to published baselines at similar scale

---

## Lesson 12: Supervised Fine-Tuning (SFT)

**Goal:** Turn a base language model into an instruction-following assistant.

| Topic | Details |
|-------|---------|
| Base vs Chat Models | Why raw LLMs don't follow instructions well |
| Instruction Datasets | Alpaca, OpenAssistant, FLAN, ShareGPT format |
| Chat Templates | System/user/assistant roles, chat formatting |
| SFT Training | Same cross-entropy loss, but only on assistant responses |
| Full Fine-Tuning vs LoRA | Trade-offs: quality vs memory vs speed |
| LoRA / QLoRA | Low-rank adaptation, quantized fine-tuning |
| Hyperparameters | Lower LR, fewer epochs (1-3), smaller batch size |

**Exercises:**
- Format an instruction dataset into chat template
- Fine-tune your base model using full SFT
- Implement LoRA from scratch (low-rank matrices injected into attention)
- Compare full fine-tune vs LoRA quality and training cost

---

## Lesson 13: Preference Alignment (RLHF / DPO)

**Goal:** Align the model to human preferences — make outputs helpful, harmless, and honest.

| Topic | Details |
|-------|---------|
| Why Alignment Matters | SFT models still produce harmful/unhelpful outputs |
| RLHF Pipeline | Reward model → PPO training → policy optimization |
| Reward Modeling | Training a model to score response quality |
| Proximal Policy Optimization (PPO) | RL algorithm used in original ChatGPT |
| Direct Preference Optimization (DPO) | Simpler alternative — no reward model needed |
| Preference Datasets | Chosen/rejected response pairs |
| KL Penalty | Preventing the model from diverging too far from SFT |

**Exercises:**
- Create a small preference dataset (chosen vs rejected pairs)
- Implement DPO loss function from scratch
- Fine-tune your SFT model with DPO
- Compare outputs before and after alignment

---

## Lesson 14: Quantization & Inference Optimization

**Goal:** Make your model smaller and faster for real-world deployment.

| Topic | Details |
|-------|---------|
| Why Quantize | Reduce memory by 2-4x, speed up inference |
| Weight Quantization | FP32 → FP16 → INT8 → INT4, per-channel vs per-tensor |
| GPTQ | Post-training quantization using calibration data |
| AWQ | Activation-aware weight quantization |
| GGUF Format | Quantized format for llama.cpp / local inference |
| Speculative Decoding | Use a small draft model to speed up a large model |
| Continuous Batching | Serve multiple requests efficiently |
| KV Cache Optimization | Paged attention (vLLM), GQA, MQA |

**Exercises:**
- Quantize your model to INT8 and INT4
- Measure perplexity degradation at each quantization level
- Export your model to GGUF and run it with llama.cpp
- Benchmark inference speed: FP16 vs INT8 vs INT4

---

## Lesson 15: Serving & Deployment

**Goal:** Deploy your LLM as a usable API or application.

| Topic | Details |
|-------|---------|
| Serving Frameworks | vLLM, TGI (Text Generation Inference), llama.cpp server |
| API Design | OpenAI-compatible API, streaming responses (SSE) |
| Throughput Optimization | Continuous batching, dynamic batching, prefill chunking |
| Infrastructure | GPU selection (A100, H100, L40S), cloud vs on-prem |
| Monitoring | Latency tracking, token throughput, error rates |
| Cost Estimation | Tokens/second/dollar, batch vs real-time trade-offs |
| Guardrails | Input/output filtering, content moderation, rate limiting |

**Exercises:**
- Serve your model with vLLM and expose an OpenAI-compatible API
- Build a simple chat UI (Gradio or Streamlit)
- Load test your endpoint and measure p50/p95 latency
- Implement streaming token output

---

## Lesson 16: Retrieval-Augmented Generation (RAG)

**Goal:** Extend your LLM with external knowledge retrieval.

| Topic | Details |
|-------|---------|
| Why RAG | LLMs have knowledge cutoffs and hallucinate facts |
| Embedding Models | Sentence embeddings, bi-encoders, cosine similarity |
| Vector Databases | FAISS, ChromaDB, Qdrant, pgvector |
| Chunking Strategies | Fixed-size, semantic, recursive splitting |
| Retrieval Pipeline | Query → embed → search → rerank → augment prompt |
| Reranking | Cross-encoder rerankers for precision |
| Context Window Management | Fitting retrieved docs within token limits |

**Exercises:**
- Build a document ingestion pipeline (PDF → chunks → embeddings)
- Store embeddings in a vector database
- Implement retrieve-then-generate pipeline
- Compare model answers with and without RAG

---

## Recommended Learning Path

```
Weeks 1-2:   Lessons 1-3   (Foundations)
Weeks 3-4:   Lessons 4-6   (Transformer Architecture)
Weeks 5-6:   Lessons 7-8   (Data & Pre-Training)
Week 7:      Lesson 9      (Scaling)
Week 8:      Lessons 10-11 (Generation & Evaluation)
Weeks 9-10:  Lessons 12-13 (Fine-Tuning & Alignment)
Week 11:     Lesson 14     (Quantization)
Week 12:     Lessons 15-16 (Deployment & RAG)
```

## Key References

| Resource | Type |
|----------|------|
| "Attention Is All You Need" (Vaswani et al., 2017) | Paper — the transformer |
| "Language Models are Few-Shot Learners" (Brown et al., 2020) | Paper — GPT-3 |
| "Training Compute-Optimal LLMs" (Hoffmann et al., 2022) | Paper — Chinchilla scaling laws |
| "LLaMA: Open and Efficient LLMs" (Touvron et al., 2023) | Paper — LLaMA architecture |
| Andrej Karpathy — "Let's build GPT from scratch" | Video — hands-on coding |
| Andrej Karpathy — "Let's build the GPT Tokenizer" | Video — BPE tokenizer |
| Sebastian Raschka — "Build a Large Language Model From Scratch" | Book — complete walkthrough |
