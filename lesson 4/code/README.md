# Lesson 4: Text Representation & Tokenization

Convert raw text into numbers the model can process — the very first step in any LLM pipeline.

## Prerequisites

- Python 3.10+
- No external dependencies (uses only `re` and `collections` from the standard library)
- Lesson 3 completed (neural network building blocks)

## How to Run

```bash
python3 tokenization.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Why Tokenization
The fundamental problem: neural nets need numbers, text is strings. Character-level vs word-level vs subword trade-offs. Why modern LLMs all use subword tokenization.

**Key functions:** `char_tokenize()`, `word_tokenize()`

### Part 2 — Byte Pair Encoding (BPE)
Full BPE implementation from scratch. Start with character-level tokens, count adjacent pair frequencies, merge the most frequent pair, repeat. This is the exact algorithm behind GPT-2/3/4 tokenizers.

**Key functions:** `get_word_freqs()`, `get_pair_freqs()`, `merge_pair()`, `train_bpe()`

### Part 3 — Encoding and Decoding
Build a complete `BPETokenizer` class with `encode(text → token_ids)` and `decode(token_ids → text)` methods. Verify the lossless roundtrip: `decode(encode(text)) == text`.

**Key class:** `BPETokenizer` with `encode()`, `decode()`, `_apply_merges()`

### Part 4 — Special Tokens
Implement `<bos>`, `<eos>`, `<pad>`, `<unk>` tokens. Demonstrate padding a batch of sequences to uniform length, and handling unknown characters.

**Key class:** `BPETokenizerWithSpecials` with `pad_sequences()`

### Part 5 — Vocabulary Size
Measure compression ratio at different vocabulary sizes (0 to 60 merges). Compare real-world tokenizer sizes (GPT-2, LLaMA, LLaMA-3). Show the trade-off: bigger vocab = shorter sequences but larger embedding table.

**Key function:** `train_and_measure()`

### Part 6 — Token Embeddings
Pure Python embedding lookup table. Explain how `nn.Embedding` works under the hood (matrix indexing), cosine similarity between embeddings, and weight tying (sharing embeddings with the LM head).

**Key functions:** `make_embedding_table()`, `embed_tokens()`, `cosine_sim()`

## Exercises

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | BPE Merge | Implement get_pair_freqs and merge_pair from scratch |
| 2 | Encode/Decode Roundtrip | Verify lossless text → IDs → text conversion |
| 3 | Compression Ratio | Measure tokens vs merges, find diminishing returns |
| 4 | English vs Code | Compare tokenization patterns across domains |
| 5 | WordPiece Tokenizer | Implement BERT's alternative scoring for merges |

## Key Takeaways

- **Tokenization** is the bridge between text and numbers — the first step in every LLM
- **BPE** iteratively merges the most frequent character pairs to build a subword vocabulary
- **Subword tokenization** balances vocabulary size against sequence length
- **Special tokens** (`<bos>`, `<eos>`, `<pad>`, `<unk>`) provide structure to sequences
- **Vocabulary size** directly affects model size: the embedding table is `vocab_size × d_model` parameters
- **Weight tying** shares the embedding matrix with the output layer, saving millions of parameters

## What's Next

[Lesson 5: Attention Mechanism](../../lesson%205/) — The core innovation behind transformers: queries, keys, values, scaled dot-product attention, multi-head attention, and causal masking.
