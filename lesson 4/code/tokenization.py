"""
============================================================================
 LESSON 4: Text Representation & Tokenization
============================================================================

 Goal: Convert raw text into numbers the model can process.

 Topics covered:
   1. Why Tokenization    — neural nets need numbers, text is strings
   2. Byte Pair Encoding  — BPE from scratch, the algorithm behind GPT
   3. Encoding & Decoding — text → token IDs → text roundtrip
   4. Special Tokens      — <bos>, <eos>, <pad>, <unk>
   5. Vocabulary Size     — compression vs coverage trade-offs
   6. Token Embeddings    — lookup tables, embedding dimensions, weight tying

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites: Lesson 3 (neural network building blocks)

 Dependencies: NONE — standard library only (re, collections)
============================================================================
"""

import re
import collections

SAMPLE_TEXT = (
    "the cat sat on the mat. the cat ate the rat. "
    "the dog sat on the log. the dog ate the frog. "
    "a low low low low low temperature is cold. "
    "low temperatures are lower than high temperatures."
)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — WHY TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════
#
# Neural networks operate on numbers — vectors, matrices, tensors.
# But language is text. We need a way to convert:
#     "Hello world" → [15496, 995]
#
# This mapping is called tokenization, and it's the very first step
# in any LLM pipeline. Get it wrong, and the model can't learn.
#
# There are three main approaches, each with trade-offs:
#   1. Character-level:  "hello" → ['h', 'e', 'l', 'l', 'o']
#   2. Word-level:       "hello world" → ['hello', 'world']
#   3. Subword-level:    "unhappiness" → ['un', 'happiness']  (BPE)
#
# Modern LLMs (GPT, LLaMA, Claude) all use subword tokenization.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 60)
print("PART 1: WHY TOKENIZATION")
print("=" * 60)


# --- 1.1 The fundamental problem -----------------------------------------

text = "Hello World"
print(f"\nText:         '{text}'")
print(f"Characters:   {list(text)}")
print(f"Char codes:   {[ord(c) for c in text]}")
print(f"\nNeural nets need FIXED-SIZE integer IDs, not raw characters.")
print(f"Tokenization = the mapping from text to numbers and back.\n")


# --- 1.2 Character-level tokenization ------------------------------------

"""
WHY THIS MATTERS:
Character-level is the simplest approach: each character gets its own ID.
Tiny vocabulary (~256 for ASCII/UTF-8 bytes), but sequences become VERY long.
"Hello" = 5 tokens. A 1000-word essay = ~5000 tokens.
LLMs have limited context windows, so long sequences are expensive.
"""

def char_tokenize(text):
    chars = sorted(set(text))
    char_to_id = {c: i for i, c in enumerate(chars)}
    return [char_to_id[c] for c in text], char_to_id


char_ids, char_vocab = char_tokenize("the cat sat")
print("--- Character-level ---")
print(f"Text:     'the cat sat'")
print(f"Tokens:   {char_ids}")
print(f"Vocab:    {char_vocab}")
print(f"Vocab size: {len(char_vocab)}")
print(f"Sequence length: {len(char_ids)} (every char = 1 token)")


# --- 1.3 Word-level tokenization -----------------------------------------

"""
WHY THIS MATTERS:
Word-level tokenization splits on whitespace. Each word gets an ID.
Short sequences (good!), but HUGE vocabulary (bad!).
English has 170,000+ words. Misspellings, neologisms, code, other
languages = the vocabulary explodes. Any word not in the vocab
becomes <unk> (unknown) and the model can't process it.
"""

def word_tokenize(text):
    words = text.split()
    vocab = sorted(set(words))
    word_to_id = {w: i for i, w in enumerate(vocab)}
    return [word_to_id[w] for w in words], word_to_id


word_ids, word_vocab = word_tokenize("the cat sat on the mat")
print(f"\n--- Word-level ---")
print(f"Text:     'the cat sat on the mat'")
print(f"Tokens:   {word_ids}")
print(f"Vocab size: {len(word_vocab)}")
print(f"Sequence length: {len(word_ids)} (every word = 1 token)")


# --- 1.4 The subword sweet spot ------------------------------------------

print(f"\n--- Trade-off comparison ---")
long_text = "The transformer architecture revolutionized natural language processing."
char_toks = list(long_text)
word_toks = long_text.split()
print(f"Text: '{long_text}'")
print(f"  Character tokens: {len(char_toks)} tokens (vocab ~100)")
print(f"  Word tokens:      {len(word_toks)} tokens  (vocab 170k+)")
print(f"  Subword (BPE):    ~10 tokens  (vocab 32k-100k)")
print(f"\nSubword = best of both worlds:")
print(f"  - Common words stay whole: 'the' → 1 token")
print(f"  - Rare words split: 'revolutionized' → 'revolution' + 'ized'")
print(f"  - Never see <unk>: can always fall back to characters")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — BYTE PAIR ENCODING (BPE)
# ═══════════════════════════════════════════════════════════════════════════
#
# BPE was originally a data compression algorithm. Sennrich et al. (2016)
# adapted it for NLP, and it became the tokenizer behind GPT-2/3/4.
#
# The algorithm is beautifully simple:
#   1. Start with character-level tokens
#   2. Count all adjacent pairs
#   3. Merge the most frequent pair into a new token
#   4. Repeat steps 2-3 for N iterations (N = desired merges)
#
# Each merge creates a new token. After N merges, you have a vocabulary
# of (base_chars + N) tokens. The order of merges IS the tokenizer.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 2: BYTE PAIR ENCODING (BPE)")
print("=" * 60)


# --- 2.1 Prepare training corpus -----------------------------------------

def get_word_freqs(text):
    """Split text into words and count frequencies.
    Each word is stored as a tuple of characters + end-of-word marker '_'.
    The marker ensures 'the' at end-of-word differs from 'the' mid-word.
    """
    words = text.lower().split()
    freqs = collections.Counter()
    for word in words:
        token_seq = tuple(word) + ("_",)
        freqs[token_seq] += 1
    return freqs


word_freqs = get_word_freqs(SAMPLE_TEXT)
print(f"\nTraining text: '{SAMPLE_TEXT[:60]}...'")
print(f"\nWord frequencies (as character tuples):")
for word, count in sorted(word_freqs.items(), key=lambda x: -x[1])[:8]:
    print(f"  {''.join(word):20s} count={count}")


# --- 2.2 Count pair frequencies ------------------------------------------

def get_pair_freqs(word_freqs):
    """Count how often each adjacent pair appears across all words."""
    pair_counts = collections.Counter()
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += freq
    return pair_counts


pair_freqs = get_pair_freqs(word_freqs)
print(f"\nTop 10 most frequent pairs:")
for pair, count in pair_freqs.most_common(10):
    print(f"  {pair[0]:4s} + {pair[1]:4s} = {''.join(pair):6s}  (count={count})")


# --- 2.3 Merge a pair ----------------------------------------------------

def merge_pair(word_freqs, pair):
    """Replace every occurrence of `pair` in all words with merged token."""
    new_freqs = {}
    merged = pair[0] + pair[1]
    for word, freq in word_freqs.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                new_word.append(merged)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_freqs[tuple(new_word)] = freq
    return new_freqs


# --- 2.4 Run BPE training! -----------------------------------------------

def train_bpe(text, num_merges):
    """Train BPE tokenizer: learn `num_merges` merge rules from text."""
    word_freqs = get_word_freqs(text)

    base_chars = set()
    for word in word_freqs:
        for ch in word:
            base_chars.add(ch)
    vocab = sorted(base_chars)

    merges = []

    print(f"\nInitial vocabulary ({len(vocab)} chars): {vocab}")
    print(f"\nLearning {num_merges} merges:")

    for step in range(num_merges):
        pair_freqs = get_pair_freqs(word_freqs)
        if not pair_freqs:
            print(f"  No more pairs to merge at step {step}")
            break

        best_pair = pair_freqs.most_common(1)[0]
        pair, count = best_pair
        merged = pair[0] + pair[1]

        merges.append(pair)
        vocab.append(merged)
        word_freqs = merge_pair(word_freqs, pair)

        print(f"  Step {step:2d}: merge '{pair[0]}' + '{pair[1]}' → '{merged}'  (freq={count})")

    print(f"\nFinal vocabulary ({len(vocab)} tokens): {vocab[:20]}{'...' if len(vocab) > 20 else ''}")
    return vocab, merges, word_freqs


NUM_MERGES = 20
vocab, merges, final_word_freqs = train_bpe(SAMPLE_TEXT, NUM_MERGES)


# --- 2.5 Inspect the learned vocabulary -----------------------------------

print(f"\nAll merge rules (in order):")
for i, (a, b) in enumerate(merges):
    print(f"  {i:2d}: '{a}' + '{b}' → '{a}{b}'")

print(f"\nFinal tokenized words:")
for word, freq in sorted(final_word_freqs.items(), key=lambda x: -x[1])[:10]:
    print(f"  {str(word):40s} (freq={freq})")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — ENCODING AND DECODING
# ═══════════════════════════════════════════════════════════════════════════
#
# A tokenizer needs two operations:
#   encode: text → list of token IDs
#   decode: list of token IDs → text
#
# Encoding applies the merge rules in learned order.
# Decoding looks up each ID in the vocabulary and concatenates.
#
# The ROUNDTRIP must be lossless: decode(encode(text)) == text
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 3: ENCODING AND DECODING")
print("=" * 60)


class BPETokenizer:
    """A minimal BPE tokenizer built from scratch."""

    def __init__(self, vocab, merges):
        self.vocab = vocab
        self.merges = merges
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}

    def _apply_merges(self, tokens):
        """Apply all learned merges to a list of tokens, in order."""
        for pair in self.merges:
            new_tokens = []
            i = 0
            merged = pair[0] + pair[1]
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text):
        """Convert text to a list of token IDs."""
        words = text.lower().split()
        all_ids = []
        for word in words:
            chars = list(word) + ["_"]
            tokens = self._apply_merges(chars)
            for tok in tokens:
                if tok in self.token_to_id:
                    all_ids.append(self.token_to_id[tok])
                else:
                    for ch in tok:
                        all_ids.append(self.token_to_id.get(ch, -1))
        return all_ids

    def decode(self, ids):
        """Convert token IDs back to text."""
        tokens = [self.id_to_token.get(i, "?") for i in ids]
        text = "".join(tokens)
        text = text.replace("_", " ")
        return text.strip()

    @property
    def vocab_size(self):
        return len(self.vocab)


tokenizer = BPETokenizer(vocab, merges)

# --- 3.1 Encode demo -----------------------------------------------------

test_texts = [
    "the cat sat",
    "the dog ate",
    "low temperature",
]

print(f"\n--- Encoding ---")
for text in test_texts:
    ids = tokenizer.encode(text)
    tokens = [tokenizer.id_to_token[i] for i in ids]
    print(f"  '{text}'")
    print(f"    tokens: {tokens}")
    print(f"    ids:    {ids}")
    print()

# --- 3.2 Decode demo -----------------------------------------------------

print(f"--- Decoding ---")
for text in test_texts:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"  Original: '{text}'")
    print(f"  Decoded:  '{decoded}'")
    match = text.lower().strip() == decoded.strip()
    print(f"  Roundtrip match: {match}")
    print()

# --- 3.3 Tokenization details -------------------------------------------

print(f"--- Vocabulary info ---")
print(f"  Vocab size: {tokenizer.vocab_size}")
print(f"  Token → ID mapping (first 15):")
for tok, tid in list(tokenizer.token_to_id.items())[:15]:
    print(f"    '{tok}' → {tid}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — SPECIAL TOKENS
# ═══════════════════════════════════════════════════════════════════════════
#
# Real tokenizers include special tokens that don't come from the text:
#
#   <bos>  Beginning of Sequence — marks the start of a text
#   <eos>  End of Sequence      — marks the end (model should stop generating)
#   <pad>  Padding              — fills shorter sequences to a uniform length
#   <unk>  Unknown              — fallback for tokens not in the vocabulary
#
# These tokens get their own IDs and embeddings. The model learns to
# use them: <bos> triggers text generation, <eos> signals "I'm done."
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 4: SPECIAL TOKENS")
print("=" * 60)


SPECIAL_TOKENS = {
    "<pad>": 0,
    "<unk>": 1,
    "<bos>": 2,
    "<eos>": 3,
}


class BPETokenizerWithSpecials:
    """BPE tokenizer extended with special tokens."""

    def __init__(self, vocab, merges, special_tokens):
        self.special_tokens = special_tokens
        num_special = len(special_tokens)

        self.token_to_id = dict(special_tokens)
        self.id_to_token = {v: k for k, v in special_tokens.items()}

        for i, token in enumerate(vocab):
            idx = i + num_special
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

        self.merges = merges
        self.pad_id = special_tokens["<pad>"]
        self.unk_id = special_tokens["<unk>"]
        self.bos_id = special_tokens["<bos>"]
        self.eos_id = special_tokens["<eos>"]

    def _apply_merges(self, tokens):
        for pair in self.merges:
            new_tokens = []
            i = 0
            merged = pair[0] + pair[1]
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def encode(self, text, add_special=True):
        """Encode text to IDs, optionally wrapping with <bos>/<eos>."""
        words = text.lower().split()
        ids = []
        if add_special:
            ids.append(self.bos_id)
        for word in words:
            chars = list(word) + ["_"]
            tokens = self._apply_merges(chars)
            for tok in tokens:
                ids.append(self.token_to_id.get(tok, self.unk_id))
        if add_special:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids):
        """Decode IDs to text, skipping special tokens."""
        skip = set(self.special_tokens.values())
        tokens = []
        for i in ids:
            if i in skip:
                continue
            tokens.append(self.id_to_token.get(i, "?"))
        text = "".join(tokens).replace("_", " ")
        return text.strip()

    def pad_sequences(self, batch, max_len=None):
        """Pad a batch of sequences to uniform length."""
        if max_len is None:
            max_len = max(len(seq) for seq in batch)
        padded = []
        for seq in batch:
            padded.append(seq + [self.pad_id] * (max_len - len(seq)))
        return padded

    @property
    def vocab_size(self):
        return len(self.token_to_id)


tok = BPETokenizerWithSpecials(vocab, merges, SPECIAL_TOKENS)

# --- 4.1 Special tokens in action ----------------------------------------

print(f"\n--- Special token IDs ---")
for name, tid in SPECIAL_TOKENS.items():
    print(f"  {name:6s} → ID {tid}")

text = "the cat sat"
ids_with = tok.encode(text, add_special=True)
ids_without = tok.encode(text, add_special=False)
labels_with = [tok.id_to_token[i] for i in ids_with]

print(f"\n--- Encoding with special tokens ---")
print(f"  Text: '{text}'")
print(f"  Without specials: {ids_without}")
print(f"  With specials:    {ids_with}")
print(f"  Token labels:     {labels_with}")

# --- 4.2 Padding demo ---------------------------------------------------

print(f"\n--- Padding a batch ---")
batch_texts = ["the cat", "the dog sat on the mat", "low"]
batch_ids = [tok.encode(t) for t in batch_texts]

print(f"  Before padding:")
for t, ids in zip(batch_texts, batch_ids):
    print(f"    '{t}' → {ids} (len={len(ids)})")

padded = tok.pad_sequences(batch_ids)
print(f"\n  After padding (to length {len(padded[0])}):")
for t, ids in zip(batch_texts, padded):
    labels = [tok.id_to_token[i] for i in ids]
    print(f"    '{t}' → {ids}")
    print(f"      labels: {labels}")

# --- 4.3 Unknown token demo ----------------------------------------------

print(f"\n--- Unknown token handling ---")
exotic_text = "xyz quantum"
exotic_ids = tok.encode(exotic_text)
exotic_labels = [tok.id_to_token[i] for i in exotic_ids]
print(f"  Text:   '{exotic_text}'")
print(f"  IDs:    {exotic_ids}")
print(f"  Labels: {exotic_labels}")
print(f"  Characters not in vocab → <unk> (ID {tok.unk_id})")

print(f"\n  Total vocab size (with specials): {tok.vocab_size}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — VOCABULARY SIZE
# ═══════════════════════════════════════════════════════════════════════════
#
# Vocabulary size is a critical hyperparameter:
#
#   Small vocab (256-1000):
#     + Tiny embedding table, less memory
#     - Very long sequences (more tokens per sentence)
#     - Slower training and inference
#
#   Large vocab (50k-100k):
#     + Short sequences (better compression)
#     + Common words = single token
#     - Huge embedding table (vocab_size × d_model parameters)
#     - Rare tokens get poor embeddings (few training examples)
#
# GPT-2: 50,257 | GPT-4: ~100k | LLaMA: 32,000 | LLaMA-3: 128,256
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 5: VOCABULARY SIZE")
print("=" * 60)


def train_and_measure(text, num_merges):
    """Train BPE with given merges and return compression stats."""
    word_freqs = get_word_freqs(text)

    base_chars = set()
    for word in word_freqs:
        for ch in word:
            base_chars.add(ch)
    v = sorted(base_chars)
    m = []

    for _ in range(num_merges):
        pair_freqs = get_pair_freqs(word_freqs)
        if not pair_freqs:
            break
        best_pair = pair_freqs.most_common(1)[0][0]
        m.append(best_pair)
        v.append(best_pair[0] + best_pair[1])
        word_freqs = merge_pair(word_freqs, best_pair)

    total_tokens = sum(
        len(word) * freq for word, freq in word_freqs.items()
    )
    return len(v), total_tokens, v, m


# Measure character-level baseline
char_count = len(SAMPLE_TEXT.lower().replace(" ", "_"))
print(f"\nTest text: '{SAMPLE_TEXT[:50]}...'")
print(f"Character count (baseline): {char_count}")

print(f"\n--- Compression at different vocab sizes ---")
print(f"{'Merges':>8s} | {'Vocab':>6s} | {'Tokens':>7s} | {'Ratio':>7s} | {'Visual'}")
print("-" * 65)

for n_merges in [0, 5, 10, 20, 40, 60]:
    v_size, n_tokens, _, _ = train_and_measure(SAMPLE_TEXT, n_merges)
    ratio = char_count / max(n_tokens, 1)
    bar_len = int(ratio * 10)
    bar = "█" * bar_len + "░" * (20 - bar_len)
    print(f"{n_merges:8d} | {v_size:6d} | {n_tokens:7d} | {ratio:6.2f}x | {bar}")


# --- 5.2 Real-world vocab size comparison --------------------------------

print(f"""
--- Real-world tokenizer vocab sizes ---
  GPT-2:    50,257 tokens  (BPE)
  GPT-4:   ~100,000 tokens (BPE)
  LLaMA:    32,000 tokens  (SentencePiece BPE)
  LLaMA-3: 128,256 tokens  (BPE)
  BERT:     30,522 tokens  (WordPiece)

Rule of thumb: bigger vocab = shorter sequences = faster, but
the embedding table grows as vocab_size × d_model.
  GPT-2: 50,257 × 768 = 38.6M params just for embeddings!
  LLaMA-3: 128,256 × 4,096 = 525M params just for embeddings!
""")

# --- 5.3 Tokens per word metric ------------------------------------------

test_sentences = [
    "The cat sat on the mat.",
    "Tokenization is the first step in NLP.",
    "def train_model(learning_rate=0.001):",
    "Supercalifragilisticexpialidocious!",
]

print(f"--- Tokens per word for our BPE tokenizer ---")
for sent in test_sentences:
    ids = tokenizer.encode(sent)
    n_words = len(sent.split())
    tpw = len(ids) / max(n_words, 1)
    print(f"  '{sent}'")
    print(f"    words={n_words}, tokens={len(ids)}, tokens/word={tpw:.1f}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — TOKEN EMBEDDINGS
# ═══════════════════════════════════════════════════════════════════════════
#
# After tokenization, each token ID becomes a dense vector through an
# EMBEDDING LOOKUP TABLE. This is how the model "understands" tokens.
#
#   token_id → embedding_table[token_id] → d_model-dimensional vector
#
# In PyTorch: nn.Embedding(vocab_size, d_model)
# Under the hood: it's just a matrix of shape (vocab_size, d_model),
# and embedding lookup = indexing one row.
#
# Weight tying: many LLMs share the embedding table between the input
# embeddings and the final output projection (LM head). This saves
# parameters and often improves performance.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("PART 6: TOKEN EMBEDDINGS")
print("=" * 60)

import random
random.seed(42)


# --- 6.1 Pure Python embedding table --------------------------------------

def make_embedding_table(vocab_size, d_model):
    """Create a random embedding table (dict of token_id → vector)."""
    return {
        i: [random.gauss(0, 0.02) for _ in range(d_model)]
        for i in range(vocab_size)
    }


def embed_tokens(token_ids, embedding_table):
    """Look up embeddings for a list of token IDs."""
    return [embedding_table[tid] for tid in token_ids]


VOCAB_SIZE = tokenizer.vocab_size
D_MODEL = 8

embedding_table = make_embedding_table(VOCAB_SIZE, D_MODEL)

print(f"\n--- Embedding lookup ---")
print(f"  Vocab size: {VOCAB_SIZE}")
print(f"  Embedding dim (d_model): {D_MODEL}")
print(f"  Table size: {VOCAB_SIZE} × {D_MODEL} = {VOCAB_SIZE * D_MODEL} parameters")

text = "the cat"
ids = tokenizer.encode(text)
embeddings = embed_tokens(ids, embedding_table)

print(f"\n  Text: '{text}'")
print(f"  Token IDs: {ids}")
for i, (tid, emb) in enumerate(zip(ids, embeddings)):
    tok_str = tokenizer.id_to_token.get(tid, "?")
    vec_str = "[" + ", ".join(f"{v: .3f}" for v in emb[:4]) + ", ...]"
    print(f"    ID {tid:3d} ('{tok_str}') → {vec_str}")


# --- 6.2 How nn.Embedding works ------------------------------------------

print(f"""
--- How nn.Embedding works in PyTorch ---

  # Create embedding table
  emb = nn.Embedding(num_embeddings=50257, embedding_dim=768)

  # emb.weight is a matrix of shape (50257, 768)
  # Each row is the learned vector for one token

  # Lookup: just index into the matrix
  token_ids = torch.tensor([15496, 995])  # "Hello world"
  vectors = emb(token_ids)                # shape: (2, 768)

  # That's it! Embedding is just a fancy indexing operation.
  # During training, backprop updates the embedding vectors
  # so that similar tokens get similar vectors.
""")


# --- 6.3 Cosine similarity between embeddings ----------------------------

import math


def cosine_sim(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


print(f"--- Embedding similarity (random, untrained) ---")
sample_ids = list(range(min(6, VOCAB_SIZE)))
sample_tokens = [tokenizer.id_to_token.get(i, f"tok_{i}") for i in sample_ids]

print(f"  Tokens: {sample_tokens}")
print(f"  Cosine similarities:")
for i in range(len(sample_ids)):
    for j in range(i + 1, len(sample_ids)):
        sim = cosine_sim(embedding_table[sample_ids[i]], embedding_table[sample_ids[j]])
        print(f"    '{sample_tokens[i]}' vs '{sample_tokens[j]}': {sim:.4f}")

print(f"\n  (Random embeddings ≈ 0 similarity. After training, semantically")
print(f"   similar tokens like 'cat'/'dog' would have high similarity.)")


# --- 6.4 Weight tying concept -------------------------------------------

print(f"""
--- Weight Tying ---

  In a language model, there are TWO places that use token ↔ vector:
    1. Input:  token_id → embedding vector  (nn.Embedding)
    2. Output: hidden_state → logits        (nn.Linear, called "LM head")

  The LM head weight has the SAME shape as the embedding table:
    embedding: (vocab_size, d_model)
    lm_head:   (vocab_size, d_model)

  Weight tying: share the SAME matrix for both!
    model.lm_head.weight = model.embedding.weight

  Benefits:
    - Saves vocab_size × d_model parameters
    - GPT-2 saves 38.6M params (768 × 50,257)
    - Forces consistent token representations
    - Often improves generation quality

  Used by: GPT-2, LLaMA, T5, many modern LLMs
""")


# --- 6.5 Parameter cost of embeddings ------------------------------------

print(f"--- Embedding parameter costs ---")
configs = [
    ("GPT-2 Small", 50257, 768),
    ("GPT-2 XL", 50257, 1600),
    ("LLaMA-7B", 32000, 4096),
    ("LLaMA-3-8B", 128256, 4096),
]

for name, v, d in configs:
    params = v * d
    mb = params * 4 / 1e6
    savings = params * 4 / 1e6
    print(f"  {name:15s}: {v:>7,} × {d:>5,} = {params:>12,} params ({mb:>7.1f} MB float32)")

print(f"\n  Weight tying saves exactly this many params per model.")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EXERCISES")
print("=" * 60)
print("""
Exercise 1: Implement BPE Merge
---------------------------------
Without looking at the code above, implement the BPE training loop:
  a) Write get_pair_freqs(word_freqs) → Counter of pair frequencies
  b) Write merge_pair(word_freqs, pair) → new word_freqs with pair merged
  c) Run 10 merges on: "aabdaaabac" (split into characters first)
  d) What is the final tokenization? Verify by hand.

Exercise 2: Encode/Decode Roundtrip
-------------------------------------
  a) Create a BPETokenizer using the vocab and merges from Part 2
  b) Encode these strings and decode them back:
     - "the cat"
     - "the dog sat on the mat"
     - "low temperatures"
  c) Verify: decoded == original.lower() for all cases
  d) What happens if you encode a word with characters NOT in the vocab?

Exercise 3: Compression Ratio Analysis
----------------------------------------
  a) Train BPE with 0, 10, 20, 50, and 100 merges on this text:
     "the quick brown fox jumps over the lazy dog " * 20
  b) For each, count the total tokens needed to encode the text
  c) Compute compression ratio = original_chars / total_tokens
  d) Plot (or print) compression ratio vs number of merges
  e) At what point does adding more merges stop helping?

Exercise 4: English vs Code Tokenization
------------------------------------------
  a) Train a BPE tokenizer (30 merges) on English text:
     "Natural language processing is a subfield of linguistics."
  b) Train another (30 merges) on Python code:
     "def forward(self, x): return self.linear(self.relu(x))"
  c) Compare the learned merge rules. How do they differ?
  d) What does this tell you about why LLMs need large vocabularies
     to handle both English AND code well?

Exercise 5: Simple WordPiece Tokenizer
-----------------------------------------
WordPiece (used by BERT) is similar to BPE but scores pairs differently:
  score(pair) = freq(pair) / (freq(first) * freq(second))

Instead of merging the most FREQUENT pair, merge the pair with
the highest SCORE (which favors pairs where both parts are rare).

  a) Implement WordPiece scoring
  b) Train on the same SAMPLE_TEXT with 15 merges
  c) Compare the merge order to BPE — what's different?
  d) Why might WordPiece produce different results on rare words?
""")
