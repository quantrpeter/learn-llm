"""
============================================================================
 LESSON 7: Data Pipeline for Pre-Training
============================================================================

 Goal: Build the complete data pipeline that feeds text into an LLM during
       pre-training — from raw documents to GPU-ready token batches.

 Topics covered:
   1. Data Sources        — Common Crawl, Wikipedia, FineWeb; simulated corpus
   2. Data Cleaning       — deduplication, length filtering, quality filtering
   3. Data Formatting     — sequence packing vs padding, chunking into fixed len
   4. DataLoader Design   — custom Dataset, DataLoader, streaming vs map-style
   5. Data Mixing         — weighted sampling from wiki, books, code sources
   6. Tokenization Integ. — char-level tokenizer, token batches for training

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites:
   - Lesson 4 (tokenization, BPE concepts)
   - Lesson 6 (transformer architecture — you know what the model expects)
============================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, IterableDataset, WeightedRandomSampler
import hashlib
import random
import time
import os
import string

torch.manual_seed(42)
random.seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — DATA SOURCES
# ═══════════════════════════════════════════════════════════════════════════
#
# Training an LLM requires ENORMOUS amounts of text. GPT-3 trained on
# ~300 billion tokens. LLaMA-2 trained on 2 TRILLION tokens.
#
# Where does all this text come from?
#
#   Common Crawl  — petabytes of raw web pages, biggest source (~60%)
#   Wikipedia     — high-quality encyclopedic text (~3-5%)
#   Books         — BookCorpus, Project Gutenberg (~5-10%)
#   Code          — GitHub, StackOverflow (~5-10%)
#   FineWeb       — cleaned/filtered Common Crawl by HuggingFace (15T tokens)
#   The Pile      — 800GB curated mix by EleutherAI
#
# In practice you DON'T train on raw Common Crawl — it's full of spam,
# duplicates, and garbage. FineWeb and The Pile are pre-cleaned versions.
#
# For this lesson we simulate a text corpus so everything runs locally
# with no downloads needed.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: DATA SOURCES")
print("=" * 70)


# --- 1.1 Simulated corpus ------------------------------------------------

WIKI_TOPICS = [
    "Neural networks are computing systems inspired by biological neural networks. "
    "They consist of layers of interconnected nodes that process information. "
    "Deep learning uses neural networks with many layers to learn representations.",

    "The transformer architecture was introduced in the paper Attention Is All You Need. "
    "It replaced recurrent networks with self-attention mechanisms. "
    "Transformers process all tokens in parallel rather than sequentially.",

    "Language models predict the probability of the next word in a sequence. "
    "Modern language models like GPT use billions of parameters. "
    "Pre-training on large text corpora gives them broad knowledge.",

    "Gradient descent is an optimization algorithm used to train neural networks. "
    "The learning rate controls how large each update step is. "
    "Adam and AdamW are the most popular optimizers for training transformers.",

    "Tokenization converts raw text into numerical tokens for the model. "
    "Byte pair encoding is the most common subword tokenization method. "
    "A typical vocabulary size for LLMs is between 32000 and 128000 tokens.",
]

BOOK_PASSAGES = [
    "It was the best of times, it was the worst of times. The age of wisdom "
    "and the age of foolishness were upon us. We had everything before us "
    "and nothing before us. The spring of hope and winter of despair.",

    "Call me Ishmael. Some years ago, having little money in my purse, "
    "I thought I would sail about and see the watery part of the world. "
    "It is a way I have of driving off the spleen and regulating circulation.",

    "In a hole in the ground there lived a hobbit. Not a nasty dirty wet hole "
    "filled with the ends of worms, but a comfortable hobbit hole with a round "
    "door painted green and a brass knob in the exact middle.",
]

CODE_SNIPPETS = [
    "def forward(self, x):\n    # Apply attention followed by feed-forward\n"
    "    residual = x\n    x = self.norm1(x)\n    x = self.attn(x)\n"
    "    x = residual + x\n    return x\n",

    "class DataLoader:\n    def __init__(self, dataset, batch_size=32):\n"
    "        self.dataset = dataset\n        self.batch_size = batch_size\n"
    "    def __iter__(self):\n        indices = list(range(len(self.dataset)))\n"
    "        random.shuffle(indices)\n        for i in range(0, len(indices), self.batch_size):\n"
    "            yield [self.dataset[j] for j in indices[i:i+self.batch_size]]\n",

    "import torch\nimport torch.nn as nn\n\nmodel = nn.Transformer(\n"
    "    d_model=512,\n    nhead=8,\n    num_encoder_layers=6\n)\n"
    "optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)\n",
]


def generate_corpus(n_wiki=50, n_books=30, n_code=20):
    """
    WHY THIS MATTERS:
    Real pre-training corpora contain trillions of tokens from diverse sources.
    We simulate this at small scale — same pipeline logic, manageable size.
    Each "document" has a source tag so we can track provenance through
    the entire cleaning and mixing pipeline.
    """
    corpus = []
    for i in range(n_wiki):
        base = random.choice(WIKI_TOPICS)
        doc = base + f" This is document number {i} from the encyclopedia."
        corpus.append({"text": doc, "source": "wiki", "id": f"wiki_{i}"})

    for i in range(n_books):
        base = random.choice(BOOK_PASSAGES)
        doc = base + f" Chapter {i} continues with further adventures."
        corpus.append({"text": doc, "source": "books", "id": f"book_{i}"})

    for i in range(n_code):
        base = random.choice(CODE_SNIPPETS)
        doc = base + f"\n# module_{i}.py\n"
        corpus.append({"text": doc, "source": "code", "id": f"code_{i}"})

    # Add some intentional duplicates and low-quality docs for cleaning
    for i in range(10):
        corpus.append({"text": corpus[0]["text"], "source": "wiki", "id": f"dup_{i}"})

    for i in range(5):
        corpus.append({"text": "!@#$%^&*()" * 20, "source": "web", "id": f"junk_{i}"})

    for i in range(5):
        corpus.append({"text": "hi", "source": "web", "id": f"short_{i}"})

    random.shuffle(corpus)
    return corpus


corpus = generate_corpus()

print(f"\nSimulated corpus:")
print(f"  Total documents:  {len(corpus)}")
print(f"  Sources:")
source_counts = {}
for doc in corpus:
    source_counts[doc["source"]] = source_counts.get(doc["source"], 0) + 1
for src, count in sorted(source_counts.items()):
    print(f"    {src:8s}: {count:3d} docs")

total_chars = sum(len(d["text"]) for d in corpus)
print(f"  Total characters: {total_chars:,}")
print(f"\nSample document ({corpus[0]['source']}):")
print(f"  {corpus[0]['text'][:120]}...")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════
#
# Raw web text is MESSY. Before training, you must clean it:
#
#   1. Deduplication     — remove exact and near-duplicate documents
#   2. Length filtering   — remove docs that are too short or too long
#   3. Quality filtering  — remove docs with too many special chars, etc.
#
# FineWeb's cleaning pipeline removes ~90% of raw Common Crawl.
# The quality of your training data matters MORE than quantity.
# "Garbage in, garbage out" is the #1 rule of data pipelines.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: DATA CLEANING")
print("=" * 70)

torch.manual_seed(42)


# --- 2.1 Hash-based deduplication -----------------------------------------

def deduplicate(documents):
    """
    WHY THIS MATTERS:
    Common Crawl contains massive amounts of duplicated content — boilerplate
    headers, copied articles, scraped mirrors. Training on duplicates wastes
    compute and can cause the model to memorize/regurgitate specific passages.

    We use SHA-256 hashing for exact dedup. Production systems also use
    MinHash / SimHash for near-duplicate detection (fuzzy matching).
    """
    seen_hashes = set()
    unique_docs = []
    duplicates_removed = 0

    for doc in documents:
        text_hash = hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_docs.append(doc)
        else:
            duplicates_removed += 1

    return unique_docs, duplicates_removed


deduped, n_dups = deduplicate(corpus)
print(f"\n--- Deduplication (SHA-256 hash) ---")
print(f"  Before: {len(corpus)} documents")
print(f"  Duplicates removed: {n_dups}")
print(f"  After:  {len(deduped)} documents")


# --- 2.2 Length filtering -------------------------------------------------

def filter_by_length(documents, min_chars=50, max_chars=100_000):
    """
    WHY THIS MATTERS:
    Very short documents (< 50 chars) are usually navigation elements, error
    pages, or meaningless fragments — they add noise. Very long documents
    can dominate training and may be auto-generated content (e.g., log dumps).
    """
    passed = []
    removed = {"too_short": 0, "too_long": 0}

    for doc in documents:
        length = len(doc["text"])
        if length < min_chars:
            removed["too_short"] += 1
        elif length > max_chars:
            removed["too_long"] += 1
        else:
            passed.append(doc)

    return passed, removed


length_filtered, length_stats = filter_by_length(deduped)
print(f"\n--- Length Filtering (min=50, max=100K chars) ---")
print(f"  Before: {len(deduped)} documents")
print(f"  Too short: {length_stats['too_short']}")
print(f"  Too long:  {length_stats['too_long']}")
print(f"  After:  {len(length_filtered)} documents")


# --- 2.3 Quality filtering -----------------------------------------------

def quality_score(text):
    """
    WHY THIS MATTERS:
    Even after dedup and length filtering, many documents are low quality:
    cookie banners, keyword-stuffed SEO pages, random character sequences.
    Quality heuristics catch these. FineWeb uses ~15 quality filters.

    We implement three simple signals:
      - special_char_ratio: fraction of non-alphanumeric characters
      - avg_line_length: very short avg lines suggest lists/navigation
      - alpha_ratio: fraction of alphabetic characters (vs numbers/symbols)
    """
    if len(text) == 0:
        return {"special_char_ratio": 1.0, "avg_line_length": 0, "alpha_ratio": 0.0}

    alpha_count = sum(1 for c in text if c.isalpha())
    special_count = sum(1 for c in text if not c.isalnum() and not c.isspace())

    lines = text.split("\n")
    avg_line_len = sum(len(l) for l in lines) / max(len(lines), 1)

    return {
        "special_char_ratio": special_count / len(text),
        "avg_line_length": avg_line_len,
        "alpha_ratio": alpha_count / len(text),
    }


def filter_by_quality(documents, max_special_ratio=0.3, min_alpha_ratio=0.4):
    passed = []
    removed = 0

    for doc in documents:
        scores = quality_score(doc["text"])
        if (scores["special_char_ratio"] <= max_special_ratio and
                scores["alpha_ratio"] >= min_alpha_ratio):
            passed.append(doc)
        else:
            removed += 1

    return passed, removed


quality_filtered, n_quality_removed = filter_by_quality(length_filtered)
print(f"\n--- Quality Filtering ---")
print(f"  Before: {len(length_filtered)} documents")
print(f"  Removed (low quality): {n_quality_removed}")
print(f"  After:  {len(quality_filtered)} documents")


# --- 2.4 Full cleaning pipeline ------------------------------------------

def clean_pipeline(raw_corpus):
    """Run the complete cleaning pipeline and report statistics."""
    print(f"\n  === Full Cleaning Pipeline ===")
    print(f"  Raw documents:          {len(raw_corpus)}")

    step1, n_dup = deduplicate(raw_corpus)
    print(f"  After dedup:            {len(step1)} (-{n_dup} duplicates)")

    step2, len_stats = filter_by_length(step1)
    print(f"  After length filter:    {len(step2)} (-{len_stats['too_short']} short, -{len_stats['too_long']} long)")

    step3, n_qual = filter_by_quality(step2)
    print(f"  After quality filter:   {len(step3)} (-{n_qual} low quality)")

    removed_total = len(raw_corpus) - len(step3)
    pct = removed_total / len(raw_corpus) * 100
    print(f"  Total removed:          {removed_total} ({pct:.1f}%)")
    return step3


clean_corpus = clean_pipeline(corpus)

print(f"\n  Sample quality scores:")
for doc in clean_corpus[:3]:
    scores = quality_score(doc["text"])
    print(f"    [{doc['source']:5s}] alpha={scores['alpha_ratio']:.2f}  "
          f"special={scores['special_char_ratio']:.2f}  "
          f"avg_line={scores['avg_line_length']:.0f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — DATA FORMATTING (Sequence Packing)
# ═══════════════════════════════════════════════════════════════════════════
#
# Transformers need fixed-length input sequences (e.g., 2048 tokens).
# But documents vary wildly in length. Two approaches:
#
#   1. Padding:  each doc gets its own sequence, pad short ones with <pad>
#               → Simple but wastes compute on pad tokens
#
#   2. Packing:  concatenate docs with <sep>, then slice into chunks
#               → No wasted compute, standard for pre-training
#
# Example (seq_len=8):
#   Doc A: "the cat"     Doc B: "a dog runs"     Doc C: "hi"
#
#   Padding: [the, cat, <pad>, <pad>, <pad>, <pad>, <pad>, <pad>]  ← 75% waste!
#            [a,   dog, runs,  <pad>, <pad>, <pad>, <pad>, <pad>]  ← 63% waste!
#            [hi,  <pad>, ...]
#
#   Packing: [the, cat, <sep>, a, dog, runs, <sep>, hi]            ← 0% waste!
#
# Packing is universally used for pre-training. Padding is used for
# fine-tuning where you need clean document boundaries.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: DATA FORMATTING (Sequence Packing)")
print("=" * 70)


# --- 3.1 Naive padding approach ------------------------------------------

def pad_sequences(documents, seq_len, pad_token="<pad>"):
    """
    WHY THIS MATTERS:
    Padding is the simplest approach — every document becomes one sequence.
    Short docs get padded, long docs get truncated. This wastes massive
    amounts of compute: if your average doc is 200 tokens but seq_len is
    2048, you're wasting 90% of every forward pass on pad tokens.
    """
    sequences = []
    total_tokens = 0
    total_padding = 0

    for doc in documents:
        tokens = doc["text"].split()
        if len(tokens) > seq_len:
            tokens = tokens[:seq_len]
        padding_needed = seq_len - len(tokens)
        padded = tokens + [pad_token] * padding_needed
        sequences.append(padded)
        total_tokens += len(tokens)
        total_padding += padding_needed

    return sequences, total_tokens, total_padding


SEQ_LEN = 32
padded_seqs, pad_real, pad_waste = pad_sequences(clean_corpus, SEQ_LEN)

print(f"\n--- Naive Padding (seq_len={SEQ_LEN}) ---")
print(f"  Sequences created: {len(padded_seqs)}")
print(f"  Real tokens:       {pad_real}")
print(f"  Padding tokens:    {pad_waste}")
pct_waste = pad_waste / (pad_real + pad_waste) * 100
print(f"  Padding waste:     {pct_waste:.1f}%")
print(f"\n  Example padded sequence:")
print(f"    {padded_seqs[0][:10]}...")


# --- 3.2 Sequence packing ------------------------------------------------

def pack_sequences(documents, seq_len, sep_token="<sep>"):
    """
    WHY THIS MATTERS:
    Packing concatenates all documents with a separator token, then slices
    the stream into fixed-length chunks. Zero wasted compute.

    This is how GPT-2, GPT-3, LLaMA, and almost every LLM is pre-trained.
    The model learns to handle document boundaries naturally.
    """
    token_stream = []
    for doc in documents:
        tokens = doc["text"].split()
        token_stream.extend(tokens)
        token_stream.append(sep_token)

    sequences = []
    for i in range(0, len(token_stream) - seq_len + 1, seq_len):
        chunk = token_stream[i:i + seq_len]
        sequences.append(chunk)

    leftover = len(token_stream) % seq_len
    return sequences, len(token_stream), leftover


packed_seqs, pack_total, pack_leftover = pack_sequences(clean_corpus, SEQ_LEN)

print(f"\n--- Sequence Packing (seq_len={SEQ_LEN}) ---")
print(f"  Total tokens in stream: {pack_total}")
print(f"  Sequences created:      {len(packed_seqs)}")
print(f"  Leftover (discarded):   {pack_leftover}")
pct_pack_waste = pack_leftover / pack_total * 100
print(f"  Waste:                  {pct_pack_waste:.1f}%")
print(f"\n  Example packed sequence:")
print(f"    {packed_seqs[0][:10]}...")


# --- 3.3 Compare the two approaches --------------------------------------

print(f"\n--- Padding vs Packing Comparison ---")
print(f"  {'Metric':<25s} {'Padding':>10s} {'Packing':>10s}")
print(f"  {'─'*25} {'─'*10} {'─'*10}")
print(f"  {'Sequences':<25s} {len(padded_seqs):>10d} {len(packed_seqs):>10d}")
print(f"  {'Useful tokens':<25s} {pad_real:>10d} {pack_total:>10d}")
print(f"  {'Wasted tokens':<25s} {pad_waste:>10d} {pack_leftover:>10d}")
print(f"  {'Waste %':<25s} {pct_waste:>9.1f}% {pct_pack_waste:>9.1f}%")

# Text bar chart
print(f"\n  Padding waste: {'█' * int(pct_waste / 2)}{'░' * (50 - int(pct_waste / 2))} {pct_waste:.1f}%")
print(f"  Packing waste: {'█' * int(pct_pack_waste / 2)}{'░' * (50 - int(pct_pack_waste / 2))} {pct_pack_waste:.1f}%")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — DATALOADER DESIGN
# ═══════════════════════════════════════════════════════════════════════════
#
# PyTorch provides two Dataset styles:
#
#   Map-style (Dataset):      random access via __getitem__(idx)
#                             + easy shuffling, resumable
#                             - must fit index in memory
#
#   Streaming (IterableDataset): yields samples one at a time
#                             + handles datasets too large for memory
#                             - harder to shuffle, harder to resume
#
# For small-to-medium datasets (< 100GB), map-style is simpler.
# For web-scale pre-training (terabytes), streaming is necessary.
#
# The DataLoader wraps a Dataset, adding batching, shuffling, and
# multi-process data loading (num_workers > 0).
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 4: DATALOADER DESIGN")
print("=" * 70)

torch.manual_seed(42)


# --- 4.1 Map-style Dataset -----------------------------------------------

class TextDataset(Dataset):
    """
    WHY THIS MATTERS:
    This is the simplest way to serve pre-training data. You pack all your
    text into fixed-length sequences, store them, and let the DataLoader
    shuffle and batch them. Every major training framework (PyTorch,
    Megatron-LM, LitGPT) builds on this pattern.
    """

    def __init__(self, documents, seq_len, sep_token="<sep>"):
        token_stream = []
        for doc in documents:
            tokens = doc["text"].split()
            token_stream.extend(tokens)
            token_stream.append(sep_token)

        self.vocab = {word: i for i, word in enumerate(
            sorted(set(token_stream))
        )}
        self.vocab_size = len(self.vocab)

        token_ids = [self.vocab[t] for t in token_stream]
        n_seqs = len(token_ids) // seq_len
        self.data = torch.tensor(token_ids[:n_seqs * seq_len]).view(n_seqs, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x


dataset = TextDataset(clean_corpus, seq_len=SEQ_LEN)
print(f"\n--- Map-Style TextDataset ---")
print(f"  Vocab size:      {dataset.vocab_size}")
print(f"  Num sequences:   {len(dataset)}")
print(f"  Sequence shape:  {dataset[0].shape}")
print(f"  Sample (ids):    {dataset[0][:10].tolist()}")


# --- 4.2 DataLoader with batching and shuffling ---------------------------

"""
WHY THIS MATTERS:
The DataLoader is the bridge between your data and the GPU. It handles:
  - Batching: group sequences into tensors of shape (batch_size, seq_len)
  - Shuffling: randomize order each epoch (crucial for training)
  - Prefetching: load next batch while GPU trains on current one
  - Multi-process: num_workers>0 parallelizes data loading
"""

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    drop_last=True,
)

print(f"\n--- DataLoader ---")
print(f"  Batch size:       4")
print(f"  Total batches:    {len(loader)}")
print(f"  Shuffle:          True")
print(f"  Drop last:        True (avoids partial batches)")

batch = next(iter(loader))
print(f"\n  First batch shape: {batch.shape}")
print(f"  First batch[0]:    {batch[0][:10].tolist()}...")


# --- 4.3 Streaming Dataset -----------------------------------------------

class StreamingTextDataset(IterableDataset):
    """
    WHY THIS MATTERS:
    When your dataset is terabytes (Common Crawl, FineWeb), you can't load
    it all into memory. A streaming dataset reads and processes data on the
    fly, yielding one sample at a time. This is how large-scale training
    frameworks like Megatron-LM and Mosaic StreamingDataset work.
    """

    def __init__(self, documents, seq_len, sep_token="<sep>"):
        self.documents = documents
        self.seq_len = seq_len
        self.sep_token = sep_token

        token_stream = []
        for doc in documents:
            token_stream.extend(doc["text"].split())
            token_stream.append(sep_token)
        self.vocab = {word: i for i, word in enumerate(sorted(set(token_stream)))}

    def __iter__(self):
        buffer = []
        for doc in self.documents:
            tokens = doc["text"].split()
            for t in tokens:
                buffer.append(self.vocab.get(t, 0))
                if len(buffer) == self.seq_len:
                    yield torch.tensor(buffer)
                    buffer = []
            buffer.append(self.vocab.get(self.sep_token, 0))
            if len(buffer) == self.seq_len:
                yield torch.tensor(buffer)
                buffer = []


streaming_ds = StreamingTextDataset(clean_corpus, seq_len=SEQ_LEN)
streaming_loader = DataLoader(streaming_ds, batch_size=4)

print(f"\n--- Streaming Dataset ---")
n_streamed = 0
for batch in streaming_loader:
    n_streamed += 1
print(f"  Batches yielded: {n_streamed}")
print(f"  Batch shape:     {batch.shape}")

first_batch = next(iter(DataLoader(streaming_ds, batch_size=4)))
print(f"  First batch[0]:  {first_batch[0][:10].tolist()}...")


# --- 4.4 Throughput measurement ------------------------------------------

print(f"\n--- DataLoader Throughput ---")

map_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

start = time.perf_counter()
total_tokens = 0
for batch in map_loader:
    total_tokens += batch.numel()
elapsed = time.perf_counter() - start

print(f"  Map-style:     {total_tokens:,} tokens in {elapsed:.4f}s "
      f"= {total_tokens / elapsed:,.0f} tokens/sec")

stream_loader = DataLoader(
    StreamingTextDataset(clean_corpus, seq_len=SEQ_LEN), batch_size=8
)

start = time.perf_counter()
total_tokens = 0
for batch in stream_loader:
    total_tokens += batch.numel()
elapsed = time.perf_counter() - start

print(f"  Streaming:     {total_tokens:,} tokens in {elapsed:.4f}s "
      f"= {total_tokens / elapsed:,.0f} tokens/sec")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — DATA MIXING
# ═══════════════════════════════════════════════════════════════════════════
#
# LLMs don't train on a single data source — they train on a carefully
# balanced MIX. The proportions matter enormously:
#
#   Too much web text   → model outputs generic SEO-style prose
#   Too much code       → model over-generates code syntax in chat
#   Too little Wikipedia → model lacks factual grounding
#   Too little books    → model has poor long-form coherence
#
# Typical proportions (LLaMA-2 style):
#   Web:  67%   Wikipedia: 4.5%   Books: 4.5%   Code: 4.5%
#   ArXiv: 2.5%  StackExchange: 2%  CommonCrawl-filtered: 15%
#
# The mix is a hyperparameter you tune based on downstream performance.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: DATA MIXING")
print("=" * 70)

torch.manual_seed(42)


# --- 5.1 Separate sources ------------------------------------------------

def split_by_source(documents):
    sources = {}
    for doc in documents:
        src = doc["source"]
        if src not in sources:
            sources[src] = []
        sources[src].append(doc)
    return sources


source_splits = split_by_source(clean_corpus)
print(f"\n--- Data Sources ---")
for src, docs in sorted(source_splits.items()):
    total_chars = sum(len(d["text"]) for d in docs)
    print(f"  {src:8s}: {len(docs):3d} docs, {total_chars:6,} chars")


# --- 5.2 Weighted sampling -----------------------------------------------

class MixedDataset(Dataset):
    """
    WHY THIS MATTERS:
    Data mixing controls what "knowledge" your model absorbs. If Wikipedia
    is 3% of the raw data but you sample it at 10%, the model gets 3x more
    exposure to encyclopedic facts. This is a core pre-training decision.

    WeightedRandomSampler lets you over-sample minority sources and
    under-sample majority sources to hit your target proportions.
    """

    def __init__(self, source_datasets, seq_len):
        self.datasets = []
        self.source_labels = []
        self.seq_len = seq_len

        for src_name, docs in source_datasets.items():
            for doc in docs:
                tokens = doc["text"].split()
                if len(tokens) >= seq_len:
                    self.datasets.append(tokens[:seq_len])
                    self.source_labels.append(src_name)

        self.unique_sources = sorted(set(self.source_labels))

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        return self.datasets[idx], self.source_labels[idx]


mixed_ds = MixedDataset(source_splits, seq_len=16)
print(f"\n--- MixedDataset ---")
print(f"  Total samples: {len(mixed_ds)}")

source_in_ds = {}
for i in range(len(mixed_ds)):
    _, src = mixed_ds[i]
    source_in_ds[src] = source_in_ds.get(src, 0) + 1
for src, count in sorted(source_in_ds.items()):
    pct = count / len(mixed_ds) * 100
    print(f"  {src:8s}: {count:3d} ({pct:5.1f}%)")


# --- 5.3 Create weighted sampler -----------------------------------------

TARGET_WEIGHTS = {"wiki": 0.50, "books": 0.30, "code": 0.20}

print(f"\n--- Target Mixing Proportions ---")
for src, w in TARGET_WEIGHTS.items():
    bar = "█" * int(w * 40)
    print(f"  {src:8s}: {bar} {w*100:.0f}%")

source_counts_ds = {}
for i in range(len(mixed_ds)):
    _, src = mixed_ds[i]
    source_counts_ds[src] = source_counts_ds.get(src, 0) + 1

sample_weights = []
for i in range(len(mixed_ds)):
    _, src = mixed_ds[i]
    target_p = TARGET_WEIGHTS.get(src, 0.0)
    n_src = source_counts_ds.get(src, 1)
    sample_weights.append(target_p / n_src)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(mixed_ds),
    replacement=True,
)

mixed_loader = DataLoader(mixed_ds, batch_size=8, sampler=sampler)


# --- 5.4 Verify mixing proportions ---------------------------------------

print(f"\n--- Actual Sampling Distribution (1 epoch) ---")
sample_counts = {}
for batch_tokens, batch_sources in mixed_loader:
    for src in batch_sources:
        sample_counts[src] = sample_counts.get(src, 0) + 1

total_sampled = sum(sample_counts.values())
for src in sorted(sample_counts.keys()):
    actual_pct = sample_counts[src] / total_sampled * 100
    target_pct = TARGET_WEIGHTS.get(src, 0) * 100
    bar = "█" * int(actual_pct / 100 * 40)
    print(f"  {src:8s}: {bar} {actual_pct:5.1f}% (target: {target_pct:.0f}%)")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — TOKENIZATION INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════
#
# In a real pipeline, tokenization happens BEFORE packing:
#
#   raw text → tokenizer → token IDs → pack into sequences → DataLoader
#
# We build a simple character-level tokenizer here (since we're avoiding
# external dependencies). In production you'd use a BPE tokenizer
# (Lesson 4) — the pipeline logic is identical, only the tokenizer changes.
#
# The output is what your transformer model actually consumes:
#   input_ids:  (batch_size, seq_len)     — token IDs
#   labels:     (batch_size, seq_len)     — shifted by 1 for next-token prediction
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 6: TOKENIZATION INTEGRATION")
print("=" * 70)

torch.manual_seed(42)


# --- 6.1 Simple character-level tokenizer ---------------------------------

class CharTokenizer:
    """
    WHY THIS MATTERS:
    Every LLM training pipeline starts with tokenization. The tokenizer
    converts raw text → integer token IDs that the model can process.
    We use character-level here for simplicity. Production LLMs use BPE
    (Lesson 4) with vocab sizes of 32K-128K, but the pipeline is the same:
        encode("hello") → [5, 2, 9, 9, 12]
        decode([5, 2, 9, 9, 12]) → "hello"
    """

    def __init__(self):
        chars = sorted(set(
            string.ascii_letters + string.digits +
            string.punctuation + " \n\t"
        ))
        self.special_tokens = {"<pad>": 0, "<sep>": 1, "<unk>": 2}
        self.char_to_id = {**self.special_tokens}
        for ch in chars:
            if ch not in self.char_to_id:
                self.char_to_id[ch] = len(self.char_to_id)
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)

    def encode(self, text):
        return [self.char_to_id.get(ch, self.special_tokens["<unk>"]) for ch in text]

    def decode(self, ids):
        return "".join(self.id_to_char.get(i, "?") for i in ids)


tokenizer = CharTokenizer()
print(f"\n--- Character Tokenizer ---")
print(f"  Vocab size: {tokenizer.vocab_size}")

sample_text = "Hello, LLM!"
encoded = tokenizer.encode(sample_text)
decoded = tokenizer.decode(encoded)
print(f"  Encode '{sample_text}' → {encoded}")
print(f"  Decode {encoded} → '{decoded}'")
print(f"  Round-trip match: {sample_text == decoded}")


# --- 6.2 Tokenized packing pipeline --------------------------------------

def tokenize_and_pack(documents, tokenizer, seq_len):
    """Full pipeline: documents → tokenize → pack → tensor sequences."""
    all_ids = []
    sep_id = tokenizer.special_tokens["<sep>"]

    for doc in documents:
        ids = tokenizer.encode(doc["text"])
        all_ids.extend(ids)
        all_ids.append(sep_id)

    n_seqs = len(all_ids) // seq_len
    if n_seqs == 0:
        return torch.tensor([]), 0

    tensor = torch.tensor(all_ids[:n_seqs * seq_len], dtype=torch.long)
    return tensor.view(n_seqs, seq_len), len(all_ids)


TOKEN_SEQ_LEN = 128
packed_tokens, total_token_count = tokenize_and_pack(
    clean_corpus, tokenizer, TOKEN_SEQ_LEN
)

print(f"\n--- Tokenized + Packed Sequences ---")
print(f"  Sequence length:    {TOKEN_SEQ_LEN}")
print(f"  Total token IDs:    {total_token_count:,}")
print(f"  Packed sequences:   {packed_tokens.shape[0]}")
print(f"  Tensor shape:       {packed_tokens.shape}")
print(f"  Sample (first 20):  {packed_tokens[0, :20].tolist()}")
print(f"  Decoded: '{tokenizer.decode(packed_tokens[0, :40].tolist())}'")


# --- 6.3 Training-ready Dataset (input_ids + labels) ---------------------

class PreTrainingDataset(Dataset):
    """
    WHY THIS MATTERS:
    Language model training uses next-token prediction:
      input:  [t0, t1, t2, ..., t_{n-1}]
      label:  [t1, t2, t3, ..., t_n]

    The label is just the input shifted right by one position.
    This is the EXACT format that nn.CrossEntropyLoss expects when
    training GPT, LLaMA, or any causal language model.
    """

    def __init__(self, token_sequences):
        self.data = token_sequences

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        input_ids = seq[:-1]   # all tokens except the last
        labels = seq[1:]       # all tokens except the first
        return input_ids, labels


pretrain_ds = PreTrainingDataset(packed_tokens)
pretrain_loader = DataLoader(pretrain_ds, batch_size=4, shuffle=True, drop_last=True)

print(f"\n--- Pre-Training DataLoader ---")
print(f"  Dataset size:      {len(pretrain_ds)} sequences")
print(f"  Batch size:        4")
print(f"  Batches per epoch: {len(pretrain_loader)}")

input_ids, labels = next(iter(pretrain_loader))
print(f"\n  input_ids shape:   {input_ids.shape}")
print(f"  labels shape:      {labels.shape}")
print(f"  input_ids[0][:10]: {input_ids[0][:10].tolist()}")
print(f"  labels[0][:10]:    {labels[0][:10].tolist()}")
print(f"  (labels = input shifted by 1 position)")


# --- 6.4 End-to-end: one training step -----------------------------------

"""
WHY THIS MATTERS:
This connects the ENTIRE pipeline to actual model training.
Raw text → clean → tokenize → pack → batch → model → loss → backward.
Every LLM in the world follows this exact flow.
"""

print(f"\n--- End-to-End: Data Pipeline → Training Step ---")

VOCAB_SIZE = tokenizer.vocab_size
D_MODEL = 64
N_HEADS = 4
SEQ_LEN_MODEL = TOKEN_SEQ_LEN - 1

simple_model = nn.Sequential(
    nn.Embedding(VOCAB_SIZE, D_MODEL),
    nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=256,
            batch_first=True, dropout=0.0,
        ),
        num_layers=2,
    ),
    nn.Linear(D_MODEL, VOCAB_SIZE),
)

optimizer = torch.optim.AdamW(simple_model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

n_params = sum(p.numel() for p in simple_model.parameters())
print(f"  Model params:  {n_params:,}")
print(f"  Vocab size:    {VOCAB_SIZE}")
print(f"  Seq length:    {SEQ_LEN_MODEL}")

print(f"\n  Running 5 training steps...")
simple_model.train()
step_loader = DataLoader(pretrain_ds, batch_size=8, shuffle=True, drop_last=True)
step_iter = iter(step_loader)

for step in range(5):
    try:
        input_ids, labels = next(step_iter)
    except StopIteration:
        step_iter = iter(step_loader)
        input_ids, labels = next(step_iter)

    causal_mask = nn.Transformer.generate_square_subsequent_mask(input_ids.size(1))

    logits = simple_model[0](input_ids)
    logits = simple_model[1](logits, mask=causal_mask, is_causal=True)
    logits = simple_model[2](logits)

    loss = loss_fn(logits.view(-1, VOCAB_SIZE), labels.view(-1))

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"    Step {step}: loss={loss.item():.4f}  "
          f"perplexity={torch.exp(loss).item():.1f}")

print(f"\n  Pipeline complete: raw text → clean → tokenize → pack → train!")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXERCISES")
print("=" * 70)
print("""
Exercise 1: Near-Duplicate Detection
--------------------------------------
Hash-based dedup only catches EXACT duplicates. Implement near-duplicate
detection:
  a) For each document, compute a set of 3-word "shingles" (sliding window)
     Example: "the cat sat" → {"the cat sat", "cat sat on", "sat on the"}
  b) Define similarity as: |shingles_A ∩ shingles_B| / |shingles_A ∪ shingles_B|
     (Jaccard similarity)
  c) Two documents are near-duplicates if similarity > 0.8
  d) Run on the corpus — how many more duplicates do you find vs exact dedup?
  e) What is the time complexity of comparing all pairs? Why is MinHash used
     in practice?

Exercise 2: Sequence Packing with Attention Masks
---------------------------------------------------
Our packing implementation doesn't prevent attention across document
boundaries. Fix this:
  a) When packing, track where each document starts and ends
  b) Create an attention mask of shape (seq_len, seq_len) where:
     - mask[i][j] = 1 if tokens i and j are in the SAME document
     - mask[i][j] = 0 if they are in DIFFERENT documents
  c) This prevents "cross-contamination" between unrelated documents
  d) Test: pack 3 short documents into one sequence and verify the mask
  e) Why do most pre-training runs skip this and just let attention cross
     boundaries? (Hint: what happens to the loss on boundary tokens?)

Exercise 3: Custom Collate Function
--------------------------------------
Build a collate function that creates training batches with:
  a) input_ids: (batch_size, seq_len) — token IDs
  b) labels: (batch_size, seq_len) — shifted by 1
  c) attention_mask: (batch_size, seq_len) — 1 for real tokens, 0 for padding
  d) Use it with DataLoader(collate_fn=your_function)
  e) Verify: the attention_mask should have 0s only at pad positions
  f) Compute the "effective batch size" (total non-pad tokens per batch)

Exercise 4: Data Pipeline Throughput
--------------------------------------
Measure your pipeline's throughput at each stage:
  a) Time the cleaning pipeline (docs/second)
  b) Time tokenization (chars/second)
  c) Time packing (tokens/second)
  d) Time DataLoader iteration (batches/second, tokens/second)
  e) Which stage is the bottleneck?
  f) Try DataLoader with num_workers=0, 1, 2, 4 — how does throughput
     change? (Note: on some systems multiprocessing has overhead for
     small datasets)

Exercise 5: Dynamic Data Mixing
---------------------------------
Implement a curriculum learning strategy where mixing proportions
change during training:
  a) Start with 80% wiki, 10% books, 10% code (epoch 0)
  b) Shift to 50% wiki, 30% books, 20% code (epoch 5)
  c) Linearly interpolate proportions between epochs
  d) Implement a MixingScheduler class that returns weights for each epoch
  e) Log the actual proportions each epoch and verify they match targets
  f) Why might you want to increase code proportion later in training?
     (Hint: think about what capabilities emerge at different stages)
""")
