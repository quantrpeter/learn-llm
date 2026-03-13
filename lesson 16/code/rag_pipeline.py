"""
============================================================================
 LESSON 16: Retrieval-Augmented Generation (RAG)
============================================================================

 Goal: Extend your LLM with external knowledge retrieval so it can answer
       questions about information it was never trained on — without
       hallucinating.

 Topics covered:
   1. Why RAG          — knowledge cutoffs, hallucination, grounding
   2. Embedding Models — sentence embeddings via mean pooling, cosine sim
   3. Vector Database  — in-memory vector store from scratch (add / search)
   4. Chunking         — fixed-size, sentence-based, recursive splitting
   5. Retrieval Pipeline — query → embed → search → rerank → augment
   6. Context Window   — token counting, truncation, fitting docs

 Every section has:
   - A short explanation (read the docstrings)
   - Working code you can run and modify
   - Exercises at the bottom

 Prerequisites: Lessons 1-15 (especially tokenization, attention, serving)
============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import re
import textwrap

torch.manual_seed(42)


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — WHY RAG
# ═══════════════════════════════════════════════════════════════════════════
#
# Large language models are trained on a static snapshot of the internet.
# Once training ends, their knowledge is frozen. They can't know about:
#   - Events after the training cutoff date
#   - Your private company documents
#   - Real-time data (stock prices, weather, live sports)
#
# Worse, when asked about things they don't know, LLMs confidently
# fabricate plausible-sounding answers — this is called "hallucination."
#
# RAG solves both problems: retrieve relevant documents first, then feed
# them to the LLM as context. The model generates answers grounded in
# real data instead of relying solely on memorized knowledge.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 1: WHY RAG — The Hallucination Problem")
print("=" * 70)

# --- 1.1 Simulating hallucination -----------------------------------------

knowledge_base = {
    "Python":  "Python was created by Guido van Rossum and first released in 1991.",
    "Rust":    "Rust was first released in 2015 by Mozilla Research.",
    "PyTorch": "PyTorch was released by Meta AI (Facebook) in 2016.",
}

queries_with_answers = [
    ("When was Python created?", "Python"),
    ("Who made Rust?", "Rust"),
    ("Tell me about QuantumLang 3.0", None),  # not in knowledge base
]

"""
WHY THIS MATTERS:
Without RAG, an LLM would try to answer the QuantumLang question from
"memory" — and hallucinate a confident but completely fabricated answer.
With RAG, it first checks the knowledge base. If nothing relevant is
found, it can honestly say "I don't have information about that."
"""

print("\n--- Without RAG (pure LLM, simulated) ---")
for query, key in queries_with_answers:
    if key and key in knowledge_base:
        print(f"  Q: {query}")
        print(f"  A: {knowledge_base[key]}")
    else:
        print(f"  Q: {query}")
        print(f"  A: [HALLUCINATION] QuantumLang 3.0 was released in 2024 "
              f"by DeepMind...")  # fake!

print("\n--- With RAG (retrieve first, then answer) ---")
for query, key in queries_with_answers:
    if key and key in knowledge_base:
        print(f"  Q: {query}")
        print(f"  Retrieved: \"{knowledge_base[key]}\"")
        print(f"  A: (grounded in retrieved document)")
    else:
        print(f"  Q: {query}")
        print(f"  Retrieved: (nothing relevant found)")
        print(f"  A: I don't have information about QuantumLang 3.0.")


# --- 1.2 The three problems RAG solves ------------------------------------

"""
WHY THIS MATTERS:
RAG is the most practical way to extend LLMs without retraining:

 Problem 1: Knowledge Cutoff
   LLMs are frozen at training time. RAG injects fresh information.

 Problem 2: Hallucination
   By grounding answers in retrieved documents, the model fabricates less.

 Problem 3: Domain Specificity
   Your company's internal docs aren't in any training set.
   RAG lets you build a chatbot over YOUR data.
"""

problems = [
    ("Knowledge Cutoff",   "Training data ends at a fixed date"),
    ("Hallucination",      "LLM invents facts when uncertain"),
    ("Domain Specificity", "Private/internal data not in training set"),
]

print("\n--- Three problems RAG solves ---")
for i, (problem, desc) in enumerate(problems, 1):
    print(f"  {i}. {problem}: {desc}")

print("\n  RAG approach: Retrieve → Read → Respond")
print("  Instead of:  Remember → Guess → Hope for the best")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — EMBEDDING MODELS
# ═══════════════════════════════════════════════════════════════════════════
#
# To find relevant documents for a query, we need to represent both the
# query and documents as vectors in the same space. Similar meanings
# should produce vectors that are close together (high cosine similarity).
#
# Real embedding models (e.g., sentence-transformers) use a pretrained
# transformer + mean pooling over token embeddings. Here we build a
# simplified version from scratch to show the mechanics.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 2: EMBEDDING MODELS — Text to Vectors")
print("=" * 70)

torch.manual_seed(42)


# --- 2.1 Building a vocabulary -------------------------------------------

class SimpleTokenizer:
    """Word-level tokenizer that maps words to integer IDs."""

    def __init__(self):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = {0: "<pad>", 1: "<unk>"}
        self.next_id = 2

    def fit(self, texts: list[str]):
        for text in texts:
            for word in self._tokenize(text):
                if word not in self.word2idx:
                    self.word2idx[word] = self.next_id
                    self.idx2word[self.next_id] = word
                    self.next_id += 1
        return self

    def encode(self, text: str) -> list[int]:
        return [self.word2idx.get(w, 1) for w in self._tokenize(text)]

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b\w+\b', text.lower())

    @property
    def vocab_size(self) -> int:
        return self.next_id


# --- 2.2 Sentence embedding with mean pooling -----------------------------

"""
WHY THIS MATTERS:
Real embedding models (like all-MiniLM-L6-v2) work exactly this way:
  1. Tokenize the input
  2. Pass tokens through a transformer → get per-token embeddings
  3. Mean-pool the token embeddings → single sentence vector
  4. Normalize to unit length

We simulate the transformer with an embedding table + a linear projection.
The key insight: mean pooling collapses variable-length text into a
fixed-size vector that captures the "average meaning" of all tokens.
"""


class SimpleEmbedder(nn.Module):
    """Minimal sentence embedding model: embed → project → mean pool → normalize."""

    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj = nn.Linear(embed_dim, hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len) — padded integer sequences
        returns:   (batch, hidden_dim) — L2-normalized sentence vectors
        """
        mask = (token_ids != 0).unsqueeze(-1).float()    # (batch, seq, 1)
        x = self.embed(token_ids)                         # (batch, seq, embed_dim)
        x = F.gelu(self.proj(x))                          # (batch, seq, hidden_dim)
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # mean pool
        return F.normalize(x, p=2, dim=-1)                # L2 normalize


# Build tokenizer and embedder on sample corpus
corpus = [
    "Python is a popular programming language for machine learning.",
    "PyTorch is a deep learning framework built on Python.",
    "Retrieval augmented generation combines search with language models.",
    "Vector databases store embeddings for fast similarity search.",
    "Transformers use self-attention to process sequences in parallel.",
    "The cat sat on the mat and purred quietly.",
    "Cooking pasta requires boiling water and a pinch of salt.",
]

tokenizer = SimpleTokenizer().fit(corpus)
embedder = SimpleEmbedder(tokenizer.vocab_size, embed_dim=64, hidden_dim=128)
embedder.eval()

print(f"\nVocabulary size: {tokenizer.vocab_size}")
print(f"Embedding output dimension: {embedder.output_dim}")


def encode_texts(texts: list[str], tok: SimpleTokenizer, model: SimpleEmbedder) -> torch.Tensor:
    """Encode a list of texts into normalized embedding vectors."""
    encoded = [tok.encode(t) for t in texts]
    max_len = max(len(e) for e in encoded)
    padded = [e + [0] * (max_len - len(e)) for e in encoded]
    with torch.no_grad():
        return model(torch.tensor(padded))


corpus_embeddings = encode_texts(corpus, tokenizer, embedder)
print(f"Corpus embeddings shape: {corpus_embeddings.shape}")  # (7, 128)


# --- 2.3 Cosine similarity search ----------------------------------------

"""
WHY THIS MATTERS:
Cosine similarity is the standard metric for comparing embeddings.
  cos(a, b) = (a · b) / (||a|| × ||b||)

Since we L2-normalize our vectors, cosine similarity simplifies to
just the dot product: cos(a, b) = a · b. This makes search fast —
it's just a matrix multiply.
"""


def cosine_similarity_search(query_vec: torch.Tensor, doc_vecs: torch.Tensor, top_k: int = 3):
    """Return top-k most similar documents by cosine similarity."""
    scores = query_vec @ doc_vecs.T   # (1, hidden) @ (n, hidden).T → (1, n)
    scores = scores.squeeze(0)
    top_scores, top_indices = torch.topk(scores, k=min(top_k, len(scores)))
    return top_scores.tolist(), top_indices.tolist()


query = "How does retrieval augmented generation work?"
query_vec = encode_texts([query], tokenizer, embedder)

scores, indices = cosine_similarity_search(query_vec, corpus_embeddings, top_k=3)

print(f"\nQuery: \"{query}\"")
print(f"\nTop-3 most similar documents:")
for rank, (score, idx) in enumerate(zip(scores, indices), 1):
    print(f"  {rank}. [score={score:.4f}] \"{corpus[idx]}\"")

# Show the full similarity matrix
print(f"\nFull similarity scores against corpus:")
all_scores = (query_vec @ corpus_embeddings.T).squeeze(0)
for i, (doc, score) in enumerate(zip(corpus, all_scores.tolist())):
    bar = "█" * int(max(0, score) * 30)
    print(f"  [{score:+.4f}] {bar:30s} {doc[:50]}...")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — VECTOR DATABASE (from scratch)
# ═══════════════════════════════════════════════════════════════════════════
#
# A vector database stores document embeddings and supports fast nearest-
# neighbor search. Production systems (FAISS, ChromaDB, Qdrant) use
# advanced indexing (HNSW, IVF) for billion-scale search.
#
# Here we build a naive but complete vector store to understand the API
# and data flow. It stores vectors in a flat tensor and does brute-force
# cosine similarity — exact search, O(n) per query.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 3: VECTOR DATABASE — In-Memory Store")
print("=" * 70)

torch.manual_seed(42)


class NaiveVectorStore:
    """
    Minimal vector database: add documents, search by similarity.

    In production you'd use FAISS (Facebook AI Similarity Search),
    ChromaDB, Qdrant, or pgvector. They all share this same API:
      - add(vectors, metadata)
      - search(query_vector, top_k)
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = torch.zeros(0, dim)
        self.documents: list[str] = []
        self.metadata: list[dict] = []

    def add(self, texts: list[str], vectors: torch.Tensor, metadata: list[dict] | None = None):
        """Add document vectors with their text and optional metadata."""
        assert vectors.shape[1] == self.dim, f"Expected dim {self.dim}, got {vectors.shape[1]}"
        self.vectors = torch.cat([self.vectors, vectors], dim=0)
        self.documents.extend(texts)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(texts))

    def search(self, query_vector: torch.Tensor, top_k: int = 3) -> list[dict]:
        """Find top-k nearest neighbors by cosine similarity."""
        if len(self.documents) == 0:
            return []
        scores = (query_vector @ self.vectors.T).squeeze(0)
        k = min(top_k, len(self.documents))
        top_scores, top_indices = torch.topk(scores, k=k)

        results = []
        for score, idx in zip(top_scores.tolist(), top_indices.tolist()):
            results.append({
                "text": self.documents[idx],
                "score": score,
                "metadata": self.metadata[idx],
                "index": idx,
            })
        return results

    def __len__(self):
        return len(self.documents)

    def __repr__(self):
        return f"NaiveVectorStore(docs={len(self)}, dim={self.dim})"


"""
WHY THIS MATTERS:
Every RAG system needs a vector store. The interface is always:
  1. Ingest: chunk documents → embed → store vectors + text
  2. Query:  embed query → find nearest vectors → return text

Our brute-force O(n) search is fine for <100K documents.
FAISS uses HNSW (Hierarchical Navigable Small World) graphs for
approximate nearest-neighbor search in O(log n) — same idea as
binary search trees but for high-dimensional vectors.
"""

# Build vector store from our corpus
store = NaiveVectorStore(dim=embedder.output_dim)
store.add(
    texts=corpus,
    vectors=corpus_embeddings,
    metadata=[{"source": f"doc_{i}"} for i in range(len(corpus))],
)

print(f"\n{store}")

# Search the store
query = "What framework should I use for deep learning?"
query_vec = encode_texts([query], tokenizer, embedder)
results = store.search(query_vec, top_k=3)

print(f"\nQuery: \"{query}\"")
print(f"Results:")
for r in results:
    print(f"  [{r['score']:.4f}] {r['text']}")
    print(f"          metadata: {r['metadata']}")


# --- 3.2 Multiple queries -------------------------------------------------

queries = [
    "How do vector databases work?",
    "What is a good recipe for dinner?",
    "How does attention work in neural networks?",
]

print(f"\n--- Batch querying ---")
for q in queries:
    q_vec = encode_texts([q], tokenizer, embedder)
    results = store.search(q_vec, top_k=1)
    best = results[0]
    print(f"  Q: {q}")
    print(f"  → [{best['score']:.4f}] {best['text']}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — CHUNKING STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════
#
# Real documents are too long to embed as a single vector — embeddings
# work best on paragraph-sized chunks. Chunking strategy directly affects
# retrieval quality:
#
#   Too large  → chunks contain mixed topics, diluting relevance
#   Too small  → chunks lose context, answers become fragmented
#   Just right → each chunk is a coherent unit of information
#
# We implement three strategies and compare them on the same document.
# ═══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("PART 4: CHUNKING STRATEGIES — Splitting Documents")
print("=" * 70)


SAMPLE_DOCUMENT = textwrap.dedent("""\
    Retrieval-Augmented Generation (RAG) is a technique that enhances large language \
    models by providing them with external knowledge. Instead of relying solely on \
    parameters learned during training, RAG systems retrieve relevant documents from \
    a knowledge base and include them in the prompt.

    The key components of a RAG system are: an embedding model that converts text to \
    vectors, a vector database that stores and indexes these vectors, and a retrieval \
    pipeline that finds the most relevant documents for a given query.

    Chunking is a critical preprocessing step. Documents must be split into smaller \
    pieces before embedding. The chunk size affects both retrieval precision and the \
    quality of generated answers. Typical chunk sizes range from 256 to 1024 tokens.

    There are several chunking strategies. Fixed-size chunking splits text at regular \
    intervals. Sentence-based chunking respects sentence boundaries. Recursive chunking \
    tries to split on paragraph boundaries first, then sentences, then words.

    Reranking is an optional but powerful step. After retrieving candidate documents, \
    a cross-encoder model scores each (query, document) pair more carefully. This \
    produces better results than embedding similarity alone, because cross-encoders \
    can attend to both query and document tokens simultaneously.\
""")

print(f"\nSample document length: {len(SAMPLE_DOCUMENT)} characters")


# --- 4.1 Fixed-size chunking ----------------------------------------------

def chunk_fixed_size(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """Split text into fixed-size character chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


fixed_chunks = chunk_fixed_size(SAMPLE_DOCUMENT, chunk_size=200, overlap=50)

print(f"\n--- Fixed-size chunking (200 chars, 50 overlap) ---")
print(f"Number of chunks: {len(fixed_chunks)}")
for i, chunk in enumerate(fixed_chunks):
    preview = chunk[:60].replace("\n", " ")
    print(f"  Chunk {i}: ({len(chunk):3d} chars) \"{preview}...\"")


# --- 4.2 Sentence-based chunking ------------------------------------------

"""
WHY THIS MATTERS:
Fixed-size chunking can split mid-sentence, breaking meaning.
Sentence-based chunking keeps each sentence intact, then groups
them until the target size is reached. This preserves coherence.
"""


def chunk_by_sentences(text: str, max_chars: int = 300) -> list[str]:
    """Group sentences into chunks up to max_chars."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    current = []
    current_len = 0

    for sent in sentences:
        if current_len + len(sent) > max_chars and current:
            chunks.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sent)
        current_len += len(sent) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


sentence_chunks = chunk_by_sentences(SAMPLE_DOCUMENT, max_chars=300)

print(f"\n--- Sentence-based chunking (max 300 chars) ---")
print(f"Number of chunks: {len(sentence_chunks)}")
for i, chunk in enumerate(sentence_chunks):
    preview = chunk[:60].replace("\n", " ")
    print(f"  Chunk {i}: ({len(chunk):3d} chars) \"{preview}...\"")


# --- 4.3 Recursive chunking -----------------------------------------------

def chunk_recursive(text: str, max_chars: int = 300, separators: list[str] | None = None) -> list[str]:
    """
    Recursively split text using a hierarchy of separators:
      1. Try paragraph boundaries first (\\n\\n)
      2. Then sentence boundaries (. ! ?)
      3. Then word boundaries (spaces)
      4. Finally, hard character split

    This is the strategy used by LangChain's RecursiveCharacterTextSplitter.
    """
    if separators is None:
        separators = ["\n\n", ". ", "! ", "? ", " ", ""]

    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []

    for sep in separators:
        if sep == "":
            return [text[i:i + max_chars].strip()
                    for i in range(0, len(text), max_chars)
                    if text[i:i + max_chars].strip()]

        parts = text.split(sep) if sep else [text]
        if len(parts) <= 1:
            continue

        chunks = []
        current = ""
        for part in parts:
            candidate = current + sep + part if current else part
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current.strip())
                if len(part) > max_chars:
                    remaining_seps = separators[separators.index(sep) + 1:]
                    chunks.extend(chunk_recursive(part, max_chars, remaining_seps))
                    current = ""
                else:
                    current = part
        if current:
            chunks.append(current.strip())

        if chunks:
            return [c for c in chunks if c]

    return [text.strip()] if text.strip() else []


recursive_chunks = chunk_recursive(SAMPLE_DOCUMENT, max_chars=300)

print(f"\n--- Recursive chunking (max 300 chars) ---")
print(f"Number of chunks: {len(recursive_chunks)}")
for i, chunk in enumerate(recursive_chunks):
    preview = chunk[:60].replace("\n", " ")
    print(f"  Chunk {i}: ({len(chunk):3d} chars) \"{preview}...\"")


# --- 4.4 Comparison -------------------------------------------------------

print(f"\n--- Chunking comparison ---")
print(f"{'Strategy':<20s} {'Chunks':>6s}  {'Avg len':>8s}  {'Min len':>8s}  {'Max len':>8s}")
print("-" * 58)
for name, chunks in [("Fixed-size", fixed_chunks),
                      ("Sentence-based", sentence_chunks),
                      ("Recursive", recursive_chunks)]:
    lens = [len(c) for c in chunks]
    avg = sum(lens) / len(lens) if lens else 0
    print(f"{name:<20s} {len(chunks):>6d}  {avg:>8.1f}  {min(lens):>8d}  {max(lens):>8d}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — RETRIEVAL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
#
# A complete RAG pipeline chains everything together:
#   Query → Embed → Search → Rerank → Augment Prompt → Generate
#
# We build each step, including a simple reranker that uses cross-
# attention-style scoring between query and document tokens.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 5: RETRIEVAL PIPELINE — End to End")
print("=" * 70)

torch.manual_seed(42)


# --- 5.1 Document ingestion -----------------------------------------------

"""
WHY THIS MATTERS:
The ingestion pipeline runs once (or periodically) to index your
knowledge base. The quality of your chunks and embeddings determines
the ceiling for your retrieval quality — garbage in, garbage out.
"""

knowledge_docs = [
    "PyTorch is an open-source machine learning framework developed by Meta AI. "
    "It provides tensor computation with GPU acceleration and automatic differentiation.",

    "Retrieval-Augmented Generation (RAG) enhances LLMs by retrieving relevant "
    "documents from a knowledge base before generating answers.",

    "FAISS (Facebook AI Similarity Search) is a library for efficient similarity "
    "search and clustering of dense vectors. It supports GPU acceleration.",

    "Transformers process input sequences using self-attention, allowing each token "
    "to attend to all other tokens. This enables capturing long-range dependencies.",

    "Fine-tuning adapts a pretrained model to a specific task using a smaller "
    "dataset. LoRA is a parameter-efficient fine-tuning technique.",

    "Vector databases like ChromaDB and Qdrant store high-dimensional embeddings "
    "and support fast approximate nearest-neighbor search using HNSW indexing.",

    "Tokenization converts raw text into integer token IDs. Byte Pair Encoding (BPE) "
    "is the most common subword tokenization algorithm used in modern LLMs.",

    "The attention mechanism computes scores between query and key vectors, then uses "
    "those scores to weight value vectors. Multi-head attention runs this in parallel.",

    "Quantization reduces model size by using lower-precision numbers. INT8 and INT4 "
    "quantization can shrink models by 2-4x with minimal quality loss.",

    "Prompt engineering involves designing input prompts that elicit the best responses "
    "from language models. Including examples (few-shot) often improves quality.",
]

# Re-fit tokenizer on knowledge docs and re-create embedder
all_texts = corpus + knowledge_docs
tokenizer = SimpleTokenizer().fit(all_texts)
embedder = SimpleEmbedder(tokenizer.vocab_size, embed_dim=64, hidden_dim=128)
embedder.eval()

# Build the retrieval index
rag_store = NaiveVectorStore(dim=embedder.output_dim)
doc_vecs = encode_texts(knowledge_docs, tokenizer, embedder)
rag_store.add(
    texts=knowledge_docs,
    vectors=doc_vecs,
    metadata=[{"doc_id": i, "source": "knowledge_base"} for i in range(len(knowledge_docs))],
)

print(f"\nIngested {len(rag_store)} documents into vector store")


# --- 5.2 Retrieval --------------------------------------------------------

def retrieve(query: str, store: NaiveVectorStore, tok: SimpleTokenizer,
             model: SimpleEmbedder, top_k: int = 5) -> list[dict]:
    """Embed query and retrieve top-k documents."""
    q_vec = encode_texts([query], tok, model)
    return store.search(q_vec, top_k=top_k)


query = "How does RAG work with vector databases?"
retrieved = retrieve(query, rag_store, tokenizer, embedder, top_k=5)

print(f"\nQuery: \"{query}\"")
print(f"\nRetrieved {len(retrieved)} documents:")
for i, r in enumerate(retrieved):
    print(f"  {i+1}. [score={r['score']:.4f}] {r['text'][:80]}...")


# --- 5.3 Reranking -------------------------------------------------------

"""
WHY THIS MATTERS:
Bi-encoder retrieval (embed query + embed doc separately) is fast but
coarse. A cross-encoder looks at query AND document TOGETHER, using
full attention between them. This is 10-100x slower but much more
accurate — so we use it only on the top-k candidates, not the full corpus.

We implement a simplified cross-encoder that computes token-level
attention between query and document tokens.
"""


class SimpleCrossEncoder(nn.Module):
    """Minimal cross-encoder: concatenate query+doc tokens, project, score."""

    def __init__(self, vocab_size: int, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attn_proj = nn.Linear(embed_dim, embed_dim)
        self.score_head = nn.Linear(embed_dim, 1)

    def forward(self, query_ids: torch.Tensor, doc_ids: torch.Tensor) -> torch.Tensor:
        q_emb = self.embed(query_ids)              # (1, q_len, dim)
        d_emb = self.embed(doc_ids)                # (n, d_len, dim)

        q_proj = self.attn_proj(q_emb)             # (1, q_len, dim)
        attn = torch.bmm(
            q_proj.expand(d_emb.size(0), -1, -1),
            d_emb.transpose(1, 2),
        )                                           # (n, q_len, d_len)
        attn_weights = F.softmax(attn, dim=-1)
        attended = torch.bmm(attn_weights, d_emb)  # (n, q_len, dim)
        pooled = attended.mean(dim=1)               # (n, dim)
        return self.score_head(pooled).squeeze(-1)  # (n,)


cross_encoder = SimpleCrossEncoder(tokenizer.vocab_size, embed_dim=64)
cross_encoder.eval()


def rerank(query: str, candidates: list[dict], tok: SimpleTokenizer,
           model: SimpleCrossEncoder, top_k: int = 3) -> list[dict]:
    """Rerank retrieved candidates using a cross-encoder."""
    q_ids = torch.tensor([tok.encode(query)])
    doc_texts = [c["text"] for c in candidates]
    encoded_docs = [tok.encode(t) for t in doc_texts]
    max_len = max(len(e) for e in encoded_docs)
    padded = [e + [0] * (max_len - len(e)) for e in encoded_docs]
    d_ids = torch.tensor(padded)

    with torch.no_grad():
        scores = model(q_ids, d_ids)

    ranked_indices = torch.argsort(scores, descending=True)[:top_k]
    reranked = []
    for idx in ranked_indices.tolist():
        result = dict(candidates[idx])
        result["rerank_score"] = scores[idx].item()
        reranked.append(result)
    return reranked


reranked = rerank(query, retrieved, tokenizer, cross_encoder, top_k=3)

print(f"\nAfter reranking (top 3):")
for i, r in enumerate(reranked):
    print(f"  {i+1}. [bi-enc={r['score']:.4f}, cross-enc={r['rerank_score']:.4f}]")
    print(f"     {r['text'][:80]}...")


# --- 5.4 Prompt augmentation ----------------------------------------------

"""
WHY THIS MATTERS:
This is where RAG meets generation. We construct a prompt that includes:
  1. System instruction (answer based on provided context)
  2. Retrieved context (the documents)
  3. The user's question

The LLM then generates an answer grounded in the retrieved documents.
This is exactly how ChatGPT with "Browse" or Perplexity AI works.
"""


def build_rag_prompt(query: str, context_docs: list[dict],
                     max_context_chars: int = 2000) -> str:
    """Construct a RAG-augmented prompt from retrieved documents."""
    system = ("Answer the question based ONLY on the provided context. "
              "If the context doesn't contain enough information, say so.\n\n")

    context_parts = []
    total_chars = 0
    for i, doc in enumerate(context_docs):
        text = doc["text"]
        if total_chars + len(text) > max_context_chars:
            break
        context_parts.append(f"[Document {i+1}] {text}")
        total_chars += len(text)

    context = "Context:\n" + "\n\n".join(context_parts)
    question = f"\n\nQuestion: {query}\n\nAnswer:"

    return system + context + question


rag_prompt = build_rag_prompt(query, reranked)
print(f"\n--- RAG-augmented prompt ---")
print(rag_prompt)


# --- 5.5 Full pipeline function -------------------------------------------

def rag_pipeline(query: str, store: NaiveVectorStore, tok: SimpleTokenizer,
                 bi_encoder: SimpleEmbedder, cross_enc: SimpleCrossEncoder,
                 retrieve_k: int = 5, rerank_k: int = 3) -> str:
    """Complete RAG pipeline: retrieve → rerank → build prompt."""
    candidates = retrieve(query, store, tok, bi_encoder, top_k=retrieve_k)
    reranked = rerank(query, candidates, tok, cross_enc, top_k=rerank_k)
    return build_rag_prompt(query, reranked)


print(f"\n--- Full pipeline demo ---")
test_queries = [
    "What is FAISS and why is it useful?",
    "How does tokenization work in LLMs?",
    "What is LoRA fine-tuning?",
]

for q in test_queries:
    prompt = rag_pipeline(q, rag_store, tokenizer, embedder, cross_encoder)
    lines = prompt.split("\n")
    doc_lines = [l for l in lines if l.startswith("[Document")]
    print(f"\n  Q: {q}")
    print(f"  Retrieved docs: {len(doc_lines)}")
    for dl in doc_lines:
        print(f"    {dl[:80]}...")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — CONTEXT WINDOW MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════
#
# LLMs have a fixed context window (e.g., 4096 or 128K tokens). You need
# to fit: system prompt + retrieved documents + user query + room for the
# generated answer. If you stuff too many documents in, the model either
# truncates or degrades in quality ("lost in the middle" problem).
#
# Good context management is the difference between a RAG system that
# works and one that doesn't.
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PART 6: CONTEXT WINDOW MANAGEMENT")
print("=" * 70)


# --- 6.1 Token counting ---------------------------------------------------

"""
WHY THIS MATTERS:
You must count tokens, not characters. "machine learning" is 2 words
but might be 3 tokens (e.g., "machine", " learn", "ing" in BPE).
Exceeding the context window causes silent truncation or errors.

Real tokenizers (tiktoken, sentencepiece) give exact counts.
Here we approximate with a words-based estimate: ~1.3 tokens per word
is a reasonable rule of thumb for English BPE tokenizers.
"""


def estimate_tokens(text: str, tokens_per_word: float = 1.3) -> int:
    """Estimate token count from text (approximate BPE heuristic)."""
    words = len(text.split())
    return int(words * tokens_per_word)


def exact_tokens(text: str, tok: SimpleTokenizer) -> int:
    """Count tokens using our word-level tokenizer (exact for our model)."""
    return len(tok.encode(text))


sample_texts = [
    "Hello world",
    "Retrieval-augmented generation is a powerful technique.",
    "PyTorch provides tensor computation with GPU acceleration and automatic differentiation for deep learning.",
]

print(f"\n--- Token counting ---")
print(f"{'Text':<70s} {'Words':>5s} {'Est.':>5s} {'Exact':>5s}")
print("-" * 90)
for t in sample_texts:
    words = len(t.split())
    est = estimate_tokens(t)
    exact = exact_tokens(t, tokenizer)
    print(f"{t:<70s} {words:>5d} {est:>5d} {exact:>5d}")


# --- 6.2 Context budget allocation ----------------------------------------

"""
WHY THIS MATTERS:
A typical context budget for a 4096-token model:
  System prompt:     ~100 tokens
  Retrieved docs:    ~2500 tokens (the main allocation)
  User query:        ~100 tokens
  Generation budget: ~1300 tokens (room for the answer)
  Safety margin:     ~96 tokens

If you retrieve too many documents, you eat into the generation budget
and the model can't produce a full answer.
"""


class ContextBudget:
    """Manages token allocation across prompt components."""

    def __init__(self, max_tokens: int = 4096, generation_budget: int = 1024,
                 safety_margin: int = 64):
        self.max_tokens = max_tokens
        self.generation_budget = generation_budget
        self.safety_margin = safety_margin

    @property
    def available_for_context(self) -> int:
        return self.max_tokens - self.generation_budget - self.safety_margin

    def allocate(self, system_prompt: str, query: str,
                 documents: list[str]) -> tuple[list[str], dict]:
        """Select documents that fit within the context budget."""
        system_tokens = estimate_tokens(system_prompt)
        query_tokens = estimate_tokens(query)
        overhead = system_tokens + query_tokens

        remaining = self.available_for_context - overhead
        selected = []
        total_doc_tokens = 0

        for doc in documents:
            doc_tokens = estimate_tokens(doc)
            if total_doc_tokens + doc_tokens > remaining:
                break
            selected.append(doc)
            total_doc_tokens += doc_tokens

        stats = {
            "max_tokens": self.max_tokens,
            "system_tokens": system_tokens,
            "query_tokens": query_tokens,
            "doc_tokens": total_doc_tokens,
            "docs_selected": len(selected),
            "docs_total": len(documents),
            "generation_budget": self.generation_budget,
            "tokens_used": overhead + total_doc_tokens,
            "utilization": (overhead + total_doc_tokens) / self.available_for_context * 100,
        }
        return selected, stats


budget = ContextBudget(max_tokens=4096, generation_budget=1024)

system = "Answer based on the provided context only."
query = "How does RAG work?"
docs = [d["text"] for d in retrieved]

selected_docs, stats = budget.allocate(system, query, docs)

print(f"\n--- Context budget allocation (4096 token window) ---")
print(f"  Available for context: {budget.available_for_context} tokens")
for key, val in stats.items():
    if key == "utilization":
        print(f"  {key}: {val:.1f}%")
    else:
        print(f"  {key}: {val}")


# --- 6.3 Truncation strategies --------------------------------------------

def truncate_end(text: str, max_tokens: int, tok: SimpleTokenizer) -> str:
    """Keep the beginning of the text, truncate the end."""
    tokens = tok.encode(text)
    if len(tokens) <= max_tokens:
        return text
    words = text.split()
    result = []
    count = 0
    for word in words:
        word_tokens = len(tok.encode(word))
        if count + word_tokens > max_tokens:
            break
        result.append(word)
        count += word_tokens
    return " ".join(result) + "..."


def truncate_middle(text: str, max_tokens: int, tok: SimpleTokenizer) -> str:
    """Keep beginning and end, cut the middle (preserves intro + conclusion)."""
    tokens = tok.encode(text)
    if len(tokens) <= max_tokens:
        return text
    words = text.split()
    keep_each_side = max_tokens // 2
    head_words = []
    head_count = 0
    for w in words:
        wt = len(tok.encode(w))
        if head_count + wt > keep_each_side:
            break
        head_words.append(w)
        head_count += wt
    tail_words = []
    tail_count = 0
    for w in reversed(words):
        wt = len(tok.encode(w))
        if tail_count + wt > keep_each_side:
            break
        tail_words.insert(0, w)
        tail_count += wt
    return " ".join(head_words) + " [...] " + " ".join(tail_words)


long_text = " ".join(knowledge_docs[:3])

print(f"\n--- Truncation strategies ---")
print(f"Original: {exact_tokens(long_text, tokenizer)} tokens")
print(f"  Full: \"{long_text[:80]}...\"")

truncated_end = truncate_end(long_text, 20, tokenizer)
print(f"\n  Truncate-end (20 tokens): \"{truncated_end}\"")

truncated_mid = truncate_middle(long_text, 20, tokenizer)
print(f"  Truncate-middle (20 tokens): \"{truncated_mid}\"")


# --- 6.4 The "lost in the middle" problem ----------------------------------

"""
WHY THIS MATTERS:
Research shows LLMs pay most attention to documents at the BEGINNING
and END of the context, and tend to ignore documents in the MIDDLE.
(Liu et al., 2023 — "Lost in the Middle")

Practical implication: put the most relevant documents first and last,
not in the middle. Or interleave relevant/less-relevant documents.
"""

print(f"\n--- 'Lost in the middle' document ordering ---")
print(f"  Best practice: place most relevant docs at positions 1 and N")
print(f"  Worst practice: bury the best document in the middle")
print(f"\n  Strategy: sort by relevance, then interleave:")

example_docs = [f"Doc{i} (score={0.9 - i*0.1:.1f})" for i in range(6)]
print(f"  Original order (by score): {example_docs}")

interleaved = []
left, right = [], []
for i, doc in enumerate(example_docs):
    if i % 2 == 0:
        left.append(doc)
    else:
        right.append(doc)
interleaved = left + list(reversed(right))
print(f"  Interleaved order:         {interleaved}")
print(f"  → Most relevant at start AND end of context window")


# ═══════════════════════════════════════════════════════════════════════════
# EXERCISES
# ═══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("EXERCISES")
print("=" * 70)
print("""
Exercise 1: Better Embeddings
-------------------------------
Improve the SimpleEmbedder by adding:
  a) A second linear layer with GELU activation after the projection
  b) Positional encoding (add position indices to embeddings)
  c) Test: embed "king - man + woman" and see if the result is closer
     to "queen" than "king" in cosine similarity.
  Hint: even with random weights, the architecture should support this.

Exercise 2: Semantic Chunking
-------------------------------
Implement a semantic chunking strategy:
  a) Split the document into sentences
  b) Embed each sentence using the SimpleEmbedder
  c) Compute cosine similarity between consecutive sentences
  d) Split at points where similarity drops below a threshold (e.g., 0.7)
  e) Compare the chunks to recursive chunking — are the boundaries better?
  Hint: this groups sentences that are semantically related, not just
  adjacent by character count.

Exercise 3: Hybrid Search
---------------------------
Implement a hybrid retrieval system that combines:
  a) Dense search: cosine similarity on embeddings (what we built)
  b) Sparse search: BM25-style keyword matching (TF-IDF)
     — Count word frequencies in each doc (TF)
     — Compute inverse document frequency (IDF) across corpus
     — Score = sum of TF * IDF for query terms in document
  c) Combine scores: final = alpha * dense + (1-alpha) * sparse
  d) Test with alpha = 0.0, 0.5, 1.0 — which works best for
     keyword-heavy vs semantic queries?

Exercise 4: Context Window Packing
-------------------------------------
Implement an optimal context packing strategy:
  a) Given a list of retrieved documents sorted by relevance
  b) A fixed token budget (e.g., 2048 tokens)
  c) Pack as many high-relevance documents as possible (knapsack-style)
  d) If a document is too long, truncate it (keep the first N tokens)
  e) Apply "lost in the middle" reordering
  f) Return the final packed context string
  Test with documents of varying lengths: [100, 500, 200, 800, 150] tokens.

Exercise 5: End-to-End RAG Evaluation
---------------------------------------
Build a mini evaluation harness:
  a) Create 10 question-answer pairs where answers come from knowledge_docs
  b) For each question, run the full RAG pipeline
  c) Check if the correct source document appears in the retrieved context
  d) Compute Recall@K for K = 1, 3, 5
  e) Try different chunk sizes (100, 200, 400 chars) and compare Recall@3
  f) Which chunking strategy gives the best retrieval accuracy?
""")
