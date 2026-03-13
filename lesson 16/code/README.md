# Lesson 16: Retrieval-Augmented Generation (RAG)

Extend your LLM with external knowledge retrieval — ground answers in real data instead of hallucinated facts.

## Prerequisites

- Python 3.10+
- PyTorch (`pip install torch`)
- No other external dependencies

## How to Run

```bash
python3 rag_pipeline.py
```

The script runs all 6 parts sequentially, printing explanations and results for each section.

## Contents

### Part 1 — Why RAG
LLMs hallucinate when asked about things outside their training data. This section demonstrates the problem with concrete examples and introduces the three core issues RAG solves: knowledge cutoffs, hallucination, and domain specificity.

### Part 2 — Embedding Models
Convert text into fixed-size vectors that capture semantic meaning. Implements a sentence embedding model from scratch using token embeddings, mean pooling, and L2 normalization. Demonstrates cosine similarity search over a corpus.

**Key classes:** `SimpleTokenizer`, `SimpleEmbedder`
**Key functions:** `encode_texts()`, `cosine_similarity_search()`

### Part 3 — Vector Database
Build a naive in-memory vector store from scratch with add/search operations. Covers the API that all production vector databases share (FAISS, ChromaDB, Qdrant) and explains why brute-force O(n) search works for small corpora while HNSW is needed at scale.

**Key classes:** `NaiveVectorStore`

### Part 4 — Chunking Strategies
Three document splitting strategies compared on the same sample document: fixed-size chunking with overlap, sentence-based chunking that respects boundaries, and recursive chunking (the LangChain approach). Includes a quantitative comparison table.

**Key functions:** `chunk_fixed_size()`, `chunk_by_sentences()`, `chunk_recursive()`

### Part 5 — Retrieval Pipeline
Complete end-to-end pipeline: query → embed → search → rerank → augment prompt. Implements a simplified cross-encoder reranker that scores query-document pairs using token-level attention. Shows how to construct a RAG-augmented prompt ready for generation.

**Key classes:** `SimpleCrossEncoder`
**Key functions:** `retrieve()`, `rerank()`, `build_rag_prompt()`, `rag_pipeline()`

### Part 6 — Context Window Management
Token counting, context budget allocation, and truncation strategies for fitting retrieved documents within a fixed context window. Covers the "lost in the middle" problem and document ordering strategies.

**Key classes:** `ContextBudget`
**Key functions:** `estimate_tokens()`, `truncate_end()`, `truncate_middle()`

## Exercises

5 exercises are printed at the end of the script:

| # | Topic | What You Practice |
|---|-------|-------------------|
| 1 | Better Embeddings | Multi-layer projections, positional encoding, analogy test |
| 2 | Semantic Chunking | Split documents at meaning boundaries using embedding similarity |
| 3 | Hybrid Search | Combine dense (embedding) and sparse (BM25/TF-IDF) retrieval |
| 4 | Context Window Packing | Knapsack-style doc selection + lost-in-the-middle reordering |
| 5 | End-to-End RAG Evaluation | Recall@K metrics across chunk sizes and strategies |

## Key Takeaways

- **RAG solves hallucination** by grounding answers in retrieved documents, not memorized knowledge
- **Embedding models** map text to vectors; cosine similarity finds semantically related content
- **Vector databases** store embeddings and support fast nearest-neighbor search (brute-force or HNSW)
- **Chunking matters** — too large dilutes relevance, too small loses context; recursive splitting is the best default
- **Reranking** with a cross-encoder dramatically improves precision on top-k candidates
- **Context budget** must account for system prompt + docs + query + generation room
- **Lost in the middle** — put the most relevant documents at the start and end of context, not the middle

## What's Next

Congratulations — you've completed the Build Your Own LLM course! You now have the knowledge to build, train, fine-tune, align, quantize, serve, and augment a language model from scratch.
