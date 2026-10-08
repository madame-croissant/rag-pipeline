# Modular RAG Pipeline

> **Project Status: Under Active Development**
> This repository is currently in active development. Features, API interfaces, and evaluation benchmarks are subject to refinement and updates.

A modular Retrieval-Augmented Generation (RAG) system built in Python. It combines hybrid search, cross-Encoder re-ranking, confidence guardrails, Groq LLM generation, citation verification, and automated evaluation into a clean pipeline.

---

## Key Features

- **Flexible Text Chunking**: Supports Fixed-Size, Recursive boundary, and Semantic splitting strategies.
- **Hybrid Search**: Combines dense vector search (ChromaDB) with sparse keyword search (BM25) using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Re-Ranking**: Scores and re-ranks top candidates using `ms-marco-MiniLM-L-6-v2`.
- **Retrieval Guardrails**: Detects and intercepts low-confidence or out-of-domain queries to prevent hallucinations.
- **Citation Verification**: Parses inline bracketed citations (`[1]`, `[2]`) from generated answers and verifies if cited context explicitly supports each claim.
- **Composite Scoring**: Evaluates outputs across retrieval confidence, citation accuracy, and query completeness.

---

## Project Structure

```text
├── data/
│   └── raw/                   # Raw Markdown source documents
├── preprocessing/
│   ├── loader.py              # Front-matter cleaning and document loading
│   ├── chunking.py            # Fixed, Recursive, and Semantic chunking
│   ├── vector_store.py        # ChromaDB vector database manager
│   ├── bm25.py                # Sparse BM25 indexer
│   └── preprocessing_handler.py # Full preprocessing workflow coordinator
├── retrieval/
│   └── hybrid_retrieval.py    # Dense/sparse retrieval, RRF, and re-ranking
├── generation/
│   └── generator.py           # Prompt formatting and Groq response generation
├── evaluation/
│   ├── verifier.py            # Inline citation verifier
│   ├── guardrail.py           # Retrieval confidence threshold check
│   ├── scorer.py              # Confidence and composite scoring logic
│   └── eval_suite.py          # Groq-powered LLM judge and test suite
├── keys.txt                   # Plain-text API key configuration
├── main.py                    # Main pipeline entry point
└── README.md