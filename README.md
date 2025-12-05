# OOP-RAG
OOP for RAG with qdrant flow and embedded memory.
# RAG Refactor (simple demo)

This repository contains a refactored Retrieval-Augmented-Generation (RAG) demo:
- deterministic fake embedder (stable for tests)
- Qdrant integration (optional)
- HuggingFace LLM optional (controlled with env var)
- short-answer defaults and cleaning to avoid noisy outputs

# pip install -r requirements.txt
# docker run -p 6333:6333 qdrant/qdrant
# uvicorn "with_llm or app":app --reload --port 8000
