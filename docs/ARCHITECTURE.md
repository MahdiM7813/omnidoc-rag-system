# Architecture

The repository separates logic, configuration, execution, and interfaces.

## Layers

- `src/core`: shared primitives that do not depend on LangChain runtime objects.
- `src/modules`: reusable infrastructure modules such as loaders, summarizers, embeddings, vector stores, memory, and evaluation.
- `src/pipelines`: orchestration layer for ingestion and question answering.
- `app`: FastAPI interface plus a service layer.
- `scripts`: command-line execution wrappers.
- `configs`: model, retrieval, chunking, logging, and evaluation settings.

## RAG Flow

1. Ingestion receives PDF paths.
2. Text mode loads pages and splits chunks.
3. Multimodal mode partitions PDFs into text/table/image elements.
4. Summarizers convert each modality into retrieval-friendly text.
5. Documents are embedded and persisted in Chroma or FAISS.
6. Query pipeline optionally rewrites follow-up questions using memory.
7. Retriever fetches top-k documents.
8. Context formatter composes a grounded prompt.
9. Generator answers from context only.
10. Sources and session memory are returned/persisted.

## Why Summarize Multimodal Content?

Vector stores retrieve text. Tables and images are summarized into text while preserving modality metadata. This lets the retriever find visual or tabular evidence without forcing every question-time answer call to include raw images or full HTML tables.
