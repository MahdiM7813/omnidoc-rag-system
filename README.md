# OmniDoc RAG: Production-Grade Multimodal Conversational RAG for Complex PDFs

<p align="center">
  <img src="assets/Omnidoc RAG.jpg" width="950"/>
</p>

<p align="center"><i>Figure: End-to-end architecture of OmniDoc RAG system</i></p>

A production-ready repository generated from three exploratory notebooks for:

1. Text-only conversational RAG over PDFs with LangChain, FAISS, HuggingFace embeddings, and Groq.
2. Multimodal RAG over complex PDFs containing text, tables, and images.
3. Conversational multimodal RAG with history-aware query rewriting.

The original notebook logic has been redesigned into a modular, configurable, testable system. Notebook-only patterns such as Colab paths, inline package installs, and hardcoded API keys were removed.

## Architecture

```text
PDFs
  │
  ▼
src/pipelines/ingestion.py
  ├── text mode: PyPDFLoader → text splitter
  └── multimodal mode: unstructured partition_pdf → text/table/image summaries
  │
  ▼
modules/vectorstores.py
  ├── Chroma
  └── FAISS
  │
  ▼
src/pipelines/rag.py
  ├── HistoryAwareQueryRewriter
  ├── Retriever
  ├── Context Formatter
  ├── AnswerGenerator
  └── Short-term + long-term conversation memory
  │
  ▼
CLI / FastAPI / Evaluation
```

## Folder Structure

```text
production_multimodal_rag/
├── app/                    # FastAPI interface and service layer
├── configs/                # YAML configuration and evaluation questions
├── data/                   # Runtime data, indexes, and cache placeholders
├── docs/                   # Architecture, usage, evaluation, and security notes
├── scripts/                # CLI scripts for ingest, query, evaluation, downloads
├── src/
│   ├── core/               # Config, logging, shared models, exceptions
│   ├── modules/            # Loaders, summarizers, vector stores, memory, eval
│   ├── pipelines/          # Ingestion and RAG orchestration
│   └── utils/              # Filesystem, hashing, base64 helpers
├── tests/                  # Unit tests with fake dependencies
├── Dockerfile
├── Makefile
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
git clone <your-repo-url>
cd production_multimodal_rag
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e .
cp .env.example .env
```

Add keys to `.env` or export them in your shell:

```bash
export GROQ_API_KEY="..."
export OPENAI_API_KEY="..."
```

For multimodal PDF extraction, your system also needs native tools such as `poppler-utils`. The Dockerfile installs the common runtime dependencies.

## Configuration

All runtime settings live in `configs/default.yaml`. Important switches:

- `document_processing.mode`: `text` or `multimodal`
- `vectorstore.provider`: `chroma` or `faiss`
- `llm.provider`: `groq` or `openai`
- `embeddings.provider`: `openai` or `huggingface`
- `retrieval.top_k`: number of retrieved chunks
- `conversation.max_turns`: sliding chat history window

No model name, chunk size, path, or hyperparameter is hardcoded in the source code.

## Usage

Place PDFs under `data/raw/`, then ingest:

```bash
PYTHONPATH=src python scripts/ingest.py --config configs/default.yaml --pdf-dir data/raw
```

Ask a single question:

```bash
PYTHONPATH=src python scripts/query.py --config configs/default.yaml \
  --question "What are the key contributions of the paper?"
```

Start interactive chat:

```bash
PYTHONPATH=src python scripts/query.py --config configs/default.yaml
```

## API Usage

Run the API:

```bash
PYTHONPATH=src uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Health check:

```bash
curl http://localhost:8000/health
```

Ingest PDFs visible to the server:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"paths": ["data/raw"]}'
```

Query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain the attention mechanism.", "session_id": "demo"}'
```

Clear a session:

```bash
curl -X DELETE http://localhost:8000/sessions/demo
```

## Evaluation

Edit `configs/eval_questions.yaml`, then run:

```bash
PYTHONPATH=src python scripts/evaluate.py --config configs/default.yaml
```

Results are written to `data/processed/eval_results.jsonl`.

The included evaluator is intentionally lightweight and deterministic. For serious production evaluation, add semantic metrics such as retrieval recall, groundedness, citation precision, answer faithfulness, and human review workflows.

## Testing and Quality

```bash
make test
make lint
```

The test suite avoids live LLM calls by using fakes/mocks for pipeline-level behavior.

## Deployment

Build and run with Docker:

```bash
docker build -t production-multimodal-rag .
docker run --env-file .env -p 8000:8000 production-multimodal-rag
```

Mount data for persistence:

```bash
docker run --env-file .env -p 8000:8000 \
  -v "$PWD/data:/app/data" production-multimodal-rag
```

## Security Notes

- Hardcoded API keys from the notebooks were not carried into this repository.
- Use `.env`, a secret manager, or your deployment platform's secret store.
- Do not enable FAISS dangerous deserialization unless loading an index you created and trust.
- User-uploaded PDFs should be treated as untrusted input in hosted environments.

## Limitations

- Multimodal extraction quality depends on `unstructured`, OCR/native dependencies, and PDF complexity.
- Image summaries require a vision-capable model.
- The default cache is a simple JSON disk cache, not a distributed cache.
- The default evaluation is a smoke-test framework, not a full RAG benchmark.

## Future Work

- Add async ingestion jobs with a queue.
- Add authenticated upload endpoints.
- Add reranking and hybrid sparse+dense retrieval.
- Add structured citation spans.
- Add observability with OpenTelemetry.
- Add database-backed sessions and distributed cache.
