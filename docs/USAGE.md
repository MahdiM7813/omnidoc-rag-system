# Usage Guide

## Text-only RAG

Set the following in `configs/default.yaml`:

```yaml
document_processing:
  mode: text
embeddings:
  provider: huggingface
  model_name: sentence-transformers/all-MiniLM-L6-v2
vectorstore:
  provider: faiss
```

Then run:

```bash
PYTHONPATH=src python scripts/ingest.py --pdf-dir data/raw
PYTHONPATH=src python scripts/query.py --question "What is YOLOv10?"
```

## Multimodal RAG

Use:

```yaml
document_processing:
  mode: multimodal
vectorstore:
  provider: chroma
```

Then run ingestion. Multimodal mode uses `unstructured.partition.pdf` and may require Poppler and OCR dependencies.

## Downloading Google Drive PDFs

```bash
PYTHONPATH=src python scripts/download_gdrive.py "https://drive.google.com/file/d/.../view"
```

Then ingest from `data/raw`.
