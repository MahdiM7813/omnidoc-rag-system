.PHONY: install test lint format api ingest query evaluate

install:
	python -m pip install -U pip
	python -m pip install -e .

test:
	PYTHONPATH=src pytest

lint:
	ruff check src app scripts tests

format:
	ruff format src app scripts tests

api:
	PYTHONPATH=src uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

ingest:
	PYTHONPATH=src python scripts/ingest.py --config configs/default.yaml --pdf-dir data/raw

query:
	PYTHONPATH=src python scripts/query.py --config configs/default.yaml

evaluate:
	PYTHONPATH=src python scripts/evaluate.py --config configs/default.yaml
