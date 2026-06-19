# Evaluation

The repository includes a lightweight evaluation harness:

- Input: `configs/eval_questions.yaml`
- Runner: `scripts/evaluate.py`
- Output: `data/processed/eval_results.jsonl`

Current deterministic checks:

- answer contains expected keywords
- at least one source was retrieved
- latency per question

Recommended production extensions:

- retrieval recall at k
- answer groundedness and citation precision
- hallucination checks against context
- regression sets for known PDFs
- human-in-the-loop review for critical answers
