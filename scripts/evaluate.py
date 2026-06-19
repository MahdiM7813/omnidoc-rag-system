"""CLI entry point for RAG evaluation."""

from __future__ import annotations

import argparse
import json

from core.config import load_settings
from core.logging import setup_logging
from modules.evaluation import load_evaluation_questions, run_evaluation
from pipelines.rag import RagPipeline


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    """Run evaluation."""
    args = parse_args()
    settings = load_settings(args.config)
    setup_logging(settings.logging.level, settings.logging.json_logs, settings.app.verbose)
    questions = load_evaluation_questions(settings.evaluation.questions_path)
    pipeline = RagPipeline.from_config(settings)
    results = run_evaluation(pipeline, questions, settings.evaluation.output_path)
    print(json.dumps([result.model_dump(mode="json") for result in results], indent=2))


if __name__ == "__main__":
    main()
