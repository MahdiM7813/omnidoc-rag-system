"""CLI entry point for querying a persisted RAG index."""

from __future__ import annotations

import argparse

from core.config import load_settings
from core.logging import setup_logging
from pipelines.rag import RagPipeline


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Query the RAG system.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--question", help="Question to ask. If omitted, starts interactive mode.")
    parser.add_argument("--session-id", default="cli", help="Conversation session ID.")
    return parser.parse_args()


def main() -> None:
    """Run query CLI."""
    args = parse_args()
    settings = load_settings(args.config)
    setup_logging(settings.logging.level, settings.logging.json_logs, settings.app.verbose)
    pipeline = RagPipeline.from_config(settings)

    if args.question:
        response = pipeline.ask(args.question, session_id=args.session_id)
        print(response.answer)
        return

    print("Interactive RAG. Type 'exit' to quit.")
    while True:
        question = input("Question> ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        response = pipeline.ask(question, session_id=args.session_id)
        print(f"Answer> {response.answer}\n")


if __name__ == "__main__":
    main()
