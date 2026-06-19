"""Answer generation and query rewriting modules."""

from __future__ import annotations

from typing import Any

from core.config import Settings
from core.exceptions import DependencyNotInstalledError, GenerationError
from modules.llms import create_chat_model


class HistoryAwareQueryRewriter:
    """Rewrite follow-up user questions into standalone retrieval queries."""

    def __init__(self, settings: Settings):
        """Initialize rewriter."""
        self.llm = create_chat_model(settings.llm)
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        except ImportError as exc:
            raise DependencyNotInstalledError("Install langchain-core.") from exc

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "Rewrite the user's follow-up into a single standalone search query. "
                    "Return ONLY the query text. Do NOT answer the question.",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.chain = prompt | self.llm | StrOutputParser()

    def rewrite(self, question: str, chat_history: list[Any]) -> str:
        """Rewrite the question given chat history."""
        if not chat_history:
            return question
        try:
            return str(self.chain.invoke({"input": question, "chat_history": chat_history}))
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Question rewrite failed: {exc}") from exc


class AnswerGenerator:
    """Generate answers grounded in retrieved context."""

    def __init__(self, settings: Settings):
        """Initialize answer generator."""
        self.settings = settings
        self.llm = create_chat_model(settings.llm)
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        except ImportError as exc:
            raise DependencyNotInstalledError("Install langchain-core.") from exc

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", f"{settings.rag.system_prompt}\n\nContext:\n{{context}}"),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate(self, question: str, context: str, chat_history: list[Any] | None = None) -> str:
        """Generate an answer from question, context, and optional history."""
        try:
            return str(
                self.chain.invoke(
                    {
                        "input": question,
                        "context": context,
                        "chat_history": chat_history or [],
                    }
                )
            )
        except Exception as exc:  # noqa: BLE001
            raise GenerationError(f"Answer generation failed: {exc}") from exc
