"""Conversation memory modules."""

from __future__ import annotations

import json
from pathlib import Path

from core.models import ChatTurn


class ShortTermMemory:
    """In-memory sliding window chat history."""

    def __init__(self, max_turns: int = 8):
        """Initialize memory."""
        self.max_turns = max_turns
        self.turns: list[ChatTurn] = []

    def add(self, user: str, assistant: str) -> None:
        """Append a chat turn and enforce the max window."""
        self.turns.append(ChatTurn(user=user, assistant=assistant))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns :]

    def clear(self) -> None:
        """Clear all turns."""
        self.turns.clear()

    def to_langchain_messages(self) -> list[object]:
        """Convert stored turns to LangChain message objects."""
        try:
            from langchain_core.messages import AIMessage, HumanMessage
        except ImportError:
            return []
        messages: list[object] = []
        for turn in self.turns:
            messages.append(HumanMessage(content=turn.user))
            messages.append(AIMessage(content=turn.assistant))
        return messages


class LongTermMemoryStore:
    """Simple JSON-backed long-term memory grouped by session ID."""

    def __init__(self, path: str | Path):
        """Initialize store."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_session(self, session_id: str, max_turns: int = 8) -> ShortTermMemory:
        """Load a session into short-term memory."""
        memory = ShortTermMemory(max_turns=max_turns)
        payload = self._read_all()
        for item in payload.get(session_id, [])[-max_turns:]:
            memory.turns.append(ChatTurn.model_validate(item))
        return memory

    def save_session(self, session_id: str, memory: ShortTermMemory) -> None:
        """Persist a session."""
        payload = self._read_all()
        payload[session_id] = [turn.model_dump(mode="json") for turn in memory.turns]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def clear_session(self, session_id: str) -> None:
        """Delete a session from long-term memory."""
        payload = self._read_all()
        payload.pop(session_id, None)
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _read_all(self) -> dict[str, list[dict[str, str]]]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8") or "{}")
