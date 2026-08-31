"""
Keep track of a conversation between a customer and the agent.

Stores the message history and remembers:
- What order ID they were asking about
- What topic they were discussing
So follow-up questions can be answered correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Session:
    """Store one conversation: messages and what we remember about it."""

    messages: list[dict[str, str]] = field(default_factory=list)

    # Things we remember for next turn
    last_order_id: str | None = None
    last_topic: str | None = None     # like "international shipping"
    turn_count: int = 0

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})
        self.turn_count += 1

    def add_assistant(self, text: str) -> None:
        self.messages.append({"role": "assistant", "content": text})

    def history_for_api(self) -> list[dict[str, str]]:
        """Get the messages formatted for the AI model."""
        return list(self.messages)

    def context_summary(self) -> str:
        """Show what we remember about this conversation (for debugging)."""
        parts = [f"Turn: {self.turn_count}"]
        if self.last_order_id:
            parts.append(f"Order in scope: {self.last_order_id}")
        if self.last_topic:
            parts.append(f"Last topic: {self.last_topic}")
        return " | ".join(parts)

    def reset(self) -> None:
        self.__init__()
