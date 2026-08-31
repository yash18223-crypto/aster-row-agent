"""
Track what happens during a conversation for debugging.

If DEBUG mode is on, records:
- What the user asked
- What we retrieved
- What tools we called
- What the AI answered
- Did we need to hand off to a human

Never records passwords or secret information.
"""

from __future__ import annotations

import json
import sys
import os
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DEBUG


def _safe(obj: Any) -> Any:
    """Make data safe to log - remove passwords and secret keys."""
    if isinstance(obj, dict):
        return {k: _safe(v) for k, v in obj.items()
                if k not in ("ANTHROPIC_API_KEY", "api_key", "secret")}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return str(obj)


class AgentTrace:
    """Collect information about one conversation turn, then show it (if debugging)."""

    def __init__(self, query: str, session_context: str) -> None:
        self.trace: dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "session_context": session_context,
            "intent": None,
            "retrieved_chunks": [],
            "conflict_detected": False,
            "conflict_description": "",
            "tool_called": None,
            "tool_args": {},
            "tool_result": None,
            "handoff": False,
            "final_answer": None,
            "errors": [],
        }

    def set_intent(self, intent: str) -> None:
        self.trace["intent"] = intent

    def set_retrieval(self, result: dict[str, Any]) -> None:
        self.trace["retrieved_chunks"] = [
            {
                "source": c.get("source_ref"),
                "score": c.get("score"),
                "status": c.get("status"),
            }
            for c in result.get("chunks", [])
        ]
        self.trace["conflict_detected"] = result.get("conflict", False)
        self.trace["conflict_description"] = result.get("conflict_description", "")

    def set_tool_call(self, tool_name: str, args: dict) -> None:
        self.trace["tool_called"] = tool_name
        self.trace["tool_args"] = args

    def set_tool_result(self, result: dict) -> None:
        # Sanitise: never log internal/raw customer data
        safe_result = _safe(result)
        self.trace["tool_result"] = safe_result

    def set_handoff(self, handoff: bool) -> None:
        self.trace["handoff"] = handoff

    def set_answer(self, answer: str) -> None:
        self.trace["final_answer"] = answer

    def add_error(self, err: str) -> None:
        self.trace["errors"].append(err)

    def emit(self) -> None:
        if not DEBUG:
            return
        print("\n" + "=" * 60, file=sys.stderr)
        print("[AGENT TRACE]", file=sys.stderr)
        print(json.dumps(_safe(self.trace), indent=2), file=sys.stderr)
        print("=" * 60 + "\n", file=sys.stderr)
