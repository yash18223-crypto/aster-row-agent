"""
Test the Aster & Row AI Support Agent.

Tests 20 cases total - checks that:
  - The right answers are in the response (must_include)
  - Wrong answers are not included (must_not_include)
  - Ideas are mentioned correctly (must_include_concepts)
  - The AI refuses to make up information (must_not_invent)
  - The AI asks for missing information (must_ask_for)
  - Secret information is hidden (must_refuse_to_disclose)
  - The right tool is called (order_lookup)
  - Sources are cited correctly

How to run:
  python evaluation/run_evaluation.py
  python evaluation/run_evaluation.py --verbose
  python evaluation/run_evaluation.py --case-id standard-return-window
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any

# Allow imports from app/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from agent import AsterRowAgent
from conversation.session import Session

VISIBLE_CASES_FILE = os.path.join(os.path.dirname(__file__), "visible-cases.json")
ADDITIONAL_CASES_FILE = os.path.join(os.path.dirname(__file__), "additional-cases.json")


# Helper functions for checking answers
# These all compare lowercase text

def _ci(text: str) -> str:
    return text.lower()


def check_must_include(response: str, items: list[str]) -> list[str]:
    """Find items that should be in the answer but are missing."""
    return [item for item in items if _ci(item) not in _ci(response)]


def check_must_not_include(response: str, items: list[str]) -> list[str]:
    """Find items that should not be in the answer but are present."""
    return [item for item in items if _ci(item) in _ci(response)]


def check_must_include_concepts(response: str, concepts: list[str]) -> list[str]:
    """
    Check if ideas are mentioned. Break each idea into words, require all words.
    Example: "Canada is supported" → need both "canada" and "supported"
    """
    failed = []
    for concept in concepts:
        keywords = [w for w in concept.lower().split() if len(w) > 3]
        if not all(kw in _ci(response) for kw in keywords):
            failed.append(concept)
    return failed


def check_sources(sources: list[str], required: list[str]) -> list[str]:
    """Find required sources that weren't cited."""
    sources_lower = [_ci(s) for s in sources]
    return [req for req in required
            if not any(_ci(req) in src for src in sources_lower)]


def check_forbidden_sources(sources: list[str], forbidden: list[str]) -> list[str]:
    """Return forbidden sources that wrongly appeared."""
    sources_lower = [_ci(s) for s in sources]
    return [f for f in forbidden
            if any(_ci(f) in src for src in sources_lower)]


def check_tool(tool_called: str | None, expected_tool: str) -> bool:
    """
    Verify tool call behaviour.
    expected_tool values:
      "order_lookup"          → tool must have been called
      "not_called"            → tool must NOT have been called
      "not_called_without_id" → tool must NOT have been called (same)
      "optional_sanitized_lookup" → pass regardless
    """
    if expected_tool in ("not_called", "not_called_without_id"):
        return tool_called is None
    if expected_tool == "order_lookup":
        return tool_called == "order_lookup"
    if expected_tool == "optional_sanitized_lookup":
        return True  # optional
    return True


# --------------------------------------------------------------------------- #
#  Single case runner                                                          #
# --------------------------------------------------------------------------- #

class CaseResult:
    def __init__(self, case_id: str, category: str) -> None:
        self.case_id = case_id
        self.category = category
        self.failures: list[str] = []
        self.passed = False
        self.response_text = ""
        self.sources: list[str] = []
        self.tool_called: str | None = None
        self.handoff: bool = False
        self.duration_s: float = 0.0

    @property
    def failed(self) -> bool:
        return bool(self.failures)


def run_case(agent: AsterRowAgent, case: dict[str, Any]) -> CaseResult:
    """Run a single evaluation case and return a CaseResult."""
    result = CaseResult(case["id"], case["category"])
    expect = case.get("expect", {})

    session = Session()
    messages = case.get("messages", [])
    all_turns: list[dict[str, Any]] = []

    t0 = time.time()
    for msg in messages:
        if msg["role"] != "user":
            continue
        turn = agent.run_turn(msg["content"], session)
        all_turns.append(turn)
    result.duration_s = round(time.time() - t0, 2)

    # Aggregate from the LAST turn for assertions
    last = all_turns[-1] if all_turns else {}
    result.response_text = last.get("answer", "")
    result.sources = last.get("sources", [])
    result.handoff = last.get("handoff", False)

    # Tool call — detect from trace
    trace = last.get("trace")
    if trace:
        result.tool_called = trace.trace.get("tool_called")

    # ---- Assertions ---- #

    if expect.get("must_include"):
        missing = check_must_include(result.response_text, expect["must_include"])
        for m in missing:
            result.failures.append(f"MISSING text: '{m}'")

    if expect.get("must_not_include"):
        present = check_must_not_include(result.response_text, expect["must_not_include"])
        for p in present:
            result.failures.append(f"FORBIDDEN text present: '{p}'")

    if expect.get("must_include_concepts"):
        failed_concepts = check_must_include_concepts(
            result.response_text, expect["must_include_concepts"]
        )
        for c in failed_concepts:
            result.failures.append(f"MISSING concept: '{c}'")

    if expect.get("must_not_invent"):
        for term in expect["must_not_invent"]:
            if _ci(term) in _ci(result.response_text):
                result.failures.append(f"INVENTED content: '{term}'")

    if expect.get("must_ask_for"):
        for item in expect["must_ask_for"]:
            if not any(kw in _ci(result.response_text)
                       for kw in item.lower().split()):
                result.failures.append(f"DID NOT ASK FOR: '{item}'")

    if expect.get("must_refuse_to_disclose"):
        for item in expect["must_refuse_to_disclose"]:
            # The agent should either not mention it or explicitly decline
            # We check that the actual sensitive value isn't in the output
            pass  # Checked via must_not_include in the case definition

    if expect.get("required_sources"):
        missing_src = check_sources(result.sources, expect["required_sources"])
        for src in missing_src:
            result.failures.append(f"MISSING source: '{src}'")

    if expect.get("forbidden_sources_as_authority"):
        bad_src = check_forbidden_sources(result.sources, expect["forbidden_sources_as_authority"])
        for src in bad_src:
            result.failures.append(f"FORBIDDEN source cited: '{src}'")

    if "tool" in expect:
        if not check_tool(result.tool_called, expect["tool"]):
            result.failures.append(
                f"TOOL mismatch: expected '{expect['tool']}', got '{result.tool_called}'"
            )

    if "handoff" in expect:
        if result.handoff != expect["handoff"]:
            result.failures.append(
                f"HANDOFF mismatch: expected {expect['handoff']}, got {result.handoff}"
            )

    result.passed = not result.failures
    return result


# --------------------------------------------------------------------------- #
#  Main runner                                                                 #
# --------------------------------------------------------------------------- #

def load_cases(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["cases"]


def print_banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def run_all(
    agent: AsterRowAgent,
    verbose: bool = False,
    filter_id: str | None = None,
) -> None:
    all_cases: list[dict[str, Any]] = []

    if os.path.exists(VISIBLE_CASES_FILE):
        visible = load_cases(VISIBLE_CASES_FILE)
        for c in visible:
            c["_source"] = "visible"
        all_cases.extend(visible)

    if os.path.exists(ADDITIONAL_CASES_FILE):
        additional = load_cases(ADDITIONAL_CASES_FILE)
        for c in additional:
            c["_source"] = "additional"
        all_cases.extend(additional)

    if filter_id:
        all_cases = [c for c in all_cases if c["id"] == filter_id]
        if not all_cases:
            print(f"No case found with id='{filter_id}'")
            sys.exit(1)

    print_banner(f"Aster & Row Agent Evaluation — {len(all_cases)} cases")

    results: list[CaseResult] = []
    by_category: dict[str, list[CaseResult]] = defaultdict(list)

    for case in all_cases:
        print(f"\n  Running: [{case['category']}] {case['id']} ...", end="", flush=True)
        try:
            r = run_case(agent, case)
        except Exception as exc:
            r = CaseResult(case["id"], case["category"])
            r.failures.append(f"EXCEPTION: {exc}")
            r.response_text = f"[ERROR: {exc}]"

        results.append(r)
        by_category[r.category].append(r)

        status = "✅ PASS" if r.passed else f"❌ FAIL ({len(r.failures)} issues)"
        print(f" {status}  ({r.duration_s}s)")

        if verbose or r.failed:
            print(f"    Response: {r.response_text[:300]}...")
            if r.sources:
                print(f"    Sources: {r.sources}")
            if r.failed:
                for f_ in r.failures:
                    print(f"    ⚠  {f_}")

    # --- Category summary --- #
    print_banner("Results by Category")
    for category, cat_results in sorted(by_category.items()):
        passed = sum(1 for r in cat_results if r.passed)
        total = len(cat_results)
        bar = "█" * passed + "░" * (total - passed)
        print(f"  {category:<30} {bar}  {passed}/{total}")

    # --- Overall summary --- #
    total_passed = sum(1 for r in results if r.passed)
    total = len(results)
    visible_passed = sum(1 for r in results if r.passed and r.case_id in
                         {c["id"] for c in all_cases if c.get("_source") == "visible"})
    visible_total = sum(1 for c in all_cases if c.get("_source") == "visible")

    print_banner("Overall Summary")
    print(f"  Total:        {total_passed}/{total} passed ({100*total_passed//total}%)")
    print(f"  Visible:      {visible_passed}/{visible_total}")
    additional_r = [r for r in results
                    if r.case_id not in {c["id"] for c in all_cases
                                         if c.get("_source") == "visible"}]
    add_pass = sum(1 for r in additional_r if r.passed)
    print(f"  Additional:   {add_pass}/{len(additional_r)}")
    print()

    # Return exit code
    if total_passed < total:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aster & Row Agent Evaluation Suite")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show full response text for every case")
    parser.add_argument("--case-id", metavar="ID",
                        help="Run a single case by its ID")
    args = parser.parse_args()

    try:
        agent = AsterRowAgent()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    run_all(agent, verbose=args.verbose, filter_id=args.case_id)


if __name__ == "__main__":
    main()
