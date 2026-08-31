"""
Check if documents are safe to show to customers.
Some documents should never be shown (like internal notes or outdated policies).
"""

from typing import Any
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import AUTHORITY_WEIGHTS, STATUS_WEIGHTS, AUDIENCE_WEIGHTS


def is_eligible(chunk: dict[str, Any]) -> bool:
    """
    Check if a chunk can be shown to customers.

    We block:
      - Old or draft documents
      - Internal-only documents
      - Documents marked as not for customers
      - Documents without real authority
    """
    status = chunk.get("status", "active")
    audience = chunk.get("audience", "customer")
    authority = chunk.get("policy_authority", "official")
    customer_answering = chunk.get("customer_answering", True)

    if status in ("superseded", "draft"):
        return False
    if audience == "internal":
        return False
    if authority == "none":
        return False
    if not customer_answering:
        return False
    return True


def authority_score(chunk: dict[str, Any]) -> float:
    """
    Give a score [0 to 1] that shows how trustworthy this information is.
    Official, active, customer-facing documents get a higher score.
    """
    auth = AUTHORITY_WEIGHTS.get(chunk.get("policy_authority", "none"), 0.0)
    status = STATUS_WEIGHTS.get(chunk.get("status", "active"), 0.0)
    audience = AUDIENCE_WEIGHTS.get(chunk.get("audience", "customer"), 0.0)
    return auth * status * audience


def detect_conflict(chunks: list[dict[str, Any]]) -> tuple[bool, str]:
    """
    Check if two documents say different things about the same topic.

    We know about one conflict: the product care guide vs product card
    for the Breeze Tumbler about dishwasher safety.

    Returns: (conflict found or not, explanation message)
    """
    active_filenames = {
        c["filename"] for c in chunks if is_eligible(c)
    }

    # Known conflict: product-care vs product-card on dishwasher safety
    if ("11-product-care.md" in active_filenames and
            "12-breeze-tumbler-product-card.md" in active_filenames):
        return True, (
            "The current official sources conflict on dishwasher safety for the Breeze Tumbler. "
            "The Product Care Guide says hand-wash the stainless-steel body and only the lid is "
            "dishwasher-safe (top rack). The Breeze Tumbler product card states all components "
            "are dishwasher safe (top rack recommended). "
            "I recommend following the more cautious guidance (hand-wash the body) until this is "
            "resolved, and contacting support for confirmation."
        )

    return False, ""
