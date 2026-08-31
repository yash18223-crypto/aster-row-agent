"""
Look up orders safely.

Rules:
 - Fix order IDs (uppercase, remove spaces)
 - Check format before looking up
 - Never show: email, address, secret notes, risk scores
 - Don't show delivery dates for cancelled/returned orders
 - Give clear error messages for bad order IDs
"""

from __future__ import annotations

import json
import re
import os
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ORDERS_FILE

# If order is done, don't show the expected delivery date anymore
TERMINAL_STATUSES = {"cancelled", "returned", "delivered"}

# These fields must never be shown to customers
_INTERNAL_BLOCK = {"email", "shipping_address", "internal"}


def _load_orders() -> dict[str, Any]:
    """Load orders from the JSON file. Make order IDs uppercase to match."""
    with open(ORDERS_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {o["order_id"].upper(): o for o in raw.get("orders", [])}


def _sanitise(order: dict[str, Any]) -> dict[str, Any]:
    """
    Remove secret information from the order before showing it to the customer.
    Never show: email, address, risk score, internal notes, or tags.
    """
    status = order.get("status", "unknown")
    is_terminal = status in TERMINAL_STATUSES

    safe: dict[str, Any] = {
        "order_id":   order.get("order_id"),
        "status":     status,
        "items":      [
            {"name": i.get("name"), "quantity": i.get("quantity"),
             "final_sale": i.get("final_sale")}
            for i in order.get("items", [])
        ],
        "placed_at":  order.get("placed_at"),
        "membership_tier": order.get("membership_tier"),
        "customer_safe_message": order.get("customer_safe_message", ""),
    }

    # Shipping info — only when relevant
    if not is_terminal:
        safe["carrier"]          = order.get("carrier")
        safe["tracking_number"]  = order.get("tracking_number")
        safe["shipped_at"]       = order.get("shipped_at")
        safe["estimated_delivery"] = order.get("estimated_delivery")
    else:
        # Explicitly note that ETA / tracking are suppressed for terminal orders
        safe["delivery_note"] = (
            f"This order has status '{status}'. "
            "Delivery estimates and tracking details are not applicable."
        )
        safe["carrier"]       = order.get("carrier") if status == "delivered" else None
        safe["tracking_number"] = None
        safe["estimated_delivery"] = None

    return safe


def lookup_order(order_id: str) -> dict[str, Any]:
    """
    Main tool entry point.

    Returns either:
      {"found": True,  "order": <safe_dict>}
      {"found": False, "error": "<reason>"}
    """
    # 1. Normalise
    normalised = order_id.strip().upper()

    # 2. Validate format  (ORD-XXXX)
    if not re.match(r'^ORD-\d+$', normalised):
        return {
            "found": False,
            "error": (
                f"'{order_id}' does not look like a valid order ID. "
                "Order IDs follow the format ORD-followed-by-numbers (e.g. ORD-1007)."
            ),
        }

    # 3. Lookup
    orders = _load_orders()
    order = orders.get(normalised)

    if order is None:
        return {
            "found": False,
            "error": (
                f"No order with ID {normalised} was found. "
                "Please double-check the order ID or contact support."
            ),
        }

    # 4. Sanitise and return
    return {"found": True, "order": _sanitise(order)}


# For manual testing
if __name__ == "__main__":
    for test_id in ["ORD-1007", "ord-1004", " ORD-1011 ", "ORD-9999", "12345"]:
        result = lookup_order(test_id)
        print(f"\n--- {test_id!r} ---")
        print(json.dumps(result, indent=2, default=str))
