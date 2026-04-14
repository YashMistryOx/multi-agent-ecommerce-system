"""Returns / refund tools. Order existence uses the same MongoDB `orders` collection as the Orders agent."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pymongo.errors import PyMongoError

from app.agents.tools.orders import get_order_by_id, list_recent_orders
from app.db.mongo import find_order_document, get_orders_collection

_RETURN_CASES: dict[str, str] = {
    "RR-5001": "Open — awaiting item scan at warehouse",
    "RR-5002": "Refunded — ₹1,250 to original payment method (3–5 days)",
}


def _str_field(doc: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        v = doc.get(k)
        if v is not None and v != "":
            return str(v).strip()
    return default


def _eligibility_text_from_order(doc: dict[str, Any]) -> str:
    """Heuristic copy for support; tune against real order statuses."""
    oid = _str_field(doc, "order_id", "orderId", "id") or str(doc.get("_id", ""))
    st = _str_field(doc, "status", "order_status", default="").lower()
    if not st or st == "unknown":
        return (
            f"Order {oid} is on file. Confirm delivery date and return window with the customer "
            "before approving a return."
        )
    if "cancel" in st:
        return f"Order {oid} appears cancelled; returns usually do not apply unless policy says otherwise."
    if "deliver" in st:
        return (
            f"Order {oid} shows as delivered. Eligible for return if within the published return window "
            "(e.g. unused, original packaging). Confirm with policy."
        )
    if any(x in st for x in ("ship", "transit", "processing", "paid", "confirm")):
        return (
            f"Order {oid} is not yet marked delivered. Returns typically start after the customer receives "
            "the item; offer to help again once delivered."
        )
    return (
        f"Order {oid} is on file (status: {st or 'unknown'}). Confirm eligibility against Omnimarket return policy."
    )


@tool
def get_return_policy_summary() -> str:
    """Summarize Omnimarket return rules (demo). Use for general return-window and condition questions."""
    return (
        "Omnimarket returns (demo policy): Returns accepted within 7 days of delivery for "
        "unused items in original packaging. Refunds post within 5 business days after we "
        "receive and inspect the return. Some categories (digital, hygiene) may be excluded."
    )


@tool
def check_return_eligibility(order_id: str) -> str:
    """Check whether an order is eligible to start a return. Uses the same order database as order lookup."""
    try:
        col = get_orders_collection()
        doc = find_order_document(col, order_id)
        if not doc:
            return (
                f"No order found for '{order_id}'. Ask the customer to confirm the ID "
                "(format OM-#####) or use list_recent_orders / get_order_by_id."
            )
        return _eligibility_text_from_order(doc)
    except PyMongoError as e:
        return f"Database error while checking eligibility: {e}"


@tool
def start_return_request(order_id: str, reason: str) -> str:
    """Start a return request (demo). Verifies the order exists in the database first."""
    try:
        col = get_orders_collection()
        doc = find_order_document(col, order_id)
        if not doc:
            return (
                f"Cannot start return: no order '{order_id}' found. Confirm the order ID or look it up first."
            )
        oid = _str_field(doc, "order_id", "orderId", "id") or str(doc.get("_id", ""))
        case_id = "RR-5999"
        return (
            f"Return request created (demo) for order {oid}. Case ID: {case_id}. "
            f"Reason recorded: {reason[:200]}."
        )
    except PyMongoError as e:
        return f"Database error while starting return: {e}"


@tool
def get_return_case_status(case_id: str) -> str:
    """Look up status of an existing return case by ID (e.g. RR-5001)."""
    cid = case_id.strip().upper()
    if cid in _RETURN_CASES:
        return f"Case {cid}: {_RETURN_CASES[cid]}"
    return (
        f"No case '{case_id}' in demo records. Valid demo IDs: {', '.join(_RETURN_CASES)}."
    )


RETURNS_TOOLS = [
    get_order_by_id,
    list_recent_orders,
    get_return_policy_summary,
    check_return_eligibility,
    start_return_request,
    get_return_case_status,
]
