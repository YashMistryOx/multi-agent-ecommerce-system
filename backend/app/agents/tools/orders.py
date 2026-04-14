"""Order lookup tools backed by MongoDB `orders` collection."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pymongo.errors import PyMongoError

from app.db.mongo import find_order_document, get_orders_collection


def _str_field(doc: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        v = doc.get(k)
        if v is not None and v != "":
            return str(v).strip()
    return default


def _format_items(raw: Any) -> str:
    if raw is None:
        return "—"
    if isinstance(raw, list):
        parts = []
        for x in raw:
            if isinstance(x, dict):
                parts.append(
                    ", ".join(f"{k}: {v}" for k, v in x.items() if v is not None)
                )
            else:
                parts.append(str(x))
        return "; ".join(parts) if parts else "—"
    return str(raw)


def _format_order_doc(doc: dict[str, Any]) -> str:
    oid = _str_field(doc, "order_id", "orderId", "id")
    if not oid:
        oid = str(doc.get("_id", ""))

    status = _str_field(doc, "status", "order_status", default="Unknown")
    items = _format_items(doc.get("items") or doc.get("line_items"))
    carrier = _str_field(doc, "carrier", "shipping_carrier")
    eta = _str_field(doc, "eta", "estimated_delivery", "delivery_eta", "edd")
    last_update = _str_field(
        doc, "last_update", "status_message", "updated_at", "last_status_update"
    )

    return (
        f"Order {oid}\n"
        f"Status: {status}\n"
        f"Items: {items}\n"
        f"Carrier: {carrier or '—'}\n"
        f"ETA / delivery: {eta or '—'}\n"
        f"Last update: {last_update or '—'}"
    )


@tool
def get_order_by_id(order_id: str) -> str:
    """Look up a single order by Omnimarket order ID (e.g. OM-10001). Returns status, carrier, ETA, and items."""
    try:
        col = get_orders_collection()
        doc = find_order_document(col, order_id)
        if not doc:
            return (
                f"No order found for '{order_id}'. Ask the customer to confirm the ID "
                "(format OM-#####) or check their confirmation email."
            )
        return _format_order_doc(doc)
    except PyMongoError as e:
        return f"Database error while looking up the order: {e}"


@tool
def list_recent_orders(limit: int = 5) -> str:
    """List recent orders for the customer when they do not have an order ID handy."""
    lim = max(1, min(limit, 25))
    try:
        col = get_orders_collection()
        docs = list(col.find({}).sort("_id", -1).limit(lim))

        if not docs:
            return "No orders found in the database yet."

        lines = []
        for doc in docs:
            oid = _str_field(doc, "order_id", "orderId", "id") or str(doc.get("_id", ""))
            status = _str_field(doc, "status", "order_status", default="—")
            items = _format_items(doc.get("items") or doc.get("line_items"))
            short_items = items if len(items) < 80 else items[:77] + "…"
            lines.append(f"- {oid}: {status} — {short_items}")
        return "Recent orders:\n" + "\n".join(lines)
    except PyMongoError as e:
        return f"Database error while listing orders: {e}"


ORDERS_TOOLS = [get_order_by_id, list_recent_orders]
