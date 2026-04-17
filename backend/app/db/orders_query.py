"""Order queries scoped by customer email (authentication)."""

from __future__ import annotations

import re
from typing import Any

from pymongo.collection import Collection

from app.db.mongo import find_order_document, get_orders_collection


def _str_field(doc: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        v = doc.get(k)
        if v is not None and v != "":
            return str(v).strip()
    return default


def order_email_on_doc(doc: dict[str, Any]) -> str | None:
    for key in ("customer_email", "email", "buyer_email", "contact_email"):
        v = doc.get(key)
        if v and str(v).strip():
            return str(v).strip()
    return None


def _email_or_query(email: str) -> dict[str, Any]:
    e = email.strip().lower()
    esc = re.escape(e)
    return {
        "$or": [
            {"customer_email": {"$regex": f"^{esc}$", "$options": "i"}},
            {"email": {"$regex": f"^{esc}$", "$options": "i"}},
            {"buyer_email": {"$regex": f"^{esc}$", "$options": "i"}},
            {"contact_email": {"$regex": f"^{esc}$", "$options": "i"}},
        ]
    }


def _sort_key() -> list[tuple[str, int]]:
    """Prefer business date fields, then _id."""
    return [
        ("ordered_at", -1),
        ("created_at", -1),
        ("order_date", -1),
        ("_id", -1),
    ]


def find_orders_by_customer_email(email: str, limit: int = 3) -> list[dict[str, Any]]:
    """Recent orders for this email, newest first."""
    col = get_orders_collection()
    lim = max(1, min(limit, 25))
    q = _email_or_query(email)
    cur = col.find(q)
    # Try first sort that works (Mongo may not have all fields)
    for field, direction in _sort_key():
        try:
            return list(cur.sort(field, direction).limit(lim))
        except Exception:
            cur = col.find(q)
    return list(col.find(q).limit(lim))


def order_accessible_by_email(doc: dict[str, Any], email: str) -> bool:
    """True if the order is owned by this email (on-document match or in email-scoped list)."""
    e = email.strip().lower()
    on_file = order_email_on_doc(doc)
    if on_file and on_file.lower() == e:
        return True
    if on_file:
        return False
    oid = _str_field(doc, "order_id", "orderId", "id") or str(doc.get("_id", ""))
    for d in find_orders_by_customer_email(e, 50):
        d_oid = _str_field(d, "order_id", "orderId", "id") or str(d.get("_id", ""))
        if d_oid == oid:
            return True
    return False


def find_order_for_email(raw_order_id: str, email: str) -> dict[str, Any] | None:
    """Look up order by id and ensure it belongs to the authenticated email."""
    col = get_orders_collection()
    doc = find_order_document(col, raw_order_id)
    if not doc:
        return None
    if order_accessible_by_email(doc, email):
        return doc
    return None
