"""MongoDB writes for return workflow + optional SMTP confirmation email."""

from __future__ import annotations

import logging
import smtplib
import uuid
from datetime import datetime, timezone
from email.message import EmailMessage
from typing import Any

from pymongo.errors import PyMongoError

from app.db.mongo import get_orders_collection, get_returns_collection
from app.db.orders_query import order_email_on_doc
from app.settings import get_settings

log = logging.getLogger("app.request")


def _str_field(doc: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        v = doc.get(k)
        if v is not None and v != "":
            return str(v).strip()
    return default


def order_customer_email(doc: dict[str, Any]) -> str | None:
    """Email on file for the order (alias for shared helper)."""
    return order_email_on_doc(doc)


def generate_return_id() -> str:
    return f"RR-{uuid.uuid4().hex[:8].upper()}"


def persist_return_pending_review(
    order_doc: dict[str, Any],
    *,
    customer_email: str,
    reason: str,
    request_id: str,
) -> tuple[str, bool]:
    """
    Insert return record as pending review; mark order with return-pending state.
    """
    return_id = generate_return_id()
    oid = _str_field(order_doc, "order_id", "orderId", "id") or str(order_doc.get("_id", ""))
    now = datetime.now(timezone.utc)

    returns_col = get_returns_collection()
    orders_col = get_orders_collection()

    ret_doc = {
        "return_id": return_id,
        "order_id": oid,
        "customer_email": customer_email,
        "reason": reason[:2000],
        "status": "pending_review",
        "created_at": now,
    }

    try:
        returns_col.insert_one(ret_doc)
    except PyMongoError as e:
        log.exception("[%s] return_persist insert_failed err=%s", request_id, e)
        return return_id, False

    try:
        orders_col.update_one(
            {"_id": order_doc["_id"]},
            {
                "$set": {
                    "status": "return_pending_review",
                    "return_id": return_id,
                    "return_reason": reason[:2000],
                    "return_requested_at": now,
                    "return_customer_email": customer_email,
                }
            },
        )
    except PyMongoError as e:
        log.exception("[%s] return_persist order_update_failed err=%s", request_id, e)
        return return_id, False

    return return_id, True


def persist_return_initiated(
    order_doc: dict[str, Any],
    *,
    customer_email: str,
    reason: str,
    request_id: str,
) -> tuple[str, bool]:
    """
    Insert return record, set order status to return_initiated.
    Returns (return_id, db_ok).
    """
    return_id = generate_return_id()
    oid = _str_field(order_doc, "order_id", "orderId", "id") or str(order_doc.get("_id", ""))
    now = datetime.now(timezone.utc)

    returns_col = get_returns_collection()
    orders_col = get_orders_collection()

    ret_doc = {
        "return_id": return_id,
        "order_id": oid,
        "customer_email": customer_email,
        "reason": reason[:2000],
        "status": "return_initiated",
        "created_at": now,
    }

    try:
        returns_col.insert_one(ret_doc)
    except PyMongoError as e:
        log.exception("[%s] return_persist insert_failed err=%s", request_id, e)
        return return_id, False

    try:
        orders_col.update_one(
            {"_id": order_doc["_id"]},
            {
                "$set": {
                    "status": "return_initiated",
                    "return_id": return_id,
                    "return_reason": reason[:2000],
                    "return_initiated_at": now,
                    "return_customer_email": customer_email,
                }
            },
        )
    except PyMongoError as e:
        log.exception("[%s] return_persist order_update_failed err=%s", request_id, e)
        return return_id, False

    return return_id, True


def send_return_submitted_email(
    *,
    to_email: str,
    return_id: str,
    order_id: str,
    reason: str,
    request_id: str,
) -> bool:
    """Notify customer that a return request was submitted for review."""
    s = get_settings()
    if not s.smtp_host or not s.smtp_user or not s.smtp_password:
        log.warning("[%s] smtp_skip reason=not_configured", request_id)
        return False

    body = (
        f"Hello,\n\n"
        f"We received your return request. It will be **reviewed** by our team shortly.\n\n"
        f"Return request ID: {return_id}\n"
        f"Order: {order_id}\n"
        f"Reason: {reason}\n\n"
        f"We will follow up by email if we need anything else.\n"
        f"— Omnimarket Support"
    )
    msg = EmailMessage()
    msg["Subject"] = f"Return request received — {return_id}"
    msg["From"] = s.smtp_from_email or s.smtp_user
    msg["To"] = to_email
    msg.set_content(body)

    try:
        if s.smtp_use_tls:
            with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=30) as server:
                server.starttls()
                server.login(s.smtp_user, s.smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, timeout=30) as server:
                server.login(s.smtp_user, s.smtp_password)
                server.send_message(msg)
        log.info("[%s] smtp_sent to=%s return_id=%s", request_id, to_email, return_id)
        return True
    except Exception as e:
        log.exception("[%s] smtp_fail err=%s", request_id, e)
        return False


def send_return_confirmation_email(
    *,
    to_email: str,
    return_id: str,
    order_id: str,
    reason: str,
    request_id: str,
) -> bool:
    """Send confirmation via Gmail-compatible SMTP. Returns False if skipped or failed."""
    s = get_settings()
    if not s.smtp_host or not s.smtp_user or not s.smtp_password:
        log.warning("[%s] smtp_skip reason=not_configured", request_id)
        return False

    body = (
        f"Your return request is confirmed.\n\n"
        f"Return ID: {return_id}\n"
        f"Order: {order_id}\n"
        f"Reason: {reason}\n\n"
        f"We will follow up with shipping instructions if required.\n"
        f"— Omnimarket Support"
    )
    msg = EmailMessage()
    msg["Subject"] = f"Return confirmed — {return_id}"
    msg["From"] = s.smtp_from_email or s.smtp_user
    msg["To"] = to_email
    msg.set_content(body)

    try:
        if s.smtp_use_tls:
            with smtplib.SMTP(s.smtp_host, s.smtp_port, timeout=30) as server:
                server.starttls()
                server.login(s.smtp_user, s.smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(s.smtp_host, s.smtp_port, timeout=30) as server:
                server.login(s.smtp_user, s.smtp_password)
                server.send_message(msg)
        log.info("[%s] smtp_sent to=%s return_id=%s", request_id, to_email, return_id)
        return True
    except Exception as e:
        log.exception("[%s] smtp_fail err=%s", request_id, e)
        return False
