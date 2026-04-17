"""Order lookup: require authenticated email first; then order ID or last 3 orders by email."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage

from app.agents.state import AgentState
from app.agents.tools.orders import _format_order_doc
from app.agents.workflows.extract import extract_email, extract_order_id, pick_index_from_message
from app.agents.workflows.order_nl import format_conversation_tail, interpret_order_followup
from app.db.orders_query import find_order_for_email, find_orders_by_customer_email

log = logging.getLogger("app.request")


def _req_id(state: AgentState) -> str:
    return state.get("request_id") or "-"


def _last_user_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content or "")
        if getattr(m, "type", None) == "human":
            return str(getattr(m, "content", "") or "")
    return ""


def orders_workflow_node(state: AgentState) -> dict:
    rid = _req_id(state)
    wf: dict[str, Any] = dict(state.get("orders_workflow") or {})
    messages = state["messages"]
    last = _last_user_text(messages)

    session_email = (state.get("session_user_email") or "").strip().lower() or None
    prior_auth = (state.get("authenticated_email") or "").strip().lower() or None
    reg_em = extract_email(last)
    reg_oid = extract_order_id(last)

    email = prior_auth or wf.get("email") or session_email or reg_em
    if isinstance(email, str):
        email = email.strip().lower() or None

    if not email:
        msg = (
            "To protect your privacy, I need the **email address** you use with Omnimarket before "
            "I can show any order details. Please reply with that email."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "orders_workflow": wf,
            "authenticated_email": None,
            "graph_trace": ["orders_workflow"],
        }

    out_auth = email
    wf["email"] = email

    recent_ids: list[str] = list(wf.get("recent_order_ids") or [])
    pick = pick_index_from_message(last)
    if recent_ids and pick is not None and 0 <= pick < len(recent_ids):
        oid = recent_ids[pick]
        doc = find_order_for_email(oid, email)
        if doc:
            body = _format_order_doc(doc)
            msg = f"Here are the details for the order you selected (**{oid}**):\n\n{body}"
            wf.update({"phase": "done", "recent_order_ids": []})
            return {
                "messages": [AIMessage(content=msg)],
                "orders_workflow": wf,
                "authenticated_email": out_auth,
                "graph_trace": ["orders_workflow"],
            }
        msg = f"I could not load order **{oid}** for your email. Try an order ID or ask for your recent orders again."
        return {
            "messages": [AIMessage(content=msg)],
            "orders_workflow": wf,
            "authenticated_email": out_auth,
            "graph_trace": ["orders_workflow"],
        }

    if reg_oid:
        doc = find_order_for_email(reg_oid, email)
        if not doc:
            msg = (
                f"I could not find order **{reg_oid}** linked to **{email}**. "
                "Check the ID or describe which order you mean."
            )
            return {
                "messages": [AIMessage(content=msg)],
                "orders_workflow": wf,
                "authenticated_email": out_auth,
                "graph_trace": ["orders_workflow"],
            }
        body = _format_order_doc(doc)
        msg = f"Here are the details for **{reg_oid}**:\n\n{body}"
        wf.update({"phase": "done", "recent_order_ids": []})
        log.info("[%s] orders_workflow order_found order_id=%s", rid, reg_oid)
        return {
            "messages": [AIMessage(content=msg)],
            "orders_workflow": wf,
            "authenticated_email": out_auth,
            "graph_trace": ["orders_workflow"],
        }

    nl = interpret_order_followup(
        user_message=last,
        workflow="orders",
        recent_order_ids=recent_ids or None,
        conversation_tail=format_conversation_tail(messages),
        request_id=rid,
    )
    doc: dict[str, Any] | None = None
    if nl.action == "choose_numbered_list_item" and nl.list_ordinal_1based and recent_ids:
        ix = nl.list_ordinal_1based - 1
        if 0 <= ix < len(recent_ids):
            oid = recent_ids[ix]
            doc = find_order_for_email(oid, email)
            if doc:
                body = _format_order_doc(doc)
                msg = f"Here are the details for the order you selected (**{oid}**):\n\n{body}"
                wf.update({"phase": "done", "recent_order_ids": []})
                return {
                    "messages": [AIMessage(content=msg)],
                    "orders_workflow": wf,
                    "authenticated_email": out_auth,
                    "graph_trace": ["orders_workflow"],
                }
    if doc is None and nl.action == "explicit_order_id" and nl.om_order_id:
        oid = nl.om_order_id.strip().upper()
        doc = find_order_for_email(oid, email)
        if doc:
            body = _format_order_doc(doc)
            msg = f"Here are the details for **{oid}**:\n\n{body}"
            wf.update({"phase": "done", "recent_order_ids": []})
            log.info("[%s] orders_workflow order_found order_id=%s", rid, oid)
            return {
                "messages": [AIMessage(content=msg)],
                "orders_workflow": wf,
                "authenticated_email": out_auth,
                "graph_trace": ["orders_workflow"],
            }
        msg = (
            f"I could not find order **{oid}** linked to **{email}**. "
            "Check the ID or describe which order you mean."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "orders_workflow": wf,
            "authenticated_email": out_auth,
            "graph_trace": ["orders_workflow"],
        }
    if nl.action == "want_most_recent_order_only":
        docs = find_orders_by_customer_email(email, 1)
        if docs:
            d0 = docs[0]
            oid = str(
                d0.get("order_id")
                or d0.get("orderId")
                or d0.get("id")
                or d0.get("_id", "")
            )
            body = _format_order_doc(d0)
            msg = (
                f"Here are the details for your **most recent order** (**{oid}**):\n\n{body}"
            )
            wf.update({"phase": "done", "recent_order_ids": []})
            return {
                "messages": [AIMessage(content=msg)],
                "orders_workflow": wf,
                "authenticated_email": out_auth,
                "graph_trace": ["orders_workflow"],
            }
        msg = (
            f"We don't see any orders linked to **{email}** in our records. "
            "Double-check the email spelling or try another address you may have used with Omnimarket."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "orders_workflow": wf,
            "authenticated_email": out_auth,
            "graph_trace": ["orders_workflow"],
        }
    if nl.action == "want_recent_orders_list":
        docs = find_orders_by_customer_email(email, 3)
        if not docs:
            msg = (
                f"I don't see any orders linked to **{email}** in our records yet. "
                "If you used a different email, share that one, or contact support."
            )
            wf.update({"phase": "no_orders", "recent_order_ids": []})
            return {
                "messages": [AIMessage(content=msg)],
                "orders_workflow": wf,
                "authenticated_email": out_auth,
                "graph_trace": ["orders_workflow"],
            }
        lines = []
        ids: list[str] = []
        for i, d in enumerate(docs, start=1):
            oid = (
                str(d.get("order_id") or d.get("orderId") or d.get("id") or d.get("_id", ""))
            )
            ids.append(oid)
            st = str(d.get("status") or d.get("order_status") or "—")
            lines.append(f"{i}. **{oid}** — {st}")
        wf.update({"phase": "listed", "recent_order_ids": ids, "awaiting_choice": True})
        msg = (
            f"Here are your **3 most recent orders** for **{email}**:\n\n"
            + "\n".join(lines)
            + "\n\nReply with a **number** (1–3), an **order ID** (OM-#####), or describe your choice."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "orders_workflow": wf,
            "authenticated_email": out_auth,
            "graph_trace": ["orders_workflow"],
        }

    msg = (
        f"Email **{email}** is on file for this chat. Send your **order ID** (OM-#####), "
        "or describe what you need (your most recent order, or ask to see recent orders to choose)."
    )
    return {
        "messages": [AIMessage(content=msg)],
        "orders_workflow": wf,
        "authenticated_email": out_auth,
        "graph_trace": ["orders_workflow"],
    }
