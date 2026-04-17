"""Product / purchase issues: email auth → order → reason → rules + RAG policy → pending review + email."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.agents.tools.orders import _format_order_doc
from app.agents.tools.returns import _eligibility_text_from_order
from app.agents.workflows.extract import extract_email, extract_order_id, pick_index_from_message
from app.agents.workflows.order_nl import format_conversation_tail, interpret_order_followup
from app.agents.workflows.return_persistence import (
    persist_return_pending_review,
    send_return_submitted_email,
)
from app.db.orders_query import find_order_for_email, find_orders_by_customer_email
from app.rag.chat import retrieve_rag_context
from app.settings import get_settings

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


class ReturnSlots(BaseModel):
    order_id: str | None = Field(None, description="Omnimarket order id e.g. OM-10001")
    email: str | None = Field(None, description="Customer email if stated")
    reason: str | None = Field(None, description="Return reason if stated")


class ReturnPolicyDecision(BaseModel):
    allowed: bool = Field(description="True if return aligns with policy excerpt for this order")
    rationale: str = Field(description="One or two short sentences")


def _llm_slots(messages: list[AnyMessage], request_id: str) -> ReturnSlots | None:
    s = get_settings()
    if not s.openai_api_key:
        return None
    last = _last_user_text(messages)
    if not last.strip():
        return None
    llm = ChatOpenAI(
        model=s.openai_chat_model,
        temperature=0.0,
        api_key=s.openai_api_key,
    )
    structured = llm.with_structured_output(ReturnSlots)
    try:
        return structured.invoke(
            [
                SystemMessage(
                    content=(
                        "Extract fields from the customer's latest message only. "
                        "Order IDs look like OM- digits. Leave null if absent."
                    )
                ),
                HumanMessage(content=last),
            ]
        )
    except Exception as e:
        log.warning("[%s] return_slots_llm_fail err=%s", request_id, e)
        return None


def _item_blob(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("name") or item.get("title") or ""),
        str(item.get("category") or ""),
        str(item.get("product_type") or item.get("type") or ""),
        str(item.get("sku") or ""),
    ]
    return " ".join(parts).lower()


def non_returnable_from_order(doc: dict[str, Any]) -> tuple[bool, str]:
    """Food / undergarments cannot be returned (policy shortcut)."""
    raw = doc.get("items") or doc.get("line_items") or []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return False, ""

    food_kw = ("food", "beverage", "grocery", "snack", "perishable", "edible")
    under_kw = ("underwear", "undergarment", "lingerie", "intimate", "bra", "brief")

    for item in raw:
        if not isinstance(item, dict):
            continue
        b = _item_blob(item)
        if any(k in b for k in food_kw):
            return True, "Food or perishable items cannot be returned per Omnimarket policy."
        if any(k in b for k in under_kw):
            return True, "Undergarments and intimate apparel cannot be returned per Omnimarket policy."
    return False, ""


def _order_status_allows_return(doc: dict[str, Any]) -> tuple[bool, str]:
    st = (doc.get("status") or doc.get("order_status") or "").lower()
    if "cancel" in st:
        return False, "This order is cancelled; returns usually do not apply."
    if "deliver" in st:
        return True, "Order shows as delivered."
    if any(x in st for x in ("ship", "transit", "processing", "paid", "confirm", "pending")):
        return (
            False,
            "The order is not marked delivered yet. Returns are assessed after you receive the item.",
        )
    return True, _eligibility_text_from_order(doc)


def _policy_decision_llm(
    order_summary: str,
    policy_excerpt: str,
    request_id: str,
) -> ReturnPolicyDecision:
    s = get_settings()
    llm = ChatOpenAI(
        model=s.openai_chat_model,
        temperature=0.1,
        api_key=s.openai_api_key,
    )
    structured = llm.with_structured_output(ReturnPolicyDecision)
    pe = policy_excerpt.strip() or "(No policy passages retrieved from the knowledge base.)"
    return structured.invoke(
        [
            SystemMessage(
                content=(
                    "You are an Omnimarket policy specialist. The order is NOT food or undergarments "
                    "(already checked). Given the order summary and policy excerpt, decide if a return "
                    "request can reasonably be submitted for human review. If policy text is missing, "
                    "lean toward allowing review when the order is delivered and within a typical window. "
                    "Be concise in rationale."
                )
            ),
            HumanMessage(
                content=f"Policy excerpt:\n{pe}\n\nOrder / customer situation:\n{order_summary}"
            ),
        ]
    )


def return_workflow_node(state: AgentState) -> dict:
    rid = _req_id(state)
    wf: dict[str, Any] = dict(state.get("return_workflow") or {})
    messages = state["messages"]
    last = _last_user_text(messages)

    session_email = (state.get("session_user_email") or "").strip().lower() or None
    prior_auth = (state.get("authenticated_email") or "").strip().lower() or None
    reg_em = extract_email(last)
    reg_oid = extract_order_id(last)
    slots = _llm_slots(messages, rid)

    order_id = wf.get("order_id")
    if reg_oid:
        order_id = reg_oid
    # While choosing from a numbered list, ignore LLM-extracted order IDs from older turns.
    picking_from_list = (wf.get("phase") == "pick_order") and bool(wf.get("recent_order_ids"))
    if slots and slots.order_id and not picking_from_list:
        order_id = slots.order_id.strip().upper()

    email = wf.get("email") or prior_auth or session_email or reg_em
    if slots and slots.email:
        email = slots.email.strip().lower()
    if isinstance(email, str):
        email = email.strip().lower() or None

    reason = wf.get("reason")
    if slots and slots.reason and str(slots.reason).strip():
        reason = str(slots.reason).strip()
    if wf.get("phase") == "need_reason" and last.strip() and (not reason or len(str(reason)) < 3):
        reason = last.strip()
    if reason and str(reason).strip():
        wf["reason"] = str(reason).strip()

    phase = wf.get("phase") or "start"

    if phase in ("completed", "rejected", "policy_denied"):
        msg = (
            "We already processed a return request in this chat. "
            "If you need something else, start a new topic."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "return_workflow": wf,
            "authenticated_email": email or wf.get("email"),
            "graph_trace": ["returns_workflow"],
        }

    if not email:
        msg = (
            "To help with a return or product issue, I need to verify who you are. "
            "Please reply with the **email address** you use with Omnimarket (this acts as your sign-in for order lookups)."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "return_workflow": wf,
            "authenticated_email": None,
            "graph_trace": ["returns_workflow"],
        }

    out_auth = email
    wf["email"] = email

    doc: dict[str, Any] | None = None
    recent_ids: list[str] = list(wf.get("recent_order_ids") or [])
    pick = pick_index_from_message(last)
    if recent_ids and pick is not None and 0 <= pick < len(recent_ids):
        oid = recent_ids[pick]
        doc = find_order_for_email(oid, email)
        if doc:
            wf.update({"order_id": oid, "recent_order_ids": [], "phase": "have_order"})
            order_id = oid
        else:
            msg = "I could not match that selection to your email. Try again or send an order ID."
            return {
                "messages": [AIMessage(content=msg)],
                "return_workflow": wf,
                "authenticated_email": out_auth,
                "graph_trace": ["returns_workflow"],
            }

    if doc is None and order_id:
        doc = find_order_for_email(order_id, email)
        if not doc:
            msg = (
                f"I could not find order **{order_id}** for **{email}**. "
                "Check the ID or describe which order you mean."
            )
            wf.update({"order_id": None})
            return {
                "messages": [AIMessage(content=msg)],
                "return_workflow": wf,
                "authenticated_email": out_auth,
                "graph_trace": ["returns_workflow"],
            }
        wf.update({"order_id": order_id, "phase": "have_order"})

    if doc is None:
        nl = interpret_order_followup(
            user_message=last,
            workflow="returns",
            recent_order_ids=recent_ids or None,
            conversation_tail=format_conversation_tail(messages),
            request_id=rid,
        )
        if nl.action == "choose_numbered_list_item" and nl.list_ordinal_1based and recent_ids:
            ix = nl.list_ordinal_1based - 1
            if 0 <= ix < len(recent_ids):
                oid = recent_ids[ix]
                doc = find_order_for_email(oid, email)
                if doc:
                    wf.update({"order_id": oid, "recent_order_ids": [], "phase": "have_order"})
                    order_id = oid
        if doc is None and nl.action == "explicit_order_id" and nl.om_order_id:
            oid = nl.om_order_id.strip().upper()
            doc = find_order_for_email(oid, email)
            if doc:
                wf.update({"order_id": oid, "recent_order_ids": [], "phase": "have_order"})
                order_id = oid
            else:
                msg = (
                    f"I could not find order **{oid}** for **{email}**. "
                    "Check the ID or describe which order you mean."
                )
                wf.update({"order_id": None})
                return {
                    "messages": [AIMessage(content=msg)],
                    "return_workflow": wf,
                    "authenticated_email": out_auth,
                    "graph_trace": ["returns_workflow"],
                }
        if doc is None and nl.action == "want_most_recent_order_only":
            docs = find_orders_by_customer_email(email, 1)
            if docs:
                doc = docs[0]
                oid = str(
                    doc.get("order_id")
                    or doc.get("orderId")
                    or doc.get("id")
                    or doc.get("_id", "")
                )
                wf.update({"order_id": oid, "recent_order_ids": [], "phase": "have_order"})
                order_id = oid
            else:
                msg = (
                    f"We don't see any orders linked to **{email}** in our records, so we can't pick "
                    "a most recent order yet. Please **double-check the email spelling** (it must match "
                    "the one on your Omnimarket account), try another address, or send an **order ID** "
                    "(OM-#####) if you have it."
                )
                return {
                    "messages": [AIMessage(content=msg)],
                    "return_workflow": wf,
                    "authenticated_email": out_auth,
                    "graph_trace": ["returns_workflow"],
                }
        if doc is None and nl.action == "want_recent_orders_list":
            docs = find_orders_by_customer_email(email, 3)
            if not docs:
                msg = f"No orders are linked to **{email}** in our records. Try another email or contact support."
                return {
                    "messages": [AIMessage(content=msg)],
                    "return_workflow": wf,
                    "authenticated_email": out_auth,
                    "graph_trace": ["returns_workflow"],
                }
            lines = []
            ids: list[str] = []
            for i, d in enumerate(docs, start=1):
                oid = str(d.get("order_id") or d.get("orderId") or d.get("id") or d.get("_id", ""))
                ids.append(oid)
                st = str(d.get("status") or d.get("order_status") or "—")
                lines.append(f"{i}. **{oid}** — {st}")
            wf.update({"recent_order_ids": ids, "phase": "pick_order"})
            msg = (
                f"Here are your **3 most recent orders** for **{email}**:\n\n"
                + "\n".join(lines)
                + "\n\nReply with a **number** (1–3), an **order ID** (OM-#####), or describe your choice."
            )
            return {
                "messages": [AIMessage(content=msg)],
                "return_workflow": wf,
                "authenticated_email": out_auth,
                "graph_trace": ["returns_workflow"],
            }

    if doc is None:
        msg = (
            f"Email **{email}** is verified. Which order is the issue about? "
            "Send an **order ID** (OM-#####) or describe what you mean (for example your most recent order, or ask to see recent orders to choose)."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "return_workflow": wf,
            "authenticated_email": out_auth,
            "graph_trace": ["returns_workflow"],
        }

    oid_display = str(
        doc.get("order_id") or doc.get("orderId") or doc.get("id") or order_id or ""
    )

    nr, nr_msg = non_returnable_from_order(doc)
    if nr:
        wf.update({"phase": "rejected"})
        return {
            "messages": [AIMessage(content=f"I'm sorry — {nr_msg}")],
            "return_workflow": wf,
            "authenticated_email": out_auth,
            "graph_trace": ["returns_workflow"],
        }

    ok_status, status_expl = _order_status_allows_return(doc)
    if not ok_status:
        wf.update({"phase": "rejected"})
        return {
            "messages": [AIMessage(content=f"We cannot open a return for this order yet.\n\n{status_expl}")],
            "return_workflow": wf,
            "authenticated_email": out_auth,
            "graph_trace": ["returns_workflow"],
        }

    if not reason or len(reason) < 3:
        wf.update({"phase": "need_reason", "order_id": oid_display})
        msg = (
            f"Order **{oid_display}** — {status_expl}\n\n"
            "Please describe briefly **what is wrong** (damaged, wrong item, changed mind, etc.)."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "return_workflow": wf,
            "authenticated_email": out_auth,
            "graph_trace": ["returns_workflow"],
        }

    order_summary = _format_order_doc(doc)
    delivered_hint = doc.get("delivered_at") or doc.get("delivery_date") or doc.get("eta")
    if delivered_hint:
        order_summary += f"\nDelivery / date hint: {delivered_hint}"

    policy_ctx = retrieve_rag_context(
        "Omnimarket return policy refund eligibility window conditions exclusions",
        k=6,
    )
    try:
        decision = _policy_decision_llm(
            f"{order_summary}\n\nCustomer reason: {reason}",
            policy_ctx,
            rid,
        )
    except Exception as e:
        log.exception("[%s] policy_decision_fail err=%s", rid, e)
        decision = ReturnPolicyDecision(allowed=True, rationale="Automated review fallback.")

    if not decision.allowed:
        wf.update({"phase": "policy_denied"})
        return {
            "messages": [
                AIMessage(
                    content=(
                        "Based on our policies, we cannot automatically submit this return.\n\n"
                        f"{decision.rationale}\n\n"
                        "You can contact Omnimarket support for a manual review."
                    )
                )
            ],
            "return_workflow": wf,
            "authenticated_email": out_auth,
            "graph_trace": ["returns_workflow"],
        }

    ret_id, db_ok = persist_return_pending_review(
        doc,
        customer_email=email or "",
        reason=reason,
        request_id=rid,
    )
    mailed = False
    if email:
        mailed = send_return_submitted_email(
            to_email=email,
            return_id=ret_id,
            order_id=oid_display,
            reason=reason,
            request_id=rid,
        )

    wf.update(
        {
            "phase": "completed",
            "order_id": oid_display,
            "email": email,
            "reason": reason,
            "return_id": ret_id,
        }
    )

    lines = [
        "**Your return request has been created and will be reviewed.**",
        f"- Request ID: **{ret_id}**",
        f"- Order: **{oid_display}**",
        f"- Reason: {reason}",
        "",
        f"_Assessment:_ {decision.rationale}",
    ]
    if db_ok:
        lines.append("\nWe saved your request in our system.")
    if email:
        lines.append(
            f"\nA confirmation was sent to **{email}**."
            if mailed
            else "\nWe could not send email (SMTP not configured); your request may still be recorded."
        )

    return {
        "messages": [AIMessage(content="\n".join(lines))],
        "return_workflow": wf,
        "authenticated_email": out_auth,
        "graph_trace": ["returns_workflow"],
    }
