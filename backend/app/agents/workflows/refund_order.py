"""
Refund Order Workflow
=====================
Deterministic flow for issuing a full refund in Shopify when the order total
is below the configured auto-approve ceiling (see Settings.refund_auto_approve_max_amount).

  collect_email
      ↓
  fetch_orders          (paid / partially refundable orders only)
      ↓
  select_order          ──(none)──→ END
      ↓
  evaluate_amount       ──(total >= max)──→ END (manual / support)
      ↓
  ask_confirmation      ──(user says no)──→ END
      ↓
  execute_refund
      ↓
  END
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.agents import llm
from app.agents.state import AgentState
from app.agents.tools.orders import (
    create_full_refund_for_order,
    get_orders_by_email,
)
from app.settings import get_settings


def _data(state: AgentState) -> dict:
    return state.get("workflow_data") or {}


def _set_data(state: AgentState, **kwargs) -> dict:
    return {"workflow_data": {**_data(state), **kwargs}}


def _parse_order_total(order: dict) -> float:
    raw = order.get("current_total_price") or order.get("total_price") or "0"
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def rf_collect_email(state: AgentState) -> dict:
    """Use session email or ask the user for it."""
    print("NODE: rf_collect_email")
    email = state.get("session_user_email", "").strip()
    if email:
        return _set_data(state, refund_email=email)

    email = interrupt(
        "Please provide your email address so I can look up your order for a refund."
    )
    email = email.strip()
    return {**_set_data(state, refund_email=email), "session_user_email": email}


def rf_fetch_orders(state: AgentState) -> dict:
    """Load orders for this customer that may still be refundable."""
    print("NODE: rf_fetch_orders")
    email = _data(state)["refund_email"]
    all_orders = get_orders_by_email.invoke({"email": email})

    eligible: list[dict] = []
    for o in all_orders:
        fs = (o.get("financial_status") or "").lower()
        if fs in ("voided", "refunded", "pending"):
            continue
        if fs not in ("authorized", "partially_paid", "paid", "partially_refunded"):
            continue
        if _parse_order_total(o) <= 0:
            continue
        eligible.append(o)

    return _set_data(state, refund_orders=eligible)


def rf_select_order(state: AgentState) -> dict:
    """Pick the order to refund (auto if one, otherwise interrupt)."""
    print("NODE: rf_select_order")
    orders = _data(state).get("refund_orders", [])

    if not orders:
        return {
            "workflow_data": {},
            "messages": [AIMessage(content=(
                "I couldn't find any orders on your account that are eligible for a refund "
                "(for example, already fully refunded, not paid yet, or voided). "
                "If you think this is wrong, please contact support."
            ))],
        }

    if len(orders) == 1:
        return _set_data(state, refund_selected_order=orders[0])

    summary = "\n".join(
        f"{i + 1}. Order #{o['order_number']}  |  "
        f"placed {o['created_at'][:10]}  |  "
        f"total: {o.get('current_total_price') or o.get('total_price')}  |  "
        f"status: {o.get('financial_status', 'unknown')}"
        for i, o in enumerate(orders)
    )
    choice = interrupt(
        f"Which order would you like a refund for?\n\n{summary}\n\nReply with the number."
    )
    try:
        idx = int(choice.strip()) - 1
        selected = orders[max(0, min(idx, len(orders) - 1))]
    except (ValueError, IndexError):
        selected = orders[0]

    return _set_data(state, refund_selected_order=selected)


def rf_evaluate_amount(state: AgentState) -> dict:
    """Only auto-process refunds when order total is strictly below the configured max."""
    print("NODE: rf_evaluate_amount")
    order = _data(state)["refund_selected_order"]
    order_num = order.get("order_number", order.get("id"))
    total = _parse_order_total(order)
    max_amount = get_settings().refund_auto_approve_max_amount
    auto_ok = total < max_amount

    if not auto_ok:
        msg = llm.invoke(
            f"You are a helpful customer support assistant. "
            f"The customer's order #{order_num} has a total of {total} (shop currency). "
            f"Our automated refund assistant can only process refunds for orders "
            f"strictly under {max_amount} in the same currency. "
            f"Explain politely that this order must be reviewed by the team, and suggest "
            f"they contact support. Keep it to 3–4 short lines, friendly tone."
        ).content
        return {
            "workflow_data": {},
            "messages": [AIMessage(content=msg)],
        }

    return _set_data(
        state,
        refund_auto_eligible=True,
        refund_order_total=total,
    )


def rf_ask_confirmation(state: AgentState) -> dict:
    """Confirm before calling Shopify to refund."""
    print("NODE: rf_ask_confirmation")
    data = _data(state)
    if not data.get("refund_auto_eligible"):
        return _set_data(state, refund_confirmed=False)

    order = data["refund_selected_order"]
    order_num = order.get("order_number", order.get("id"))
    total = data.get("refund_order_total", _parse_order_total(order))

    confirmation_prompt = llm.invoke(
        f"You are a helpful customer support assistant. "
        f"Write a short, casual chat message (NOT an email). "
        f"The customer asked for a refund for order #{order_num}. "
        f"The order total is {total} (shop currency). "
        f"Explain that you can process a full refund to their original payment method. "
        f"Ask them to reply 'yes' to confirm the refund or 'no' to cancel. "
        f"Keep it under 4 lines."
    ).content

    confirm = interrupt(confirmation_prompt)
    confirmed = confirm.strip().lower() in ("yes", "y")

    if not confirmed:
        abort = llm.invoke(
            f"You are a helpful assistant. The customer declined the refund for "
            f"order #{order_num}. Acknowledge briefly and offer help if they need anything else."
        ).content
        return {
            "workflow_data": {},
            "messages": [AIMessage(content=abort)],
        }

    return _set_data(state, refund_confirmed=True)


def rf_execute(state: AgentState) -> dict:
    """Create the refund in Shopify."""
    print("NODE: rf_execute")
    order = _data(state)["refund_selected_order"]
    order_num = order.get("order_number", order.get("id"))
    oid = int(order["id"])

    result = create_full_refund_for_order(oid)

    if result.get("error"):
        return {
            "workflow_data": {},
            "messages": [AIMessage(content=(
                f"I wasn't able to complete the refund for Order #{order_num}: "
                f"{result['error']}\nPlease try again or contact support."
            ))],
        }

    ok = llm.invoke(
        f"You are a helpful customer support assistant. "
        f"The refund for order #{order_num} was submitted successfully in Shopify. "
        f"Refund record id (if any): {result.get('refund_id', 'N/A')}. "
        f"Tell the customer the refund was issued, that it may take a few business days "
        f"to appear on their statement depending on their bank, and keep the tone warm and brief."
    ).content

    return {
        "workflow_data": {},
        "messages": [AIMessage(content=ok)],
    }


def _route_after_select(state: AgentState) -> str:
    if not _data(state).get("refund_selected_order"):
        return "end"
    return "evaluate_amount"


def _route_after_evaluate(state: AgentState) -> str:
    if not _data(state).get("refund_auto_eligible"):
        return "end"
    return "ask_confirmation"


def _route_after_confirm(state: AgentState) -> str:
    if _data(state).get("refund_confirmed"):
        return "execute"
    return "end"


_g = StateGraph(AgentState)

_g.add_node("collect_email",      rf_collect_email)
_g.add_node("fetch_orders",       rf_fetch_orders)
_g.add_node("select_order",       rf_select_order)
_g.add_node("evaluate_amount",    rf_evaluate_amount)
_g.add_node("ask_confirmation", rf_ask_confirmation)
_g.add_node("execute_refund",     rf_execute)

_g.set_entry_point("collect_email")

_g.add_edge("collect_email", "fetch_orders")
_g.add_edge("fetch_orders", "select_order")

_g.add_conditional_edges(
    "select_order",
    _route_after_select,
    {"evaluate_amount": "evaluate_amount", "end": END},
)

_g.add_conditional_edges(
    "evaluate_amount",
    _route_after_evaluate,
    {"ask_confirmation": "ask_confirmation", "end": END},
)

_g.add_conditional_edges(
    "ask_confirmation",
    _route_after_confirm,
    {"execute": "execute_refund", "end": END},
)

_g.add_edge("execute_refund", END)

refund_order_graph = _g.compile()