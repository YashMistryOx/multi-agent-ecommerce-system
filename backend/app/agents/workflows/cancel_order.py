"""
Cancel Order Workflow
=====================
Deterministic multi-step flow:

  collect_email
      ↓
  fetch_orders          (filters to cancellable orders only)
      ↓
  select_order          ──(no eligible orders)──→ END
      ↓
  check_eligibility     (order status + cancellation policy)
      ↓
  ask_confirmation      ──(user says no / ineligible)──→ END
      ↓
  execute_cancel
      ↓
  END

Each node that needs user input calls interrupt(), which pauses the graph
and resumes when main.py calls app.invoke(Command(resume=<reply>), config).
"""

from langchain_core.messages import AIMessage
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.agents import llm
from app.agents.state import AgentState
from app.agents.tools.orders import (
    cancel_order,
    get_cancellation_policy,
    get_orders_by_email,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(state: AgentState) -> dict:
    return state.get("workflow_data") or {}


def _set_data(state: AgentState, **kwargs) -> dict:
    return {"workflow_data": {**_data(state), **kwargs}}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def co_collect_email(state: AgentState) -> dict:
    """Use session email or ask the user for it."""
    print(f"NODE: co_collect_email")
    email = state.get("session_user_email", "").strip()
    if email:
        return _set_data(state, email=email)

    email = interrupt("Please provide your email address so I can look up your orders.")
    return _set_data(state, email=email.strip())


def co_fetch_orders(state: AgentState) -> dict:
    """Fetch orders for the collected email and filter to cancellable ones."""
    email = _data(state)["email"]
    print(f"NODE: co_fetch_orders")
    all_orders = get_orders_by_email.invoke({"email": email})

    # cancellable = not already cancelled, not fully fulfilled (shipped)
    eligible = [
        o for o in all_orders
        if o.get("cancelled_at") is None
        and o.get("fulfillment_status") not in ("fulfilled",)
        and o.get("financial_status") not in ("refunded", "voided")
    ]
    return _set_data(state, open_orders=eligible)


def co_select_order(state: AgentState) -> dict:
    """Auto-select if one order, otherwise ask the user to pick."""
    orders = _data(state).get("open_orders", [])
    print(f"NODE: co_select_order")
    if not orders:
        return {
            **_set_data(state, selected_order=None),
            "messages": [AIMessage(content=(
                "I couldn't find any open orders that are eligible for cancellation "
                "on your account. If you believe this is a mistake, please contact support."
            ))],
        }

    if len(orders) == 1:
        return _set_data(state, selected_order=orders[0])

    summary = "\n".join(
        f"{i + 1}. Order #{o['order_number']}  |  "
        f"placed {o['created_at'][:10]}  |  "
        f"status: {o.get('financial_status', 'unknown')}"
        for i, o in enumerate(orders)
    )
    print(f"NODE: co_select_order")
    choice = interrupt(
        f"You have {len(orders)} open orders. Which one would you like to cancel?\n\n"
        f"{summary}\n\nReply with the number."
    )

    try:
        idx = int(choice.strip()) - 1
        selected = orders[max(0, min(idx, len(orders) - 1))]
    except (ValueError, IndexError):
        selected = orders[0]

    return _set_data(state, selected_order=selected)


def co_check_eligibility(state: AgentState) -> dict:
    """Check order status against Shopify cancellation policy."""
    order = _data(state)["selected_order"]
    print(f"NODE: co_check_eligibility")
    policy = get_cancellation_policy.invoke({})

    fulfillment = order.get("fulfillment_status") or "unfulfilled"
    financial = order.get("financial_status", "")

    eligible = fulfillment not in ("fulfilled",) and financial not in ("refunded", "voided")

    reason = ""
    if not eligible:
        if fulfillment == "fulfilled":
            reason = (
                "This order has already been shipped and cannot be cancelled. "
                "You may be able to request a return once it arrives."
            )
        else:
            reason = f"This order (financial status: {financial}) is no longer eligible for cancellation."

    return _set_data(
        state,
        eligible=eligible,
        ineligible_reason=reason,
        cancellation_policy=policy,
    )


def co_ask_confirmation(state: AgentState) -> dict:
    """Show eligibility result and ask for explicit user confirmation."""
    data = _data(state)
    order = data["selected_order"]
    order_num = order.get("order_number", order.get("id"))

    if not data.get("eligible"):
        ineligible_msg = llm.invoke(
            f"You are a helpful customer support assistant. "
            f"Inform the customer in a polite, empathetic tone that their order cannot be cancelled. "
            f"Reason: {data.get('ineligible_reason', 'the order is no longer eligible for cancellation')}. "
            f"Order number: #{order_num}. "
            f"Keep the message concise and suggest contacting support if they need further help."
        ).content
        return {
            **_set_data(state, confirmed=False),
            "messages": [AIMessage(content=ineligible_msg)],
        }

    items = order.get("line_items", [])
    item_summary = ", ".join(
        f"{i.get('title')} x{i.get('quantity')} (${i.get('price')})"
        for i in items
    ) or "no items found"
    policy = data.get("cancellation_policy", "")

    confirmation_prompt = llm.invoke(
        f"You are a helpful customer support assistant. Form a chat message for below request to the customer: "
        f"Ask the customer to confirm cancellation of their order in a clear, friendly tone. "
        f"Include the following order details naturally in your message:\n"
        f"- Order number: #{order_num}\n"
        f"- Placed on: {order.get('created_at', '')[:10]}\n"
        f"- Items: {item_summary}\n"
        f"{'- Store cancellation policy: ' + policy[:300] if policy else ''}\n\n"
        f"End by asking them to reply with 'yes' to confirm or 'no' to cancel."
    ).content

    confirm = interrupt(confirmation_prompt)

    confirmed = confirm.strip().lower() in ("yes", "y")

    if not confirmed:
        abort_msg = llm.invoke(
            f"You are a helpful customer support assistant. "
            f"Inform the customer in a friendly tone that the cancellation for "
            f"Order #{order_num} has been aborted and their order remains active. "
            f"Keep it brief."
        ).content
        return {
            **_set_data(state, confirmed=False),
            "messages": [AIMessage(content=abort_msg)],
        }

    return _set_data(state, confirmed=True)


def co_execute(state: AgentState) -> dict:
    """Call the Shopify cancel endpoint and report the result."""
    order = _data(state)["selected_order"]
    order_num = order.get("order_number", order.get("id"))
    print(f"NODE: co_execute")
    result = cancel_order.invoke({"order_id": order["id"], "reason": "customer"})

    if "error" in result:
        return {
            **_set_data(state, workflow_data={}),
            "messages": [AIMessage(content=(
                f"Something went wrong while cancelling Order #{order_num}: "
                f"{result['error']}\nPlease try again or contact support."
            ))],
        }

    return {
        **_set_data(state, workflow_data={}),
        "messages": [AIMessage(content=(
            f"Order #{order_num} has been successfully cancelled. "
            f"If a payment was made, a refund will be processed according to the store policy."
        ))],
    }


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_after_select(state: AgentState) -> str:
    """Skip the rest of the flow if no eligible orders were found."""
    if not _data(state).get("selected_order"):
        return "end"
    return "check_eligibility"


def _route_after_confirm(state: AgentState) -> str:
    """Proceed to execution only if the user confirmed and the order is eligible."""
    if _data(state).get("confirmed"):
        return "execute"
    return "end"


# ---------------------------------------------------------------------------
# Build sub-graph
# ---------------------------------------------------------------------------

_g = StateGraph(AgentState)

_g.add_node("collect_email",     co_collect_email)
_g.add_node("fetch_orders",      co_fetch_orders)
_g.add_node("select_order",      co_select_order)
_g.add_node("check_eligibility", co_check_eligibility)
_g.add_node("ask_confirmation",  co_ask_confirmation)
_g.add_node("execute_cancel",    co_execute)

_g.set_entry_point("collect_email")

_g.add_edge("collect_email", "fetch_orders")
_g.add_edge("fetch_orders", "select_order")

_g.add_conditional_edges(
    "select_order",
    _route_after_select,
    {"check_eligibility": "check_eligibility", "end": END},
)

_g.add_edge("check_eligibility", "ask_confirmation")

_g.add_conditional_edges(
    "ask_confirmation",
    _route_after_confirm,
    {"execute": "execute_cancel", "end": END},
)

_g.add_edge("execute_cancel", END)

cancel_order_graph = _g.compile()
