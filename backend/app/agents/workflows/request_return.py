"""
Request Return Workflow
=======================
Deterministic multi-step flow:

  collect_email
      ↓
  fetch_orders          (filters to fulfilled, returnable orders only)
      ↓
  select_order          ──(no returnable orders)──→ END
      ↓
  fetch_items           (fetches fulfillment line items via GraphQL)
      ↓
  select_item           ──(interrupt if multiple items)
      ↓
  collect_reason        (fetch return reasons + interrupt to ask user)
      ↓
  ask_confirmation      ──(user says no)──→ END
      ↓
  execute_return        (submit returnRequest GraphQL mutation)
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
from app.agents.tools.returns import (
    get_returnable_orders_by_email,
    get_return_policy,
    get_return_reasons,
    get_fulfillment_line_items,
    submit_return_request,
)


def _match_reason(customer_description: str, reasons: list) -> dict:
    """Use the LLM to silently pick the best matching reason from the live Shopify list."""
    reasons_text = "\n".join(f"- {r['name']}" for r in reasons)
    decision = llm.invoke(
        f"A customer wants to return an item. Their description: \"{customer_description}\"\n\n"
        f"Match this to exactly one of the following return reasons:\n"
        f"{reasons_text}\n\n"
        f"Reply with ONLY the exact reason name from the list above, nothing else."
    )
    matched_name = decision.content.strip()
    for r in reasons:
        if r["name"].lower() == matched_name.lower():
            return r
    # fallback to last entry (typically "Other") if no match
    return reasons[-1]

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

def rr_collect_email(state: AgentState) -> dict:
    """Use session email or ask the user for it."""
    print("NODE: rr_collect_email")
    email = state.get("session_user_email", "").strip()
    if email:
        return _set_data(state, email=email)

    email = interrupt("Please provide your email address so I can look up your orders.")
    return _set_data(state, email=email.strip())


def rr_fetch_orders(state: AgentState) -> dict:
    """Fetch fulfilled orders that are eligible for a return."""
    print("NODE: rr_fetch_orders")
    email = _data(state)["email"]
    orders = get_returnable_orders_by_email.invoke({"email": email})
    return _set_data(state, returnable_orders=orders)


def rr_select_order(state: AgentState) -> dict:
    """Auto-select if one order, otherwise ask the user to pick."""
    print("NODE: rr_select_order")
    orders = _data(state).get("returnable_orders", [])

    if not orders:
        return {
            **_set_data(state, selected_order=None),
            "messages": [AIMessage(content=(
                "I couldn't find any fulfilled orders eligible for a return on your account. "
                "Only orders that have been delivered can be returned. "
                "Please contact support if you believe this is a mistake."
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
    choice = interrupt(
        f"You have {len(orders)} fulfilled orders. Which one would you like to return?\n\n"
        f"{summary}\n\nReply with the number."
    )

    try:
        idx = int(choice.strip()) - 1
        selected = orders[max(0, min(idx, len(orders) - 1))]
    except (ValueError, IndexError):
        selected = orders[0]

    return _set_data(state, selected_order=selected)


def rr_fetch_items(state: AgentState) -> dict:
    """Fetch the fulfillment line items (with GraphQL IDs) for the selected order."""
    print("NODE: rr_fetch_items")
    order = _data(state)["selected_order"]
    items = get_fulfillment_line_items(order["id"])
    return _set_data(state, fulfillment_items=items)


def rr_select_item(state: AgentState) -> dict:
    """Auto-select if one item, otherwise ask the user which item to return."""
    print("NODE: rr_select_item")
    items = _data(state).get("fulfillment_items", [])

    if not items:
        return {
            **_set_data(state, selected_item=None),
            "messages": [AIMessage(content=(
                "I couldn't retrieve the items for this order. "
                "Please contact support to initiate your return."
            ))],
        }

    if len(items) == 1:
        return _set_data(state, selected_item=items[0], return_quantity=1)

    summary = "\n".join(
        f"{i + 1}. {it['title']}"
        f"{' - ' + it['variant_title'] if it.get('variant_title') else ''}  "
        f"(qty: {it['quantity']})"
        for i, it in enumerate(items)
    )
    choice = interrupt(
        f"Which item would you like to return?\n\n{summary}\n\nReply with the number."
    )

    try:
        idx = int(choice.strip()) - 1
        selected = items[max(0, min(idx, len(items) - 1))]
    except (ValueError, IndexError):
        selected = items[0]

    return _set_data(state, selected_item=selected, return_quantity=1)


def rr_collect_reason(state: AgentState) -> dict:
    """Ask the customer to describe their reason in plain words, then auto-classify it."""
    print("NODE: rr_collect_reason")

    description = interrupt(
        "Could you briefly tell us why you'd like to return this item? "
        "(e.g. 'it's too big', 'the colour is different from the photo', 'it arrived damaged')"
    )
    description = description.strip()

    # Fetch live reasons from Shopify, then let the LLM pick the best match silently.
    reasons = get_return_reasons.invoke({})
    if not reasons:
        return _set_data(
            state,
            selected_reason_id=None,
            selected_reason_name="Other",
            reason_note=description,
        )

    matched_reason = _match_reason(description, reasons)

    return _set_data(
        state,
        selected_reason_id=matched_reason["id"],
        selected_reason_name=matched_reason["name"],
        reason_note=description,
    )


def rr_ask_confirmation(state: AgentState) -> dict:
    """Show return summary and ask for explicit user confirmation."""
    print("NODE: rr_ask_confirmation")
    data = _data(state)
    order = data["selected_order"]
    item = data["selected_item"]
    order_num = order.get("order_number", order.get("id"))
    policy = get_return_policy.invoke({})

    confirmation_prompt = llm.invoke(
        f"You are a helpful customer support assistant. "
        f"Ask the customer to confirm their return request in a clear, friendly tone. "
        f"Include the following details naturally in your message:\n"
        f"- Order number: #{order_num}\n"
        f"- Item to return: {item.get('title', 'Unknown')}"
        f"{' - ' + item['variant_title'] if item.get('variant_title') else ''}\n"
        f"- Reason: {data.get('selected_reason_name', 'Not specified')}\n"
        f"{('- Customer note: ' + data['reason_note']) if data.get('reason_note') else ''}\n"
        f"{'- Store return policy: ' + policy if policy else ''}\n\n"
        f"End by asking them to reply with 'yes' to submit or 'no' to cancel."
    ).content

    confirm = interrupt(confirmation_prompt)
    confirmed = confirm.strip().lower() in ("yes", "y")

    if not confirmed:
        abort_msg = llm.invoke(
            f"You are a helpful customer support assistant. "
            f"Inform the customer in a friendly tone that their return request for "
            f"Order #{order_num} has been cancelled. Keep it brief."
        ).content
        return {
            **_set_data(state, confirmed=False),
            "messages": [AIMessage(content=abort_msg)],
        }

    return _set_data(state, confirmed=True)


def rr_execute(state: AgentState) -> dict:
    """Submit the return request via Shopify GraphQL and report the result."""
    print("NODE: rr_execute")
    data = _data(state)
    order = data["selected_order"]
    item = data["selected_item"]
    order_num = order.get("order_number", order.get("id"))

    result = submit_return_request(
        order_id=order["id"],
        fulfillment_line_item_id=item["fulfillment_line_item_id"],
        quantity=data.get("return_quantity", 1),
        reason_id=data.get("selected_reason_id", ""),
        customer_note=data.get("reason_note", ""),
    )

    if "error" in result:
        return {
            **_set_data(state, workflow_data={}),
            "messages": [AIMessage(content=(
                f"Something went wrong while submitting your return for Order #{order_num}: "
                f"{result['error']}\nPlease try again or contact support."
            ))],
        }

    success_msg = llm.invoke(
        f"You are a helpful customer support assistant. "
        f"Inform the customer that their return request has been successfully submitted. "
        f"Return reference: {result.get('name', 'N/A')}. "
        f"Order: #{order_num}. "
        f"Item: {item.get('title', 'Unknown')}. "
        f"Status: {result.get('status', 'REQUESTED')}. "
        f"Mention that the merchant will review and approve the return request. Keep it friendly and concise."
    ).content

    return {
        **_set_data(state, workflow_data={}),
        "messages": [AIMessage(content=success_msg)],
    }


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _route_after_select_order(state: AgentState) -> str:
    if not _data(state).get("selected_order"):
        return "end"
    return "fetch_items"


def _route_after_select_item(state: AgentState) -> str:
    if not _data(state).get("selected_item"):
        return "end"
    return "collect_reason"


def _route_after_confirm(state: AgentState) -> str:
    if _data(state).get("confirmed"):
        return "execute"
    return "end"


# ---------------------------------------------------------------------------
# Build sub-graph
# ---------------------------------------------------------------------------

_g = StateGraph(AgentState)

_g.add_node("collect_email",   rr_collect_email)
_g.add_node("fetch_orders",    rr_fetch_orders)
_g.add_node("select_order",    rr_select_order)
_g.add_node("fetch_items",     rr_fetch_items)
_g.add_node("select_item",     rr_select_item)
_g.add_node("collect_reason",  rr_collect_reason)
_g.add_node("ask_confirmation", rr_ask_confirmation)
_g.add_node("execute_return",  rr_execute)

_g.set_entry_point("collect_email")

_g.add_edge("collect_email", "fetch_orders")
_g.add_edge("fetch_orders",  "select_order")

_g.add_conditional_edges(
    "select_order",
    _route_after_select_order,
    {"fetch_items": "fetch_items", "end": END},
)

_g.add_edge("fetch_items", "select_item")

_g.add_conditional_edges(
    "select_item",
    _route_after_select_item,
    {"collect_reason": "collect_reason", "end": END},
)

_g.add_edge("collect_reason", "ask_confirmation")

_g.add_conditional_edges(
    "ask_confirmation",
    _route_after_confirm,
    {"execute": "execute_return", "end": END},
)

_g.add_edge("execute_return", END)

request_return_graph = _g.compile()
