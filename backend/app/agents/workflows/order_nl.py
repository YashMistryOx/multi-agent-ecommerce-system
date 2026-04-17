"""LLM-based interpretation of how users refer to orders — no hardcoded phrase lists."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from app.agents.workflows.extract import message_is_email_only
from app.settings import get_settings

log = logging.getLogger("app.request")


def _user_means_pick_from_list_not_autopick(user_message: str) -> bool:
    """Structural cue: 'one of … order(s)/purchase' implies a list to choose from, not auto-newest."""
    t = (user_message or "").strip().lower()
    if "one of" not in t:
        return False
    return any(k in t for k in ("order", "orders", "purchase", "purchases"))


class OrderFollowupIntent(BaseModel):
    """Structured interpretation of the user's latest message about orders."""

    action: Literal[
        "explicit_order_id",
        "choose_numbered_list_item",
        "want_most_recent_order_only",
        "want_recent_orders_list",
        "still_unclear",
    ] = Field(
        description=(
            "explicit_order_id: user gave OM-##### or equivalent. "
            "choose_numbered_list_item: user selects from a numbered list the assistant already showed. "
            "want_most_recent_order_only: user wants the assistant to auto-pick the single newest order "
            "without showing a list (no 'one of', no 'which', no 'show me my orders'). "
            "want_recent_orders_list: user wants recent orders listed so they can choose "
            "(includes 'one of my orders', 'one of my last orders', show/list/recent orders to pick). "
            "still_unclear: cannot determine."
        )
    )
    om_order_id: str | None = Field(
        None,
        description="Normalized OM-##### if explicit_order_id, else null.",
    )
    list_ordinal_1based: int | None = Field(
        None,
        ge=1,
        le=10,
        description="1-based position when choosing from a numbered list the assistant displayed.",
    )


def interpret_order_followup(
    *,
    user_message: str,
    workflow: Literal["returns", "orders"],
    recent_order_ids: list[str] | None,
    conversation_tail: str,
    request_id: str,
) -> OrderFollowupIntent:
    """
    Single structured LLM call. Uses conversation_tail (recent turns) for references like
    "the second one" or "that order".
    """
    s = get_settings()
    if not s.openai_api_key or not (user_message or "").strip():
        return OrderFollowupIntent(action="still_unclear")

    # Identity-only turns must not inherit "last order" / list intent from earlier messages in the tail.
    if message_is_email_only(user_message):
        log.info("[%s] order_nl skip email_only_turn", request_id)
        return OrderFollowupIntent(action="still_unclear")

    ids = recent_order_ids or []
    list_desc = ", ".join(f"{i + 1}={oid}" for i, oid in enumerate(ids)) if ids else "(none)"

    wf = (
        "returns / refund or product issue with a purchase"
        if workflow == "returns"
        else "order status / tracking / lookup"
    )

    system = f"""You interpret the customer's **latest user message** for Omnimarket {wf}.

## Rules (read carefully)

1. **Latest message drives autopick and list requests.** Do **not** infer these from older user lines
   except for short follow-ups (see below).

2. **List vs auto-pick (critical):**
   - Use **want_recent_orders_list** when the user wants **multiple recent orders shown** so they
     can **choose** which one applies. This includes: "show my recent orders", "list my orders",
     "which order", phrasing with **"one of"** (e.g. "one of my last orders", "one of my orders") —
     because "one of" implies picking from several. Also "a few recent orders", "my last few orders".
   - Use **want_most_recent_order_only** only when they want you to **automatically use the single
     newest order** with **no list**: e.g. "my last order", "my most recent order", "the latest
     order", "just the most recent one" — **without** "one of" / "which" / "show" / "list".

3. **Short follow-ups** (e.g. "yes", "the second one", "2") may use the conversation tail for
   **choose_numbered_list_item** or to align with the assistant's last question.

4. **Numbered list** (position → order id): {list_desc}
   Use `choose_numbered_list_item` only when a list is shown **or** the user clearly refers back to it.

5. **explicit_order_id**: they state OM-##### in the latest message. Set om_order_id uppercase.

6. **still_unclear**: you cannot map to the above without guessing.

Do not invent order IDs."""

    llm = ChatOpenAI(
        model=s.openai_chat_model,
        temperature=0.0,
        api_key=s.openai_api_key,
    )
    structured = llm.with_structured_output(OrderFollowupIntent)
    human = (
        f"Recent conversation:\n{conversation_tail}\n\n---\n\nLatest user message:\n{user_message}"
    )
    try:
        out = structured.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        if out.action == "want_most_recent_order_only" and _user_means_pick_from_list_not_autopick(
            user_message
        ):
            out = OrderFollowupIntent(action="want_recent_orders_list")
            log.info("[%s] order_nl override list_pick phrasing", request_id)
        log.info("[%s] order_nl action=%s", request_id, out.action)
        return out
    except Exception as e:
        log.warning("[%s] order_nl_fail err=%s", request_id, e)
        return OrderFollowupIntent(action="still_unclear")


def format_conversation_tail(messages: list[AnyMessage], max_turns: int = 6) -> str:
    """Compact last human/assistant strings for NL context."""
    lines: list[str] = []
    for m in messages[-max_turns:]:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            c = str(m.content or "")
            if c:
                lines.append(f"Assistant: {c[:500]}")
    return "\n".join(lines) if lines else "(no prior context)"
