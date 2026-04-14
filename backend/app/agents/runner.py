"""Invoke the multi-agent LangGraph from session message history."""

import logging
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agents.graph import get_compiled_graph
from app.settings import get_settings

log = logging.getLogger("app.request")


def _session_to_lc_messages(session_messages: list[dict[str, Any]]) -> list:
    out = []
    for row in session_messages:
        role = row.get("role")
        content = row.get("content") or ""
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def run_multi_agent(
    session_messages: list[dict[str, Any]],
    request_id: str | None = None,
) -> str:
    """
    Run router + specialist (orders / returns / RAG QnA / clarify) and return
    the assistant reply text.
    """
    rid = request_id or uuid.uuid4().hex[:12]
    s = get_settings()
    if not s.openai_api_key:
        log.warning("[%s] lifecycle=graph_skip reason=no_openai_key", rid)
        return "Chat is not configured: set OPENAI_API_KEY in .env."

    lc_messages = _session_to_lc_messages(session_messages)
    if not lc_messages:
        log.warning("[%s] lifecycle=graph_skip reason=empty_messages", rid)
        return "Please send a non-empty message."

    last = lc_messages[-1]
    if not isinstance(last, HumanMessage) or not str(last.content).strip():
        log.warning("[%s] lifecycle=graph_skip reason=empty_last_turn", rid)
        return "Please send a non-empty message."

    log.info(
        "[%s] lifecycle=graph_invoke_start turns=%s",
        rid,
        len(lc_messages),
    )
    graph = get_compiled_graph()
    result = graph.invoke(
        {"messages": lc_messages, "route": None, "request_id": rid}
    )
    route = result.get("route")
    final_msgs = result.get("messages") or []
    log.info(
        "[%s] lifecycle=graph_invoke_done route=%s messages_out=%s",
        rid,
        route,
        len(final_msgs),
    )

    if not final_msgs:
        log.error("[%s] lifecycle=graph_error reason=no_output_messages", rid)
        return "Sorry, something went wrong. Please try again."

    tail = final_msgs[-1]
    if isinstance(tail, AIMessage):
        return str(tail.content) if tail.content else ""
    return str(getattr(tail, "content", tail))
