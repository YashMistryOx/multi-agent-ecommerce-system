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
    session: Any = None,
) -> str:
    """
    Run router + subgraphs (policies RAG, orders workflow, returns workflow, clarify).

    Persists `return_workflow`, `orders_workflow`, and `authenticated_email` on `session`
    when provided.
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
    rw: dict[str, Any] = {}
    ow: dict[str, Any] = {}
    session_email: str | None = None
    auth_email: str | None = None
    if session is not None:
        rw = dict(getattr(session, "return_workflow", None) or {})
        ow = dict(getattr(session, "orders_workflow", None) or {})
        session_email = getattr(session, "user_email", None)
        auth_email = getattr(session, "authenticated_email", None)

    graph = get_compiled_graph()
    result = graph.invoke(
        {
            "messages": lc_messages,
            "route": None,
            "request_id": rid,
            "graph_trace": [],
            "return_workflow": rw,
            "orders_workflow": ow,
            "session_user_email": session_email,
            "authenticated_email": auth_email,
        }
    )

    if session is not None:
        session.return_workflow = dict(result.get("return_workflow") or {})
        session.orders_workflow = dict(result.get("orders_workflow") or {})
        ae = result.get("authenticated_email")
        if ae:
            session.authenticated_email = ae
    route = result.get("route")
    trace = list(result.get("graph_trace") or [])
    final_msgs = result.get("messages") or []
    path_str = " → ".join(trace) if trace else "(empty)"
    print(
        f"\n[{rid}] ========== GRAPH EXECUTION COMPLETE ==========\n"
        f"  graph_trace (full list): {trace}\n"
        f"  graph_trace (path):      {path_str}\n"
        f"  steps (node count):      {len(trace)}\n"
        f"  router intent (route):   {route}\n"
        f"==============================================\n",
        flush=True,
    )
    log.info(
        "[%s] lifecycle=graph_invoke_done route=%s graph_trace=%s nodes=%s messages_out=%s",
        rid,
        route,
        trace,
        len(trace),
        len(final_msgs),
    )

    if not final_msgs:
        log.error("[%s] lifecycle=graph_error reason=no_output_messages", rid)
        return "Sorry, something went wrong. Please try again."

    tail = final_msgs[-1]
    if isinstance(tail, AIMessage):
        return str(tail.content) if tail.content else ""
    return str(getattr(tail, "content", tail))
