"""LangGraph: router → policies (RAG) | orders workflow | returns workflow | clarify."""

import logging
from functools import lru_cache
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.agents.workflows.subgraphs import (
    build_orders_subgraph,
    build_returns_workflow_subgraph,
)
from app.rag.chat import answer_with_rag
from app.settings import get_settings

log = logging.getLogger("app.request")


def _req_id(state: AgentState) -> str:
    return state.get("request_id") or "-"


ROUTER_SYSTEM = """You route Omnimarket customer messages to exactly one intent:

- policies: Questions about Omnimarket, the marketplace, shipping policies, return policy overview, FAQs, warranties, company info — answered from the knowledge base (RAG).
- orders: Order status, tracking, "where is my package", looking up **their** orders, purchase history, order IDs, ETA — requires email + order data from our database.
- returns: Product problems, wrong/damaged item, doesn't want the purchase, starting a return or refund for **their** order — return workflow with eligibility and review.
- clarify: Empty, off-topic, or too ambiguous.

Use conversation context. Prefer the most specific intent."""


def _router_session_context(state: AgentState) -> str:
    """Summarize in-progress workflows so the router can continue vs switch topics without hardcoded sticky rules."""
    rw = state.get("return_workflow") or {}
    ow = state.get("orders_workflow") or {}
    lines: list[str] = []
    ph_r = rw.get("phase") or ""
    ph_o = ow.get("phase") or ""
    if rw.get("email") and ph_r not in ("completed", "rejected", "policy_denied"):
        pending = bool(rw.get("recent_order_ids"))
        lines.append(
            f"- Returns flow active: phase={ph_r!r}. "
            f"The user may be answering a follow-up (reason, order choice, or list item). "
            f"Assistant showed a numbered order list: {pending}."
        )
    if ow.get("email") and ph_o not in ("done", "no_orders"):
        pending = bool(ow.get("recent_order_ids"))
        lines.append(
            f"- Order lookup flow active: phase={ph_o!r}. "
            f"The user may be continuing order lookup. Assistant showed a numbered order list: {pending}."
        )
    if not lines:
        return ""
    return (
        "\n\n## Session context (routing hints only)\n"
        + "\n".join(lines)
        + "\n\nIf the latest message continues that workflow (including short replies like a number, "
        "an order id, or natural language about *their* order), choose the matching intent "
        "(**returns** or **orders**). Choose **policies** only if they clearly moved to a general "
        "knowledge question. Choose **clarify** only if the message is empty or unusable."
    )


class RouteIntent(BaseModel):
    intent: Literal["policies", "orders", "returns", "clarify"] = Field(
        description="Single routing intent for the user's latest need"
    )


def _tail(msgs: list[AnyMessage], n: int = 12) -> list[AnyMessage]:
    return list(msgs[-n:]) if len(msgs) > n else list(msgs)


def _last_human_text(messages: list[AnyMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return str(m.content) if m.content else ""
        if getattr(m, "type", None) == "human":
            return str(getattr(m, "content", "") or "")
    return ""


def _router_node(state: AgentState) -> dict:
    rid = _req_id(state)
    log.info("[%s] step=router_enter", rid)
    s = get_settings()
    if not s.openai_api_key:
        log.warning("[%s] step=router_exit intent=policies reason=no_api_key", rid)
        return {
            "route": "policies",
            "graph_trace": ["router"],
            "return_workflow": {},
            "orders_workflow": {},
        }

    router_prompt = ROUTER_SYSTEM + _router_session_context(state)
    llm = ChatOpenAI(
        model=s.openai_chat_model,
        temperature=0.0,
        api_key=s.openai_api_key,
    )
    structured = llm.with_structured_output(RouteIntent)
    try:
        out = structured.invoke(
            [SystemMessage(content=router_prompt), *_tail(state["messages"])]
        )
        intent = out.intent
    except Exception as e:
        log.warning("[%s] step=router_fallback intent=policies err=%s", rid, e)
        intent = "policies"
    log.info("[%s] step=router_exit intent=%s", rid, intent)
    updates: dict = {"route": intent, "graph_trace": ["router"]}
    if intent != "returns":
        updates["return_workflow"] = {}
    if intent != "orders":
        updates["orders_workflow"] = {}
    return updates


def _policies_node(state: AgentState) -> dict:
    rid = _req_id(state)
    log.info("[%s] step=policies_rag_enter", rid)
    text = answer_with_rag(_last_human_text(state["messages"]), mode="policies")
    log.info("[%s] step=policies_rag_exit reply_len=%s", rid, len(text or ""))
    return {"messages": [AIMessage(content=text)], "graph_trace": ["policies"]}


def _clarify_node(state: AgentState) -> dict:
    rid = _req_id(state)
    log.info("[%s] step=clarify_enter", rid)
    s = get_settings()
    llm = ChatOpenAI(
        model=s.openai_chat_model,
        temperature=0.3,
        api_key=s.openai_api_key,
    )
    msg = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are Omnimarket support. The user's message was unclear. "
                    "Ask one short, friendly clarifying question: policies / their orders / "
                    "or a return or product issue."
                )
            ),
            *_tail(state["messages"], 8),
        ]
    )
    content = msg.content if hasattr(msg, "content") else str(msg)
    log.info("[%s] step=clarify_exit reply_len=%s", rid, len(str(content)))
    return {"messages": [AIMessage(content=str(content))], "graph_trace": ["clarify"]}


def _router_to_subgraph(state: AgentState) -> str:
    r = state.get("route")
    if r == "policies":
        return "policies"
    if r == "orders":
        return "orders"
    if r == "returns":
        return "returns"
    if r == "clarify":
        return "clarify"
    return "policies"


def build_graph():
    orders_sg = build_orders_subgraph()
    returns_sg = build_returns_workflow_subgraph()

    g = StateGraph(AgentState)
    g.add_node("router", _router_node)
    g.add_node("policies", _policies_node)
    g.add_node("orders", orders_sg)
    g.add_node("returns", returns_sg)
    g.add_node("clarify", _clarify_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _router_to_subgraph,
        {
            "policies": "policies",
            "orders": "orders",
            "returns": "returns",
            "clarify": "clarify",
        },
    )
    g.add_edge("policies", END)
    g.add_edge("orders", END)
    g.add_edge("returns", END)
    g.add_edge("clarify", END)
    return g.compile()


@lru_cache
def get_compiled_graph():
    return build_graph()


def clear_graph_cache() -> None:
    """Clear cached graphs (e.g. after tests or model changes)."""
    get_compiled_graph.cache_clear()
