"""LangGraph: router → orders | returns | qna | clarify."""

import logging
from functools import lru_cache
from typing import Literal

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.agents.tools.orders import ORDERS_TOOLS
from app.agents.tools.returns import RETURNS_TOOLS
from app.rag.chat import answer_with_rag
from app.settings import get_settings

log = logging.getLogger("app.request")


def _req_id(state: AgentState) -> str:
    return state.get("request_id") or "-"


ROUTER_SYSTEM = """You route Omnimarket customer messages to exactly one intent:

- orders: Order status, tracking, shipping, delivery, "where is my package", order IDs, ETA, carrier.
- returns: Returns, refunds, exchanges, damaged/wrong item, how to send back, refund timing.
- qna: Product catalog, services, warranties, general policies, FAQs, company info, anything best answered from the knowledge base.
- clarify: The message is empty, off-topic, or too ambiguous to route (e.g. only "help" with no topic).

Use the conversation context. Prefer the most specific intent."""


ORDERS_PROMPT = (
    "You are the Omnimarket **Orders** specialist. Help with order lookup, status, and "
    "shipping details. Use tools to fetch data. If the customer has no order ID, use "
    "list_recent_orders or ask for their order ID (format OM-#####). "
    "When the user is asking about **returns or refunds**, still focus this step on "
    "finding and confirming the right order (ID, status, items, ship state). "
    "Do not give full return-policy detail here—that is handled in the next step. "
    "Be concise and friendly."
)

RETURNS_PROMPT = (
    "You are the Omnimarket **Returns & refunds** specialist (second step in return flows). "
    "Order lookup already ran in the previous assistant turn—use the conversation above for "
    "order IDs and status before asking again. Use only return tools: policy summary, "
    "eligibility checks, starting a return, and return-case status. "
    "Be concise, empathetic, and clear about next steps."
)


class RouteIntent(BaseModel):
    intent: Literal["orders", "returns", "qna", "clarify"] = Field(
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


@lru_cache
def _orders_react():
    s = get_settings()
    llm = ChatOpenAI(
        model=s.openai_chat_model,
        temperature=0.2,
        api_key=s.openai_api_key,
    )
    return create_agent(llm, ORDERS_TOOLS, system_prompt=ORDERS_PROMPT)


@lru_cache
def _returns_react():
    s = get_settings()
    llm = ChatOpenAI(
        model=s.openai_chat_model,
        temperature=0.2,
        api_key=s.openai_api_key,
    )
    return create_agent(llm, RETURNS_TOOLS, system_prompt=RETURNS_PROMPT)


def _router_node(state: AgentState) -> dict:
    rid = _req_id(state)
    log.info("[%s] step=router_enter", rid)
    s = get_settings()
    if not s.openai_api_key:
        log.warning("[%s] step=router_exit intent=qna reason=no_api_key", rid)
        return {"route": "qna", "graph_trace": ["router"]}
    llm = ChatOpenAI(
        model=s.openai_chat_model,
        temperature=0.0,
        api_key=s.openai_api_key,
    )
    structured = llm.with_structured_output(RouteIntent)
    try:
        out = structured.invoke(
            [SystemMessage(content=ROUTER_SYSTEM), *_tail(state["messages"])]
        )
        intent = out.intent
    except Exception as e:
        log.warning("[%s] step=router_fallback intent=qna err=%s", rid, e)
        intent = "qna"
    log.info("[%s] step=router_exit intent=%s", rid, intent)
    return {"route": intent, "graph_trace": ["router"]}


def _orders_node(state: AgentState) -> dict:
    rid = _req_id(state)
    pipeline = "return_pipeline" if state.get("route") == "returns" else "orders_only"
    log.info("[%s] step=orders_agent_enter pipeline=%s msgs_in=%s", rid, pipeline, len(state["messages"]))
    prior = list(state["messages"])
    result = _orders_react().invoke({"messages": prior})
    new_msgs = result["messages"][len(prior) :]
    log.info("[%s] step=orders_agent_exit new_msgs=%s", rid, len(new_msgs))
    return {"messages": new_msgs, "graph_trace": ["orders"]}


def _returns_node(state: AgentState) -> dict:
    rid = _req_id(state)
    log.info("[%s] step=returns_agent_enter pipeline=return_pipeline msgs_in=%s", rid, len(state["messages"]))
    prior = list(state["messages"])
    result = _returns_react().invoke({"messages": prior})
    new_msgs = result["messages"][len(prior) :]
    log.info("[%s] step=returns_agent_exit new_msgs=%s", rid, len(new_msgs))
    return {"messages": new_msgs, "graph_trace": ["returns"]}


def _qna_node(state: AgentState) -> dict:
    rid = _req_id(state)
    log.info("[%s] step=qna_rag_enter", rid)
    text = answer_with_rag(_last_human_text(state["messages"]))
    log.info("[%s] step=qna_rag_exit reply_len=%s", rid, len(text or ""))
    return {"messages": [AIMessage(content=text)], "graph_trace": ["qna"]}


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
                    "Ask one short, friendly clarifying question to learn whether they need "
                    "help with an order, a return/refund, or general Omnimarket information."
                )
            ),
            *_tail(state["messages"], 8),
        ]
    )
    content = msg.content if hasattr(msg, "content") else str(msg)
    log.info("[%s] step=clarify_exit reply_len=%s", rid, len(str(content)))
    return {"messages": [AIMessage(content=str(content))], "graph_trace": ["clarify"]}


def _router_to_first_step(state: AgentState) -> str:
    """
    Map high-level intent to first graph node.
    `returns` and `orders` both enter the **orders** node first; `returns` continues to **returns** after.
    """
    r = state.get("route")
    if r in ("orders", "returns"):
        return "orders"
    if r == "qna":
        return "qna"
    if r == "clarify":
        return "clarify"
    return "qna"


def _after_orders(state: AgentState) -> str:
    """Return/refund intent: orders agent → returns agent. Pure order intent: stop here."""
    if state.get("route") == "returns":
        return "returns"
    return "done"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("router", _router_node)
    g.add_node("orders", _orders_node)
    g.add_node("returns", _returns_node)
    g.add_node("qna", _qna_node)
    g.add_node("clarify", _clarify_node)

    g.add_edge(START, "router")
    g.add_conditional_edges(
        "router",
        _router_to_first_step,
        {
            "orders": "orders",
            "qna": "qna",
            "clarify": "clarify",
        },
    )
    g.add_conditional_edges(
        "orders",
        _after_orders,
        {
            "returns": "returns",
            "done": END,
        },
    )
    g.add_edge("returns", END)
    g.add_edge("qna", END)
    g.add_edge("clarify", END)
    return g.compile()


@lru_cache
def get_compiled_graph():
    return build_graph()


def clear_graph_cache() -> None:
    """Clear cached graphs (e.g. after tests or model changes)."""
    get_compiled_graph.cache_clear()
    _orders_react.cache_clear()
    _returns_react.cache_clear()
