"""
Orders sub-graph
================
The top-level router sends everything orders-related here.
An internal supervisor then decides:

  user message
       ↓
  [orders_supervisor]
       ├── tool_call      → orders_agent  (free-form, read-only lookups)
       ├── cancel_order   → cancel_order_graph (deterministic multi-step)
       └── process_refund → refund_order_graph (refund money in Shopify when rules allow)
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from app.agents import llm
from app.agents.orders import orders_agent
from app.agents.prompts import ORDERS_AGENT_PROMPT
from app.agents.state import AgentState
from app.agents.workflows.cancel_order import cancel_order_graph
from app.agents.workflows.refund_order import refund_order_graph

VALID_ACTIONS = {"tool_call", "cancel_order", "process_refund"}

SUPERVISOR_PROMPT = """You are an orders assistant supervisor.
Classify the user message into exactly one of these actions:

- tool_call       → read-only request: view orders, check status, tracking, order details
- cancel_order    → user wants to cancel an existing order before it ships / stop the order
- process_refund  → user wants their money back: refund, reimbursement, charge reversal,
                    "refund my payment", "I want a refund" (not the same as a merchandise return flow)

Only return one word: tool_call, cancel_order, or process_refund

Message: {message}
"""


def orders_supervisor_node(state: AgentState) -> dict:
    print(f"NODE: orders_supervisor_node")
    last = state["messages"][-1]
    content = last.content if hasattr(last, "content") else last["content"]

    decision = llm.invoke(SUPERVISOR_PROMPT.format(message=content))
    action = decision.content.strip().lower()

    if action not in VALID_ACTIONS:
        action = "tool_call"

    return {"next": action}


def orders_agent_node(state: AgentState) -> dict:
    print(f"NODE: orders_agent_node")
    input_messages = [SystemMessage(content=ORDERS_AGENT_PROMPT)] + list(
        state["messages"]
    )
    result = orders_agent.invoke({"messages": input_messages})
    return {"messages": result["messages"][len(input_messages):]}


# Build the orders sub-graph
_g = StateGraph(AgentState)

_g.add_node("supervisor",    orders_supervisor_node)
_g.add_node("orders_agent",  orders_agent_node)
# Pass the compiled graph directly so interrupt() propagates to the parent checkpointer
_g.add_node("cancel_order",  cancel_order_graph)
_g.add_node("process_refund", refund_order_graph)

_g.set_entry_point("supervisor")

_g.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "tool_call":      "orders_agent",
        "cancel_order":   "cancel_order",
        "process_refund": "process_refund",
    },
)

_g.add_edge("orders_agent", END)
_g.add_edge("cancel_order", END)
_g.add_edge("process_refund", END)

orders_graph = _g.compile()
