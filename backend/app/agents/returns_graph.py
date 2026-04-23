"""
Returns sub-graph
=================
The top-level router sends everything returns-related here.
An internal supervisor then decides:

  user message
       ↓
  [returns_supervisor]
       ├── tool_call       → returns_agent  (free-form, read-only QnA)
       └── request_return  → request_return_graph (deterministic multi-step)
"""

from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from app.agents import llm
from app.agents.returns import returns_agent
from app.agents.prompts import RETURNS_AGENT_PROMPT
from app.agents.state import AgentState
from app.agents.workflows.request_return import request_return_graph

VALID_ACTIONS = {"tool_call", "request_return"}

SUPERVISOR_PROMPT = """You are a returns assistant supervisor.
Classify the user message into exactly one of these actions:

- tool_call      → read-only request: return policy, return reasons, return eligibility, existing return status
- request_return → user wants to initiate / submit a return for an order

Only return one word: tool_call or request_return

Message: {message}
"""


def returns_supervisor_node(state: AgentState) -> dict:
    print("NODE: returns_supervisor_node")
    last = state["messages"][-1]
    content = last.content if hasattr(last, "content") else last["content"]

    decision = llm.invoke(SUPERVISOR_PROMPT.format(message=content))
    action = decision.content.strip().lower()

    if action not in VALID_ACTIONS:
        action = "tool_call"

    return {"next": action}


def returns_agent_node(state: AgentState) -> dict:
    print("NODE: returns_agent_node")
    input_messages = [SystemMessage(content=RETURNS_AGENT_PROMPT)] + list(
        state["messages"]
    )
    result = returns_agent.invoke({"messages": input_messages})
    return {"messages": result["messages"][len(input_messages):]}


# Build the returns sub-graph
_g = StateGraph(AgentState)

_g.add_node("supervisor",      returns_supervisor_node)
_g.add_node("returns_agent",   returns_agent_node)
# Pass compiled graph directly so interrupt() propagates to the parent checkpointer
_g.add_node("request_return",  request_return_graph)

_g.set_entry_point("supervisor")

_g.add_conditional_edges(
    "supervisor",
    lambda state: state["next"],
    {
        "tool_call":      "returns_agent",
        "request_return": "request_return",
    },
)

_g.add_edge("returns_agent",  END)
_g.add_edge("request_return", END)

returns_graph = _g.compile()
