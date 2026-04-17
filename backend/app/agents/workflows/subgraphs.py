"""Compiled subgraphs (single entry → specialist → END) for the parent graph."""

from langgraph.graph import END, START, StateGraph

from app.agents.state import AgentState
from app.agents.workflows.orders_workflow import orders_workflow_node
from app.agents.workflows.return_workflow import return_workflow_node


def build_orders_subgraph():
    g = StateGraph(AgentState)
    g.add_node("orders", orders_workflow_node)
    g.add_edge(START, "orders")
    g.add_edge("orders", END)
    return g.compile()


def build_returns_workflow_subgraph():
    g = StateGraph(AgentState)
    g.add_node("returns_workflow", return_workflow_node)
    g.add_edge(START, "returns_workflow")
    g.add_edge("returns_workflow", END)
    return g.compile()
