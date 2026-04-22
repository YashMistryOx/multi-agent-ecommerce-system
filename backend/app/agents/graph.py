from langchain_core.messages import SystemMessage
from langgraph.graph import END, StateGraph

from app.agents import llm
from app.agents.orders import orders_agent
from app.agents.products import products_agent
from app.agents.prompts import ORDERS_AGENT_PROMPT, PRODUCTS_AGENT_PROMPT
from app.agents.state import AgentState

VALID_ROUTES = {"products", "orders"}


def router_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    content = (
        last_message.content
        if hasattr(last_message, "content")
        else last_message["content"]
    )

    decision = llm.invoke(
        f"""Classify the user query into one of these categories:
    - orders
    - products

    Only return one word: products or orders

    Query: {content}
    """
    )

    route = decision.content.strip().lower()
    if route not in VALID_ROUTES:
        route = "products"

    return {"next": route}


def products_node(state: AgentState) -> dict:
    input_messages = [SystemMessage(content=PRODUCTS_AGENT_PROMPT)] + list(
        state["messages"]
    )
    result = products_agent.invoke({"messages": input_messages})
    return {"messages": result["messages"][len(input_messages):]}


def orders_node(state: AgentState) -> dict:
    input_messages = [SystemMessage(content=ORDERS_AGENT_PROMPT)] + list(
        state["messages"]
    )
    result = orders_agent.invoke({"messages": input_messages})
    return {"messages": result["messages"][len(input_messages):]}


# Build graph
graph = StateGraph(AgentState)

graph.add_node("router", router_node)
graph.add_node("products", products_node)
graph.add_node("orders", orders_node)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "products": "products",
        "orders": "orders",
    },
)

graph.add_edge("products", END)
graph.add_edge("orders", END)

app = graph.compile()
