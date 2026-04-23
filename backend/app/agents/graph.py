from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.agents import llm
from app.agents.orders_graph import orders_graph
from app.agents.returns_graph import returns_graph
from app.agents.products import products_agent
from app.agents.prompts import PRODUCTS_AGENT_PROMPT
from app.agents.state import AgentState

# products_node still uses a wrapper because products_agent is not a compiled
# LangGraph sub-graph, so there's no interrupt propagation issue there.

VALID_ROUTES = {"products", "orders", "returns"}


def router_node(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    content = (
        last_message.content
        if hasattr(last_message, "content")
        else last_message["content"]
    )

    decision = llm.invoke(
        f"""Classify the user query into exactly one of these categories:
- products  (browse products, search, price, stock, variants, images)
- orders    (view orders, order status, tracking, cancel an order)
- returns   (return an item, return policy, return status, refund questions)

Only return one word.

Query: {content}
"""
    )

    route = decision.content.strip().lower()
    if route not in VALID_ROUTES:
        route = "orders"

    return {"next": route}


def products_node(state: AgentState) -> dict:
    input_messages = [SystemMessage(content=PRODUCTS_AGENT_PROMPT)] + list(
        state["messages"]
    )
    result = products_agent.invoke({"messages": input_messages})
    return {"messages": result["messages"][len(input_messages):]}


# Build main graph
graph = StateGraph(AgentState)

graph.add_node("router",   router_node)
graph.add_node("products", products_node)
# Pass compiled sub-graphs directly so interrupt() propagates to the checkpointer
graph.add_node("orders",   orders_graph)
graph.add_node("returns",  returns_graph)

graph.set_entry_point("router")

graph.add_conditional_edges(
    "router",
    lambda state: state["next"],
    {
        "products": "products",
        "orders":   "orders",
        "returns":  "returns",
    },
)

graph.add_edge("products", END)
graph.add_edge("orders",   END)
graph.add_edge("returns",  END)

app = graph.compile(checkpointer=MemorySaver())
