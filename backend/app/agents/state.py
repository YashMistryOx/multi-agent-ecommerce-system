from operator import add
from typing import Annotated, Literal, Optional

from langchain_core.messages import AnyMessage
from typing_extensions import NotRequired, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """LangGraph state: conversation + router decision + workflow memory."""

    messages: Annotated[list[AnyMessage], add_messages]
    route: Optional[Literal["policies", "orders", "returns", "clarify"]]
    request_id: NotRequired[str]
    graph_trace: Annotated[list[str], add]
    # Multi-turn workflow state (persisted on ChatSession)
    return_workflow: NotRequired[dict]
    orders_workflow: NotRequired[dict]
    # Email verified in-chat (and optional client-provided session_user_email)
    authenticated_email: NotRequired[str | None]
    session_user_email: NotRequired[str | None]
