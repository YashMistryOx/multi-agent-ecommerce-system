from operator import add
from typing import Annotated, Literal, Optional

from langchain_core.messages import AnyMessage
from typing_extensions import NotRequired, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """LangGraph state: conversation + router decision."""

    messages: Annotated[list[AnyMessage], add_messages]
    route: Optional[Literal["orders", "returns", "qna", "clarify"]]
    request_id: NotRequired[str]
    # Node names visited in order (router → specialist [→ returns for return pipeline])
    graph_trace: Annotated[list[str], add]
