from typing import Annotated

from langchain_core.messages import AnyMessage
from typing_extensions import NotRequired, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    session_id: str
    session_user_email: str
    next: str
    # scratch-pad for multi-step deterministic workflows
    workflow_data: NotRequired[dict]
