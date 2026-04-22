from langchain.agents import create_agent
from app.agents.tools import ORDERS_TOOL_FUNCTIONS
from app.agents import llm
from app.agents.prompts import ORDERS_AGENT_PROMPT

orders_agent = create_agent(
    system_prompt=ORDERS_AGENT_PROMPT,
    model=llm,
    tools=ORDERS_TOOL_FUNCTIONS,
)
