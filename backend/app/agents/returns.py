from langchain.agents import create_agent

from app.agents import llm
from app.agents.prompts import RETURNS_AGENT_PROMPT
from app.agents.tools import RETURNS_TOOL_FUNCTIONS

returns_agent = create_agent(
    system_prompt=RETURNS_AGENT_PROMPT,
    model=llm,
    tools=RETURNS_TOOL_FUNCTIONS,
)
