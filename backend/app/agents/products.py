from langchain.agents import create_agent
from app.agents.tools import PRODUCT_TOOL_FUNCTIONS
from app.agents import llm
from app.agents.prompts import PRODUCTS_AGENT_PROMPT

products_agent = create_agent(
    system_prompt=PRODUCTS_AGENT_PROMPT,
    model=llm,
    tools=PRODUCT_TOOL_FUNCTIONS,
)
