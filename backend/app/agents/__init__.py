from langchain_openai import ChatOpenAI
from app.settings import get_settings

s = get_settings()
llm = ChatOpenAI(
    model=s.openai_chat_model,
    temperature=0.2,
    api_key=s.openai_api_key,
)

__all__ = ["llm"]