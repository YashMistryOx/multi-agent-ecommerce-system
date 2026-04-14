from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.settings import get_settings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    s = get_settings()
    if not s.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set in the environment or .env file")
    return OpenAIEmbeddings(
        model=s.embedding_model,
        api_key=s.openai_api_key,
    )
