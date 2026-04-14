from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BACKEND_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    milvus_uri: str = "http://127.0.0.1:19530"
    milvus_token: str | None = None
    milvus_collection_name: str = "mas_rag_kb"
    openai_chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    rag_top_k: int = 4
    assets_dir: Path = BACKEND_ROOT / "assets"

    # MongoDB (orders tools). Override MONGODB_URI in .env for non-local setups.
    mongodb_uri: str = (
        "mongodb://admin:password@127.0.0.1:27017/?authSource=admin"
    )
    mongodb_database: str = "omnimarket"
    mongodb_orders_collection: str = "orders"

    @property
    def milvus_connection_args(self) -> dict:
        args: dict = {"uri": self.milvus_uri}
        if self.milvus_token:
            args["token"] = self.milvus_token
        return args


@lru_cache
def get_settings() -> Settings:
    return Settings()
