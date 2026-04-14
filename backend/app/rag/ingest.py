import threading
from pathlib import Path

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.embeddings import get_embeddings
from app.settings import get_settings

_ingest_lock = threading.Lock()


def _load_text_documents(assets_dir: Path) -> list:
    docs: list = []
    if not assets_dir.is_dir():
        return docs

    for pattern in ("**/*.txt", "**/*.md", "**/*.markdown"):
        loader = DirectoryLoader(
            str(assets_dir),
            glob=pattern,
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            silent_errors=True,
            show_progress=False,
        )
        try:
            docs.extend(loader.load())
        except Exception:
            continue
    return docs


def _load_pdf_documents(assets_dir: Path) -> list:
    docs: list = []
    if not assets_dir.is_dir():
        return docs

    loader = DirectoryLoader(
        str(assets_dir),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        silent_errors=True,
        show_progress=False,
    )
    try:
        docs.extend(loader.load())
    except Exception:
        pass
    return docs


def run_ingestion() -> dict:
    """
    Load text/markdown/PDF from backend/assets, chunk, embed with OpenAI,
    and write vectors to Milvus (replaces existing collection data).
    """
    s = get_settings()
    if not s.openai_api_key:
        raise ValueError("OPENAI_API_KEY is not set")

    assets_dir = Path(s.assets_dir)
    with _ingest_lock:
        documents = _load_text_documents(assets_dir) + _load_pdf_documents(assets_dir)
        if not documents:
            raise ValueError(
                f"No supported files (.txt, .md, .pdf) found under {assets_dir}. "
                "Add files and retry."
            )

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)
        embeddings = get_embeddings()

        Milvus.from_documents(
            chunks,
            embeddings,
            connection_args=s.milvus_connection_args,
            collection_name=s.milvus_collection_name,
            drop_old=True,
        )

        sources = sorted({str(d.metadata.get("source", "")) for d in documents})
        return {
            "status": "ok",
            "files_loaded": len(documents),
            "chunks_indexed": len(chunks),
            "sources": sources,
            "collection": s.milvus_collection_name,
        }
