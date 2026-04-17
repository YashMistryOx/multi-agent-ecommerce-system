from typing import Literal

from langchain_community.vectorstores import Milvus
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.rag.embeddings import get_embeddings
from app.settings import get_settings


def retrieve_rag_context(user_query: str, k: int | None = None) -> str:
    """
    Top-k chunk text only (no LLM). Used to compare orders against written policy in workflows.
    """
    s = get_settings()
    if not s.openai_api_key or not user_query.strip():
        return ""
    embeddings = get_embeddings()
    kk = k if k is not None else s.rag_top_k
    try:
        store = Milvus(
            embedding_function=embeddings,
            collection_name=s.milvus_collection_name,
            connection_args=s.milvus_connection_args,
        )
        docs = store.similarity_search(user_query.strip(), k=kk)
    except Exception:
        return ""
    if not docs:
        return ""
    return "\n\n---\n\n".join(d.page_content for d in docs)


def answer_with_rag(
    user_query: str,
    *,
    mode: Literal["general", "policies"] = "general",
) -> str:
    """
    Retrieve top-k chunks from Milvus and answer with OpenAI chat model.
    `policies` focuses on Omnimarket policies, shipping, returns overview, and company info.
    """
    s = get_settings()
    if not s.openai_api_key:
        return "Chat is not configured: set OPENAI_API_KEY in .env."

    if not user_query.strip():
        return "Please send a non-empty message."

    embeddings = get_embeddings()
    try:
        store = Milvus(
            embedding_function=embeddings,
            collection_name=s.milvus_collection_name,
            connection_args=s.milvus_connection_args,
        )
        docs = store.similarity_search(user_query, k=s.rag_top_k)
    except Exception as e:
        return (
            "Could not query the vector store. Ensure Milvus is running, "
            f"run POST /api/rag/ingest, and check logs. ({e})"
        )

    if not docs:
        return (
            "No relevant passages were found in the knowledge base. "
            "Add content under backend/assets and run ingestion."
        )

    context = "\n\n---\n\n".join(d.page_content for d in docs)
    llm = ChatOpenAI(
        model=s.openai_chat_model,
        temperature=0.2,
        api_key=s.openai_api_key,
    )

    if mode == "policies":
        system = (
            "You are the Omnimarket **policies and information** assistant. "
            "Answer questions about Omnimarket, marketplace rules, shipping, returns "
            "(high level), warranties, FAQs, and company information using **only** the "
            "context below. If the context is insufficient, say so briefly. "
            "Do not invent legal deadlines or fees. Keep replies clear and scannable.\n\n"
            "Context:\n{context}"
        )
    else:
        system = (
            "You are the Omnimarket customer assistant—an ecommerce chatbot for the "
            "Omnimarket omnichannel marketplace. Your job is to help shoppers with "
            "accurate, friendly information about: products and catalog details, "
            "Omnimarket services (shipping, support, account help, and similar), "
            "returns and refunds, warranties, and any other Omnimarket policies or "
            "store information that appears in the context below.\n\n"
            "Ground every answer in the provided context only. Do not invent product "
            "names, prices, SKUs, policies, or deadlines. If the context does not "
            "contain enough information, say so briefly, offer what you can from the "
            "context, and suggest the customer check Omnimarket help or contact support "
            "for specifics. Keep replies concise and easy to scan; use short paragraphs "
            "or bullet points when listing multiple items.\n\n"
            "Context:\n{context}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    msg = chain.invoke({"context": context, "question": user_query})
    return msg.content if hasattr(msg, "content") else str(msg)
