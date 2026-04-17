import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import parse_qs

import socketio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.logging_config import setup_logging

setup_logging()

from app.agents.runner import run_multi_agent
from app.rag.ingest import run_ingestion

log = logging.getLogger("app.request")


def _parse_uuid(raw: Optional[str]) -> Optional[str]:
    if not raw:
        return None
    try:
        return str(uuid.UUID(raw))
    except (ValueError, TypeError):
        return None


@dataclass
class ChatSession:
    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    # Optional profile (e.g. from client); pre-fills email for workflows.
    user_email: str | None = None
    # Email verified in chat (shared by orders + returns workflows).
    authenticated_email: str | None = None
    return_workflow: dict[str, Any] = field(default_factory=dict)
    orders_workflow: dict[str, Any] = field(default_factory=dict)


class SessionManager:
    def __init__(self) -> None:
        self._sessions: dict[str, ChatSession] = {}

    def get_or_create(self, requested_id: Optional[str]) -> ChatSession:
        if requested_id and requested_id in self._sessions:
            return self._sessions[requested_id]
        sid = requested_id or str(uuid.uuid4())
        if sid not in self._sessions:
            self._sessions[sid] = ChatSession(session_id=sid)
        return self._sessions[sid]


sessions = SessionManager()
# Socket.IO connection id -> chat session
session_by_socket: dict[str, ChatSession] = {}

sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*",
    always_connect=True,
)

fastapi_app = FastAPI(title="MAS Chat Socket.IO API")

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@fastapi_app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@fastapi_app.post("/api/rag/ingest")
async def rag_ingest() -> dict:
    """
    Load text/markdown/PDF from backend/assets, chunk, embed with OpenAI
    (text-embedding-3-small), and index into Milvus (replaces collection).
    """
    rid = uuid.uuid4().hex[:12]
    log.info("[%s] lifecycle=ingest_start path=POST /api/rag/ingest", rid)
    try:
        result = await asyncio.to_thread(run_ingestion)
        log.info("[%s] lifecycle=ingest_done status=ok chunks=%s", rid, result.get("chunks_indexed"))
        return result
    except ValueError as e:
        log.warning("[%s] lifecycle=ingest_fail status=400 err=%s", rid, e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        log.exception("[%s] lifecycle=ingest_fail status=500", rid)
        raise HTTPException(status_code=500, detail=str(e)) from e


@sio.event
async def connect(sid: str, environ: dict) -> None:
    query = parse_qs(environ.get("QUERY_STRING", ""))
    raw_list = query.get("session_id")
    raw = raw_list[0] if raw_list else None
    parsed = _parse_uuid(raw)
    chat = sessions.get_or_create(parsed)
    email_list = query.get("user_email")
    if email_list and email_list[0].strip():
        chat.user_email = email_list[0].strip()
    session_by_socket[sid] = chat
    log.info(
        "[%s] lifecycle=socket_connect sid=%s session_id=%s resumed=%s",
        uuid.uuid4().hex[:12],
        sid[:8] + "…",
        chat.session_id,
        bool(parsed),
    )
    await sio.emit(
        "session",
        {"type": "session", "session_id": chat.session_id},
        room=sid,
    )


@sio.event
async def disconnect(sid: str) -> None:
    session_by_socket.pop(sid, None)
    log.info("[%s] lifecycle=socket_disconnect sid=%s", uuid.uuid4().hex[:12], sid[:8] + "…")


@sio.on("user_message")
async def user_message(sid: str, data: dict[str, Any]) -> None:
    session = session_by_socket.get(sid)
    if not session:
        await sio.emit(
            "chat_error",
            {"type": "error", "message": "No session for this connection"},
            room=sid,
        )
        return

    content = ""
    if isinstance(data, dict):
        content = (data.get("content") or "").strip()
        ue = (data.get("user_email") or "").strip()
        if ue:
            session.user_email = ue

    req_id = uuid.uuid4().hex[:12]
    preview = (content[:120] + "…") if len(content) > 120 else content
    log.info(
        "[%s] lifecycle=chat_turn_start sid=%s session_id=%s msg_len=%s preview=%r",
        req_id,
        sid[:8] + "…",
        session.session_id,
        len(content),
        preview,
    )

    session.messages.append({"role": "user", "content": content})
    reply = await asyncio.to_thread(run_multi_agent, session.messages, req_id, session)
    session.messages.append({"role": "assistant", "content": reply})

    log.info(
        "[%s] lifecycle=chat_turn_done sid=%s session_id=%s reply_len=%s",
        req_id,
        sid[:8] + "…",
        session.session_id,
        len(reply or ""),
    )

    await sio.emit(
        "assistant",
        {"type": "assistant", "content": reply},
        room=sid,
    )


app = socketio.ASGIApp(sio, other_asgi_app=fastapi_app)
