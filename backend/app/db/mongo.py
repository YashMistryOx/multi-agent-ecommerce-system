"""PyMongo connection pooling, database/collection accessors, and shared query helpers."""

from __future__ import annotations

import re
import threading
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from app.settings import get_settings

__all__ = [
    "find_order_document",
    "get_mongo_client",
    "get_mongo_database",
    "get_orders_collection",
]

_client_lock = threading.Lock()
_mongo_client: MongoClient | None = None


def get_mongo_client() -> MongoClient:
    """Singleton MongoClient (thread-safe, lazy)."""
    global _mongo_client
    with _client_lock:
        if _mongo_client is None:
            s = get_settings()
            _mongo_client = MongoClient(
                s.mongodb_uri,
                serverSelectionTimeoutMS=8000,
            )
        return _mongo_client


def get_mongo_database(name: str | None = None) -> Database:
    """Return a database handle (default from settings)."""
    db_name = name if name is not None else get_settings().mongodb_database
    return get_mongo_client()[db_name]


def get_orders_collection() -> Collection:
    """Orders collection configured in settings."""
    s = get_settings()
    return get_mongo_client()[s.mongodb_database][s.mongodb_orders_collection]


def find_order_document(collection: Collection, raw_id: str) -> dict[str, Any] | None:
    """
    Find one order document by order_id / orderId (case-insensitive) or MongoDB _id.
    """
    q = raw_id.strip()
    upper = q.upper()
    safe = re.escape(q)

    doc = collection.find_one({"order_id": {"$regex": f"^{safe}$", "$options": "i"}})
    if doc:
        return doc
    doc = collection.find_one({"orderId": {"$regex": f"^{safe}$", "$options": "i"}})
    if doc:
        return doc
    doc = collection.find_one({"order_id": upper}) or collection.find_one({"order_id": q})
    if doc:
        return doc
    doc = collection.find_one({"orderId": upper}) or collection.find_one({"orderId": q})
    if doc:
        return doc

    try:
        oid = ObjectId(q)
        return collection.find_one({"_id": oid})
    except (InvalidId, TypeError):
        pass

    return None
