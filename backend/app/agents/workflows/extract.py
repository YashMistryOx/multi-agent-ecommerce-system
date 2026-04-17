"""Stable patterns only: order ID (OM-#####), email, optional numeric list index."""

from __future__ import annotations

import re

ORDER_ID_RE = re.compile(r"\b(OM-\d+)\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

# Whole message is a single email (identity verification turns), not "email + sentence".
_EMAIL_ONLY_RE = re.compile(
    r"^\s*(?:mailto:)?([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})\s*$",
    re.IGNORECASE,
)


def message_is_email_only(text: str) -> bool:
    """True when the user message is only an email address (possibly mailto:, whitespace)."""
    return bool(_EMAIL_ONLY_RE.match((text or "").strip()))


def extract_order_id(text: str) -> str | None:
    m = ORDER_ID_RE.search(text or "")
    return m.group(1).upper() if m else None


def extract_email(text: str) -> str | None:
    m = EMAIL_RE.search(text or "")
    return m.group(0).strip().lower() if m else None


def pick_index_from_message(text: str) -> int | None:
    """0-based index for a single-digit or common ordinal reply (mechanical, not NL)."""
    t = (text or "").strip().lower()
    if t in ("1", "first", "one", "first one"):
        return 0
    if t in ("2", "second", "two"):
        return 1
    if t in ("3", "third", "three"):
        return 2
    m = re.match(r"^#?\s*(\d)$", t)
    if m:
        return int(m.group(1)) - 1
    return None
