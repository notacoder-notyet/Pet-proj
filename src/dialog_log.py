"""Журнал диалогов бота (JSONL) / Telegram dialog log (JSONL)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from telegram import Update

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "logs" / "telegram_dialogs.jsonl"


def log_event(
    event: str,
    update: Optional[Update] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Пишет одну строку JSON в logs/telegram_dialogs.jsonl / Append one JSON line."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
    }
    if update and update.effective_user:
        payload["user_id"] = update.effective_user.id
        payload["username"] = update.effective_user.username
    if update and update.effective_chat:
        payload["chat_id"] = update.effective_chat.id
    if update and update.message and update.message.text:
        # текст пользователя для разбора ошибок ввода / user text for debugging input errors
        payload["text"] = update.message.text[:500]
    if extra:
        payload.update(extra)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
