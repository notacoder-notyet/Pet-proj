"""SQLite: события, предсказания, репорты / SQLite: events, predictions, reports."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from telegram import Update

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "logs"
DB_PATH = DATA_DIR / "app.db"
JSONL_PATH = DATA_DIR / "telegram_dialogs.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Создаёт таблицы при первом запуске / Create tables on first run."""
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                event TEXT NOT NULL,
                user_id INTEGER,
                username TEXT,
                chat_id INTEGER,
                text TEXT,
                extra_json TEXT
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                user_id INTEGER,
                username TEXT,
                source TEXT NOT NULL,
                input_json TEXT NOT NULL,
                prediction INTEGER,
                probability REAL
            );
            CREATE TABLE IF NOT EXISTS error_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                user_id INTEGER,
                username TEXT,
                chat_id INTEGER,
                message TEXT NOT NULL,
                last_error TEXT,
                last_input TEXT
            );
            CREATE TABLE IF NOT EXISTS last_context (
                user_id INTEGER PRIMARY KEY,
                last_error TEXT,
                last_input TEXT,
                updated_at TEXT NOT NULL
            );
            """
        )


def _user_fields(update: Optional[Update]) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "user_id": None,
        "username": None,
        "chat_id": None,
        "text": None,
    }
    if update and update.effective_user:
        fields["user_id"] = update.effective_user.id
        fields["username"] = update.effective_user.username
    if update and update.effective_chat:
        fields["chat_id"] = update.effective_chat.id
    if update and update.message and update.message.text:
        fields["text"] = update.message.text[:1000]
    return fields


def log_event(
    event: str,
    update: Optional[Update] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    """Пишет событие в SQLite и JSONL / Write event to SQLite and JSONL."""
    init_db()
    ts = _now()
    user = _user_fields(update)
    extra = extra or {}
    extra_json = json.dumps(extra, ensure_ascii=False, default=str)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO events (ts, event, user_id, username, chat_id, text, extra_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, event, user["user_id"], user["username"], user["chat_id"], user["text"], extra_json),
        )
    payload = {"ts": ts, "event": event, **user, **extra}
    with JSONL_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def save_prediction(
    source: str,
    input_data: dict,
    prediction: int,
    probability: float,
    update: Optional[Update] = None,
) -> None:
    """Сохраняет вход датчиков и ответ модели / Store sensor input and model output."""
    init_db()
    user = _user_fields(update)
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO predictions (ts, user_id, username, source, input_json, prediction, probability)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                user["user_id"],
                user["username"],
                source,
                json.dumps(input_data, ensure_ascii=False, default=str),
                int(prediction),
                float(probability),
            ),
        )


def remember_error(update: Update, error: str, input_data: Optional[dict] = None) -> None:
    """Запоминает последнюю ошибку пользователя для /report / Keep last error for /report."""
    if not update.effective_user:
        return
    init_db()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO last_context (user_id, last_error, last_input, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                last_error = excluded.last_error,
                last_input = excluded.last_input,
                updated_at = excluded.updated_at
            """,
            (
                update.effective_user.id,
                error[:4000],
                json.dumps(input_data, ensure_ascii=False, default=str) if input_data else None,
                _now(),
            ),
        )


def save_error_report(update: Update, message: str) -> int:
    """Сохраняет репорт и возвращает его id / Save report and return its id."""
    init_db()
    user = _user_fields(update)
    last_error = None
    last_input = None
    if user["user_id"] is not None:
        with _connect() as conn:
            row = conn.execute(
                "SELECT last_error, last_input FROM last_context WHERE user_id = ?",
                (user["user_id"],),
            ).fetchone()
            if row:
                last_error = row["last_error"]
                last_input = row["last_input"]
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO error_reports (ts, user_id, username, chat_id, message, last_error, last_input)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                user["user_id"],
                user["username"],
                user["chat_id"],
                message[:4000],
                last_error,
                last_input,
            ),
        )
        return int(cur.lastrowid)
