from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
CREATE TABLE IF NOT EXISTS authors (
    username TEXT PRIMARY KEY,
    x_user_id TEXT NOT NULL,
    display_name TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS posts (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL,
    created_at TEXT,
    text TEXT NOT NULL,
    lang TEXT,
    public_metrics_json TEXT,
    raw_json TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    FOREIGN KEY(username) REFERENCES authors(username)
);

CREATE INDEX IF NOT EXISTS idx_posts_username_created_at
ON posts(username, created_at DESC);

CREATE TABLE IF NOT EXISTS style_profiles (
    username TEXT PRIMARY KEY,
    profile_json TEXT NOT NULL,
    sample_size INTEGER NOT NULL,
    model TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(username) REFERENCES authors(username)
);

CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    brief TEXT NOT NULL,
    platform TEXT,
    target_length TEXT,
    draft TEXT NOT NULL,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(username) REFERENCES authors(username)
);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.init()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def init(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    def upsert_author(self, username: str, x_user_id: str, display_name: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO authors(username, x_user_id, display_name, created_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    x_user_id = excluded.x_user_id,
                    display_name = excluded.display_name
                """,
                (username.lower(), x_user_id, display_name, utc_now()),
            )

    def upsert_posts(self, username: str, posts: list[dict[str, Any]]) -> int:
        with self.connect() as conn:
            before = conn.total_changes
            for post in posts:
                metrics = post.get("public_metrics")
                conn.execute(
                    """
                    INSERT INTO posts(
                        id, username, created_at, text, lang,
                        public_metrics_json, raw_json, inserted_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        username = excluded.username,
                        created_at = excluded.created_at,
                        text = excluded.text,
                        lang = excluded.lang,
                        public_metrics_json = excluded.public_metrics_json,
                        raw_json = excluded.raw_json
                    """,
                    (
                        post["id"],
                        username.lower(),
                        post.get("created_at"),
                        post.get("text", ""),
                        post.get("lang"),
                        json.dumps(metrics, ensure_ascii=False) if metrics else None,
                        json.dumps(post, ensure_ascii=False),
                        utc_now(),
                    ),
                )
            return conn.total_changes - before

    def get_posts(self, username: str, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT id, username, created_at, text, lang, public_metrics_json
                FROM posts
                WHERE username = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (username.lower(), limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_posts(self, username: str) -> int:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM posts
                WHERE username = ?
                """,
                (username.lower(),),
            ).fetchone()
        return int(row["count"]) if row else 0

    def save_style_profile(
        self,
        username: str,
        profile: dict[str, Any],
        sample_size: int,
        model: str,
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO style_profiles(username, profile_json, sample_size, model, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    sample_size = excluded.sample_size,
                    model = excluded.model,
                    updated_at = excluded.updated_at
                """,
                (
                    username.lower(),
                    json.dumps(profile, ensure_ascii=False, indent=2),
                    sample_size,
                    model,
                    utc_now(),
                ),
            )

    def get_style_profile(self, username: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT username, profile_json, sample_size, model, updated_at
                FROM style_profiles
                WHERE username = ?
                """,
                (username.lower(),),
            ).fetchone()
        if row is None:
            return None
        data = dict(row)
        data["profile"] = json.loads(data.pop("profile_json"))
        return data

    def list_style_profiles(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT sp.username, sp.sample_size, sp.model, sp.updated_at, a.display_name
                FROM style_profiles sp
                LEFT JOIN authors a ON a.username = sp.username
                ORDER BY sp.updated_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def list_authors(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT username, display_name, created_at
                FROM authors
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def save_generation(
        self,
        username: str,
        brief: str,
        platform: str | None,
        target_length: str | None,
        draft: str,
        model: str,
    ) -> int:
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO generations(
                    username, brief, platform, target_length, draft, model, created_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (username.lower(), brief, platform, target_length, draft, model, utc_now()),
            )
            return int(cursor.lastrowid)
