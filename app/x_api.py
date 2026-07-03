from __future__ import annotations

import time
from typing import Any

import requests


X_API_BASE = "https://api.x.com/2"


class XApiError(RuntimeError):
    pass


class XClient:
    def __init__(self, bearer_token: str):
        if not bearer_token:
            raise XApiError("缺少 X_BEARER_TOKEN。请先在 .env 文件里填写你的 X API Bearer Token。")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {bearer_token}"})

    def get_user(self, username: str) -> dict[str, Any]:
        clean_username = username.strip().lstrip("@")
        response = self.session.get(
            f"{X_API_BASE}/users/by/username/{clean_username}",
            params={"user.fields": "id,name,username,verified,description"},
            timeout=30,
        )
        self._raise_for_error(response)
        return response.json()["data"]

    def fetch_recent_posts(
        self,
        user_id: str,
        limit: int = 500,
        exclude_replies: bool = True,
        exclude_retweets: bool = True,
        sleep_seconds: float = 1.0,
    ) -> list[dict[str, Any]]:
        if limit < 1 or limit > 500:
            raise XApiError("limit 必须在 1 到 500 之间。")

        posts: list[dict[str, Any]] = []
        pagination_token: str | None = None

        while len(posts) < limit:
            excludes: list[str] = []
            if exclude_replies:
                excludes.append("replies")
            if exclude_retweets:
                excludes.append("retweets")

            params: dict[str, Any] = {
                "max_results": min(100, limit - len(posts)),
                "tweet.fields": "created_at,public_metrics,lang,entities,note_tweet",
            }
            if excludes:
                params["exclude"] = ",".join(excludes)
            if pagination_token:
                params["pagination_token"] = pagination_token

            response = self.session.get(
                f"{X_API_BASE}/users/{user_id}/tweets",
                params=params,
                timeout=30,
            )
            self._raise_for_error(response)
            payload = response.json()

            batch = payload.get("data", [])
            posts.extend(self._normalize_post(post) for post in batch)

            pagination_token = payload.get("meta", {}).get("next_token")
            if not pagination_token or not batch:
                break
            time.sleep(sleep_seconds)

        return posts[:limit]

    @staticmethod
    def _normalize_post(post: dict[str, Any]) -> dict[str, Any]:
        note_text = post.get("note_tweet", {}).get("text")
        article_text = post.get("article", {}).get("text")
        full_text = article_text or note_text or post.get("text", "")
        normalized = dict(post)
        normalized["text"] = full_text
        return normalized

    @staticmethod
    def _raise_for_error(response: requests.Response) -> None:
        if response.ok:
            return
        try:
            payload = response.json()
        except ValueError:
            payload = response.text
        raise XApiError(f"X API 请求失败：{response.status_code} {payload}")
