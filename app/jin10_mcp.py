from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


class Jin10McpError(RuntimeError):
    pass


@dataclass(frozen=True)
class Jin10McpResponse:
    raw: dict[str, Any]
    data: Any


class Jin10McpClient:
    def __init__(
        self,
        server_url: str,
        bearer_token: str,
        protocol_version: str = "2025-11-25",
    ):
        if not bearer_token:
            raise Jin10McpError("Missing JIN10_MCP_TOKEN.")
        self.server_url = server_url
        self.bearer_token = bearer_token
        self.protocol_version = protocol_version
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {bearer_token}",
            }
        )
        self.session_id: str | None = None
        self._next_id = 1

    def initialize(self) -> dict[str, Any]:
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {
                    "name": "twitter-style-assistant",
                    "version": "0.1.0",
                },
            },
        }
        message, response = self._post(payload)
        session_id = response.headers.get("Mcp-Session-Id")
        if not session_id:
            raise Jin10McpError("Jin10 MCP did not return Mcp-Session-Id.")
        self.session_id = session_id
        result = message.get("result")
        if not isinstance(result, dict):
            raise Jin10McpError(f"Invalid initialize response: {message}")
        return result

    def initialized(self) -> None:
        self._ensure_session()
        payload = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        self._post(payload)

    def start(self) -> dict[str, Any]:
        result = self.initialize()
        self.initialized()
        return result

    def list_tools(self) -> list[dict[str, Any]]:
        result = self._request("tools/list", {})
        tools = result.get("tools", [])
        return _repair_mojibake(tools)

    def list_resources(self) -> list[dict[str, Any]]:
        result = self._request("resources/list", {})
        resources = result.get("resources", [])
        return _repair_mojibake(resources)

    def read_resource(self, uri: str) -> Any:
        result = self._request("resources/read", {"uri": uri})
        return self._extract_data(result)

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Jin10McpResponse:
        result = self._request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments or {},
            },
        )
        if result.get("isError"):
            raise Jin10McpError(f"Jin10 tool error: {result}")
        return Jin10McpResponse(raw=result, data=self._extract_data(result))

    def get_quote(self, code: str) -> Jin10McpResponse:
        return self.call_tool("get_quote", {"code": code})

    def get_kline(
        self,
        code: str,
        time: int | None = None,
        count: int | None = None,
    ) -> Jin10McpResponse:
        args: dict[str, Any] = {"code": code}
        if time is not None:
            args["time"] = time
        if count is not None:
            args["count"] = count
        return self.call_tool("get_kline", args)

    def list_flash(self, cursor: str | None = None) -> Jin10McpResponse:
        return self.call_tool("list_flash", _pagination_args(cursor))

    def search_flash(self, keyword: str) -> Jin10McpResponse:
        return self.call_tool("search_flash", {"keyword": keyword})

    def list_news(self, cursor: str | None = None) -> Jin10McpResponse:
        return self.call_tool("list_news", _pagination_args(cursor))

    def search_news(self, keyword: str, cursor: str | None = None) -> Jin10McpResponse:
        args = {"keyword": keyword}
        args.update(_pagination_args(cursor))
        return self.call_tool("search_news", args)

    def get_news(self, news_id: str) -> Jin10McpResponse:
        return self.call_tool("get_news", {"id": news_id})

    def list_calendar(self) -> Jin10McpResponse:
        return self.call_tool("list_calendar", {})

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self._ensure_session()
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id(),
            "method": method,
            "params": params,
        }
        message, _ = self._post(payload)
        if "error" in message:
            raise Jin10McpError(f"Jin10 JSON-RPC error: {message['error']}")
        result = message.get("result")
        if not isinstance(result, dict):
            raise Jin10McpError(f"Invalid Jin10 MCP result: {message}")
        return result

    def _ensure_session(self) -> None:
        if not self.session_id:
            self.start()

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], requests.Response]:
        headers = {}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        response = self.session.post(
            self.server_url,
            headers=headers,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=30,
        )
        if response.status_code == 202:
            return {"accepted": True}, response
        if not response.ok:
            raise Jin10McpError(f"Jin10 MCP HTTP {response.status_code}: {response.text[:500]}")
        text = response.content.decode("utf-8", errors="replace")
        return _parse_mcp_response(text), response

    def _request_id(self) -> int:
        request_id = self._next_id
        self._next_id += 1
        return request_id

    @staticmethod
    def _extract_data(result: dict[str, Any]) -> Any:
        structured = result.get("structuredContent")
        if isinstance(structured, dict) and "data" in structured:
            return _repair_mojibake(structured["data"])
        if structured is not None:
            return _repair_mojibake(structured)

        content = result.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict) or item.get("type") != "text":
                    continue
                text = item.get("text")
                if not isinstance(text, str):
                    continue
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and "data" in parsed:
                    return _repair_mojibake(parsed["data"])
                return _repair_mojibake(parsed)
        return result


def _parse_mcp_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped.startswith("{"):
        return json.loads(stripped)

    data_index = stripped.find("data:")
    if data_index >= 0:
        payload = stripped[data_index + len("data:") :].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            pass

    data_lines: list[str] = []
    for line in stripped.splitlines():
        if line.startswith("data:"):
            data_lines.append(line.removeprefix("data:").strip())
    if not data_lines:
        raise Jin10McpError(f"Jin10 MCP response did not contain data lines: {text[:500]}")
    return json.loads("\n".join(data_lines))


def _pagination_args(cursor: str | None) -> dict[str, Any]:
    if cursor:
        return {"cursor": cursor}
    return {}


def _repair_mojibake(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _repair_mojibake(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_repair_mojibake(item) for item in value]
    if isinstance(value, str):
        return _repair_string(value)
    return value


def _repair_string(value: str) -> str:
    if not _looks_mojibake(value):
        return value
    try:
        repaired = value.encode("latin1").decode("utf-8")
    except UnicodeError:
        return value
    return repaired if repaired else value


def _looks_mojibake(value: str) -> bool:
    markers = ("Ã", "Â", "æ", "è", "é", "å", "ç", "ä", "\u0080", "\u0081", "\u0082")
    return any(marker in value for marker in markers)
