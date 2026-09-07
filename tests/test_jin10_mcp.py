from __future__ import annotations

import unittest
from unittest.mock import Mock

from app.jin10_mcp import Jin10McpClient


class Jin10McpClientTests(unittest.TestCase):
    def test_stateless_server_initializes_once_without_session_header(self) -> None:
        client = Jin10McpClient("https://example.test/mcp", "test-token")
        initialize_response = Mock(headers={})
        initialized_response = Mock(headers={})
        tools_response = Mock(headers={})
        client._post = Mock(
            side_effect=[
                (
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {"protocolVersion": "2025-11-25"},
                    },
                    initialize_response,
                ),
                ({"accepted": True}, initialized_response),
                (
                    {
                        "jsonrpc": "2.0",
                        "id": 2,
                        "result": {"tools": [{"name": "list_flash"}]},
                    },
                    tools_response,
                ),
            ]
        )

        tools = client.list_tools()

        self.assertEqual(tools, [{"name": "list_flash"}])
        self.assertTrue(client._started)
        self.assertIsNone(client.session_id)
        self.assertEqual(client._post.call_count, 3)


if __name__ == "__main__":
    unittest.main()
