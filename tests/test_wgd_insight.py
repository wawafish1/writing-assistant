from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from app.wgd_insight import collect_tickers, get_hot_tickers, normalize_digest_time


class WgdInsightTests(unittest.TestCase):
    @patch("app.wgd_insight.requests.get")
    def test_get_hot_tickers_returns_rows(self, mock_get: Mock) -> None:
        response = Mock(status_code=200, ok=True)
        response.json.return_value = {"tickers": [{"symbol": "NVDA", "rank": 1}]}
        mock_get.return_value = response

        self.assertEqual(get_hot_tickers(), [{"symbol": "NVDA", "rank": 1}])

    @patch("app.wgd_insight.get_ticker")
    def test_collect_tickers_preserves_requested_order(self, mock_get: Mock) -> None:
        mock_get.side_effect = lambda symbol, timeout: {"symbol": symbol}

        result = collect_tickers(["meta", "$NVDA", "bad/symbol"])

        self.assertTrue(result["ok"])
        self.assertEqual([item["symbol"] for item in result["items"]], ["META", "NVDA"])
        self.assertEqual(result["errors"], [])

    def test_normalize_digest_time(self) -> None:
        normalized = normalize_digest_time("2026-07-18 08:03 ET")

        self.assertIsNotNone(normalized)
        self.assertIn("2026-07-18T12:03:00", normalized)


if __name__ == "__main__":
    unittest.main()
