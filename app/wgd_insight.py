from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests


WGD_API_BASE_URL = "https://wgdinsight.com/api/v1/agent-skill/public"
WGD_SITE_URL = "https://wgdinsight.com"


class WgdInsightError(RuntimeError):
    pass


def get_hot_tickers(timeout: float = 12) -> list[dict[str, Any]]:
    payload = _get_json("/hot-tickers", timeout=timeout)
    rows = payload.get("tickers") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [item for item in rows if isinstance(item, dict)]


def get_ticker(symbol: str, timeout: float = 12) -> dict[str, Any] | None:
    clean_symbol = _clean_symbol(symbol)
    payload = _get_json(f"/tickers/{clean_symbol}", timeout=timeout)
    ticker = payload.get("ticker") if isinstance(payload, dict) else None
    return ticker if isinstance(ticker, dict) else None


def collect_tickers(
    symbols: list[str],
    timeout: float = 12,
    max_items: int = 8,
) -> dict[str, Any]:
    clean_symbols: list[str] = []
    for symbol in symbols:
        try:
            clean_symbol = _clean_symbol(symbol)
        except ValueError:
            continue
        if clean_symbol not in clean_symbols:
            clean_symbols.append(clean_symbol)
        if len(clean_symbols) >= max_items:
            break
    if not clean_symbols:
        return {"ok": False, "skipped": True, "items": [], "errors": []}

    items_by_symbol: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    workers = min(4, len(clean_symbols))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_symbol = {
            executor.submit(get_ticker, symbol, timeout): symbol for symbol in clean_symbols
        }
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                item = future.result()
                if item:
                    items_by_symbol[symbol] = item
            except Exception as exc:
                errors.append(f"{symbol}: {type(exc).__name__}: {_compact(str(exc), 180)}")

    items = [items_by_symbol[symbol] for symbol in clean_symbols if symbol in items_by_symbol]
    return {
        "ok": bool(items),
        "items": items,
        "errors": errors,
        "requested": clean_symbols,
    }


def public_ticker_url(symbol: str) -> str:
    clean_symbol = _clean_symbol(symbol)
    return f"{WGD_API_BASE_URL}/tickers/{clean_symbol}"


def normalize_digest_time(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    for suffix in (" ET", " EST", " EDT"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
            break
    parsed: datetime | None = None
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, pattern)
            break
        except ValueError:
            continue
    if parsed is None:
        return str(value)
    try:
        eastern = ZoneInfo("America/New_York")
    except ZoneInfoNotFoundError:
        eastern = timezone(timedelta(hours=_fallback_eastern_offset(parsed)))
    return parsed.replace(tzinfo=eastern).astimezone(UTC).isoformat()


def _fallback_eastern_offset(value: datetime) -> int:
    march_first = datetime(value.year, 3, 1)
    second_sunday_march = 1 + (6 - march_first.weekday()) % 7 + 7
    november_first = datetime(value.year, 11, 1)
    first_sunday_november = 1 + (6 - november_first.weekday()) % 7
    dst_start = datetime(value.year, 3, second_sunday_march, 2)
    dst_end = datetime(value.year, 11, first_sunday_november, 2)
    return -4 if dst_start <= value < dst_end else -5


def _get_json(path: str, timeout: float) -> Any:
    response = requests.get(
        f"{WGD_API_BASE_URL}{path}",
        headers={
            "Accept": "application/json",
            "User-Agent": "WritingAssistant/0.1",
        },
        timeout=timeout,
    )
    if response.status_code == 429:
        raise WgdInsightError("WGD Insight 匿名查询达到频率限制，请稍后再试。")
    if not response.ok:
        raise WgdInsightError(
            f"WGD Insight HTTP {response.status_code}: {_compact(response.text, 200)}"
        )
    try:
        return response.json()
    except ValueError as exc:
        raise WgdInsightError("WGD Insight 没有返回 JSON。") from exc


def _clean_symbol(symbol: Any) -> str:
    clean_symbol = str(symbol or "").strip().lstrip("$").upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", clean_symbol):
        raise ValueError("无效的美股代码。")
    return clean_symbol


def _compact(value: Any, limit: int) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."
