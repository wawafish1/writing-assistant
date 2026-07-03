from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

import requests


class ResearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResearchSettings:
    tavily_api_key: str | None
    alpha_vantage_api_key: str | None
    finnhub_api_key: str | None
    blockbeats_api_key: str | None


def collect_research(
    settings: ResearchSettings,
    topic: str,
    keywords: list[str],
    symbols: list[str],
) -> dict[str, Any]:
    topic = topic.strip()
    clean_keywords = [item.strip() for item in keywords if item.strip()]
    clean_symbols = [item.strip().upper() for item in symbols if item.strip()]
    query_terms = [topic, *clean_keywords]

    sources: dict[str, Any] = {}
    evidence: list[dict[str, Any]] = []

    tavily = tavily_search(settings.tavily_api_key, " ".join(query_terms), max_results=6)
    sources["tavily"] = tavily
    for item in tavily.get("results", []):
        evidence.append(
            {
                "source": "Tavily",
                "type": "web",
                "title": item.get("title"),
                "time": None,
                "url": item.get("url"),
                "summary": item.get("content"),
            }
        )

    finnhub = collect_finnhub(settings.finnhub_api_key, clean_symbols)
    sources["finnhub"] = finnhub
    for symbol, payload in finnhub.items():
        quote = payload.get("quote") or {}
        if quote:
            evidence.append(
                {
                    "source": "Finnhub",
                    "type": "market_quote",
                    "title": f"{symbol} quote",
                    "time": None,
                    "url": None,
                    "summary": (
                        f"{symbol}: current={quote.get('c')}, previous_close={quote.get('pc')}, "
                        f"high={quote.get('h')}, low={quote.get('l')}, open={quote.get('o')}"
                    ),
                    "data": quote,
                }
            )
        for news in payload.get("news", [])[:5]:
            evidence.append(
                {
                    "source": "Finnhub",
                    "type": "company_news",
                    "title": news.get("headline"),
                    "time": _format_unix_time(news.get("datetime")),
                    "url": news.get("url"),
                    "summary": news.get("summary"),
                    "symbol": symbol,
                }
            )

    alpha = collect_alpha_vantage(settings.alpha_vantage_api_key, clean_symbols)
    sources["alpha_vantage"] = alpha
    for symbol, quote in alpha.items():
        if quote:
            evidence.append(
                {
                    "source": "Alpha Vantage",
                    "type": "market_quote",
                    "title": f"{symbol} global quote",
                    "time": quote.get("07. latest trading day"),
                    "url": None,
                    "summary": (
                        f"{symbol}: price={quote.get('05. price')}, "
                        f"change={quote.get('09. change')}, percent={quote.get('10. change percent')}, "
                        f"volume={quote.get('06. volume')}"
                    ),
                    "data": quote,
                }
            )

    blockbeats = blockbeats_newsflash(settings.blockbeats_api_key, size=8)
    sources["blockbeats"] = blockbeats
    for item in blockbeats.get("items", []):
        content = item.get("content") or item.get("title") or item.get("description")
        evidence.append(
            {
                "source": "BlockBeats",
                "type": "crypto_flash",
                "title": item.get("title") or _first_line(content),
                "time": item.get("time") or item.get("created_at") or item.get("createdAt"),
                "url": item.get("url") or item.get("link"),
                "summary": content,
            }
        )

    return {
        "topic": topic,
        "keywords": clean_keywords,
        "symbols": clean_symbols,
        "collected_at": datetime.now(UTC).isoformat(),
        "sources": sources,
        "evidence": [item for item in evidence if item.get("summary") or item.get("title")],
    }


def tavily_search(api_key: str | None, query: str, max_results: int = 6) -> dict[str, Any]:
    if not api_key or not query.strip():
        return {"ok": False, "skipped": True, "results": []}
    response = requests.post(
        "https://api.tavily.com/search",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        },
        timeout=30,
    )
    return _json_or_error(response, "Tavily")


def collect_finnhub(api_key: str | None, symbols: list[str]) -> dict[str, Any]:
    if not api_key or not symbols:
        return {}
    today = datetime.now(UTC).date()
    start = today - timedelta(days=10)
    results: dict[str, Any] = {}
    for symbol in symbols:
        quote_response = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": api_key},
            timeout=20,
        )
        news_response = requests.get(
            "https://finnhub.io/api/v1/company-news",
            params={
                "symbol": symbol,
                "from": start.isoformat(),
                "to": today.isoformat(),
                "token": api_key,
            },
            timeout=20,
        )
        results[symbol] = {
            "quote": _json_or_error(quote_response, "Finnhub quote"),
            "news": _json_or_error(news_response, "Finnhub news"),
        }
    return results


def collect_alpha_vantage(api_key: str | None, symbols: list[str]) -> dict[str, Any]:
    if not api_key or not symbols:
        return {}
    results: dict[str, Any] = {}
    for symbol in symbols:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "GLOBAL_QUOTE", "symbol": symbol, "apikey": api_key},
            timeout=20,
        )
        payload = _json_or_error(response, "Alpha Vantage")
        results[symbol] = payload.get("Global Quote", payload)
    return results


def blockbeats_newsflash(api_key: str | None, size: int = 8) -> dict[str, Any]:
    if not api_key:
        return {"ok": False, "skipped": True, "items": []}
    response = requests.get(
        "http://api-pro.theblockbeats.info/v1/newsflash",
        headers={"api-key": api_key},
        params={"size": size, "lang": "cn"},
        timeout=20,
    )
    payload = _json_or_error(response, "BlockBeats")
    data = payload.get("data", payload)
    if isinstance(data, list):
        return {"items": data}
    if isinstance(data, dict):
        items = data.get("items") or data.get("list") or data.get("data") or []
        return {**data, "items": items if isinstance(items, list) else []}
    return {"items": []}


def format_evidence_brief(research: dict[str, Any], max_items: int = 60) -> str:
    evidence = research.get("evidence", [])
    by_source = Counter(item.get("source") or "unknown" for item in evidence)
    by_type = Counter(
        f"{item.get('source') or 'unknown'}/{item.get('type') or 'evidence'}"
        for item in evidence
    )
    lines = [
        f"研究主题：{research.get('topic', '')}",
        f"采集时间：{research.get('collected_at', '')}",
        "",
        f"证据总数：{len(evidence)}",
        "来源分布：" + "；".join(f"{source} {count}条" for source, count in by_source.items()),
        "类型分布：" + "；".join(f"{item_type} {count}条" for item_type, count in by_type.items()),
        "",
        "证据清单：",
    ]
    selected = _balanced_evidence(evidence, max_items=max_items)
    for index, item in enumerate(selected, start=1):
        title = item.get("title") or "(无标题)"
        source = item.get("source") or "unknown"
        item_type = item.get("type") or "evidence"
        time = item.get("time") or "时间未标注"
        url = item.get("url") or ""
        summary = _compact(item.get("summary") or "", 600)
        lines.append(f"{index}. [{source}/{item_type}] {title}")
        lines.append(f"   时间：{time}")
        if url:
            lines.append(f"   链接：{url}")
        if summary:
            lines.append(f"   摘要/数据：{summary}")
    remaining = max(0, len(evidence) - len(selected))
    if remaining:
        lines.append("")
        lines.append(f"还有 {remaining} 条证据未在简报中展开；接口返回的 evidence 字段里保留了完整列表。")
    return "\n".join(lines)


def _balanced_evidence(evidence: list[dict[str, Any]], max_items: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_order: list[str] = []
    for item in evidence:
        source = item.get("source") or "unknown"
        if source not in buckets:
            source_order.append(source)
        buckets[source].append(item)

    selected: list[dict[str, Any]] = []
    while len(selected) < max_items:
        changed = False
        for source in source_order:
            if buckets[source]:
                selected.append(buckets[source].pop(0))
                changed = True
                if len(selected) >= max_items:
                    break
        if not changed:
            break
    return selected


def _json_or_error(response: requests.Response, label: str) -> Any:
    if not response.ok:
        raise ResearchError(f"{label} HTTP {response.status_code}: {response.text[:300]}")
    try:
        return response.json()
    except ValueError as exc:
        raise ResearchError(f"{label} did not return JSON: {response.text[:300]}") from exc


def _compact(text: str, limit: int) -> str:
    normalized = " ".join(str(text).split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def _first_line(text: Any) -> str | None:
    if not text:
        return None
    return str(text).strip().splitlines()[0][:80]


def _format_unix_time(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        timestamp = int(value)
    except (TypeError, ValueError):
        return str(value)
    return datetime.fromtimestamp(timestamp, UTC).isoformat()
