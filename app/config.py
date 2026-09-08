from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ[key] = value


@dataclass(frozen=True)
class Settings:
    app_auth_username: str | None
    app_auth_password: str | None
    openai_api_key: str | None
    xai_api_key: str | None
    deepseek_api_key: str | None
    tavily_api_key: str | None
    alpha_vantage_api_key: str | None
    finnhub_api_key: str | None
    blockbeats_api_key: str | None
    jin10_mcp_url: str
    jin10_mcp_token: str | None
    database_path: Path
    openai_model: str
    xai_model: str
    deepseek_model: str
    openai_base_url: str | None
    xai_base_url: str
    deepseek_base_url: str
    hot_topic_provider: str
    hot_topic_model: str


def get_settings() -> Settings:
    load_dotenv()
    database_path = Path(os.getenv("DATABASE_PATH", "data/twitter_style.db"))
    return Settings(
        app_auth_username=os.getenv("APP_AUTH_USERNAME"),
        app_auth_password=os.getenv("APP_AUTH_PASSWORD"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        xai_api_key=os.getenv("XAI_API_KEY"),
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
        blockbeats_api_key=os.getenv("BLOCKBEATS_API_KEY"),
        jin10_mcp_url=os.getenv("JIN10_MCP_URL", "https://mcp.jin10.com/mcp"),
        jin10_mcp_token=os.getenv("JIN10_MCP_TOKEN"),
        database_path=database_path,
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5"),
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        xai_model=os.getenv("XAI_MODEL", "grok-4"),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro"),
        xai_base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        hot_topic_provider=os.getenv("HOT_TOPIC_PROVIDER", "deepseek").strip().lower(),
        hot_topic_model=os.getenv("HOT_TOPIC_MODEL", "deepseek-chat").strip(),
    )
