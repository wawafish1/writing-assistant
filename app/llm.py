from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


class LlmError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str


def parse_model_spec(model: str, provider: str | None = None) -> ModelSpec:
    if provider:
        return ModelSpec(provider=provider.lower(), model=model)
    if ":" in model:
        prefix, model_name = model.split(":", 1)
        normalized = prefix.lower()
        if normalized == "grok":
            normalized = "xai"
        return ModelSpec(provider=normalized, model=model_name)
    return ModelSpec(provider="openai", model=model)


def generate_text(model: str, input_text: str, provider: str | None = None) -> str:
    spec = parse_model_spec(model, provider)
    if spec.provider == "openai":
        return _generate_openai(spec.model, input_text)
    if spec.provider == "xai":
        return _generate_openai_compatible(
            api_key_name="XAI_API_KEY",
            base_url=os.getenv("XAI_BASE_URL", "https://api.x.ai/v1"),
            model=spec.model,
            input_text=input_text,
            provider_label="xAI/Grok",
        )
    if spec.provider == "deepseek":
        return _generate_openai_compatible(
            api_key_name="DEEPSEEK_API_KEY",
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=spec.model,
            input_text=input_text,
            provider_label="DeepSeek",
        )
    raise LlmError(f"不支持的模型供应商：{spec.provider}")


def generate_json(model: str, input_text: str, provider: str | None = None) -> dict[str, Any]:
    text = generate_text(model=model, input_text=input_text, provider=provider)
    try:
        return json.loads(_extract_json(text))
    except json.JSONDecodeError as exc:
        raise LlmError(f"模型没有返回合法 JSON，前 500 个字符是：{text[:500]}") from exc


def _generate_openai(model: str, input_text: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise LlmError("缺少 OPENAI_API_KEY。请先在 .env 文件里填写你的 OpenAI API Key。")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LlmError("还没有安装 openai 依赖。请运行：pip install -r requirements.txt") from exc

    base_url = os.getenv("OPENAI_BASE_URL")
    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)
    if base_url:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": input_text}],
            temperature=0.2,
        )
        content = response.choices[0].message.content
        if not content:
            raise LlmError("OpenAI 兼容接口没有返回文本。")
        return content
    response = client.responses.create(model=model, input=input_text)
    return response.output_text


def _generate_openai_compatible(
    api_key_name: str,
    base_url: str,
    model: str,
    input_text: str,
    provider_label: str,
) -> str:
    api_key = os.getenv(api_key_name)
    if not api_key:
        raise LlmError(f"缺少 {api_key_name}。请先在 .env 文件里填写你的 {provider_label} API Key。")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LlmError("还没有安装 openai 依赖。请运行：pip install -r requirements.txt") from exc

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": input_text}],
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if not content:
        raise LlmError(f"{provider_label} 没有返回文本。")
    return content


def _extract_json(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end >= start:
        return stripped[start : end + 1]
    return stripped
