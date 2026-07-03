from __future__ import annotations

import json
import re
from typing import Any

from app.llm import generate_json, generate_text


STYLE_ANALYSIS_SYSTEM = """
You analyze public posts to create a writing-style profile for an original writing assistant.
Do not identify private traits. Do not recommend impersonation.
The profile must describe transferable writing techniques only: structure, rhythm, tone,
argument patterns, openings, endings, vocabulary level, and platform tactics.
Never include exact reusable catchphrases or long quotes from the source author.
Return JSON only.
""".strip()


def analyze_style_with_llm(
    username: str,
    posts: list[dict[str, Any]],
    model: str,
    provider: str | None = None,
) -> dict[str, Any]:
    samples = _format_samples(posts)
    prompt = f"""
{STYLE_ANALYSIS_SYSTEM}

Author handle for internal labeling only: @{username}

Create a style profile from these posts. Return this JSON shape:
{{
  "style_summary": "short summary",
  "tone": ["..."],
  "opening_patterns": ["..."],
  "structure_patterns": ["..."],
  "sentence_rhythm": "...",
  "argument_moves": ["..."],
  "vocabulary": ["..."],
  "platform_tactics": ["..."],
  "do": ["..."],
  "avoid": [
    "Do not claim to be @{username}.",
    "Do not copy exact phrases, slogans, or distinctive catchphrases.",
    "Do not reuse source post wording."
  ],
  "generation_template": "A compact reusable article structure."
}}

Posts:
{samples}
""".strip()
    return generate_json(model=model, input_text=prompt, provider=provider)


def generate_article_with_llm(
    username: str,
    style_profile: dict[str, Any],
    brief: str,
    platform: str | None,
    target_length: str | None,
    model: str,
    extra_constraints: str | None = None,
    provider: str | None = None,
) -> str:
    profile_json = json.dumps(style_profile, ensure_ascii=False, indent=2)
    prompt = f"""
You are an original writing assistant.

Task:
Write a new article from the user's brief, using the style profile as guidance for
structure, pacing, argument strategy, and tone.

Hard rules:
- Do not impersonate @{username}.
- Do not say or imply the article was written by @{username}.
- Do not copy source post wording, distinctive catchphrases, or unique slogans.
- Do not mention that you are imitating a KOL.
- Produce a polished standalone article in Chinese unless the brief asks otherwise.

Style profile:
{profile_json}

User brief:
{brief}

Platform:
{platform or "not specified"}

Target length:
{target_length or "not specified"}

Publication and length rules:
- Treat the platform as the publishing context, not a decorative label. Adapt structure,
  pacing, paragraphing, and level of explanation to that platform.
- If the platform field includes a relative length mode such as compact, standard, or
  expanded, use it only as a pacing preference.
- Treat Target length as the final length constraint. If relative length mode conflicts
  with Target length, Target length wins.
- Do not exceed the requested length unless the user explicitly asks for completeness over length.

Extra constraints:
{extra_constraints or "none"}
""".strip()
    return generate_text(model=model, input_text=prompt, provider=provider)


def build_heuristic_profile(username: str, posts: list[dict[str, Any]]) -> dict[str, Any]:
    texts = [post.get("text", "") for post in posts if post.get("text")]
    sentence_lengths = [len(sentence) for text in texts for sentence in _split_sentences(text)]
    avg_sentence_length = round(sum(sentence_lengths) / max(len(sentence_lengths), 1), 1)
    question_ratio = round(sum("?" in text or "？" in text for text in texts) / max(len(texts), 1), 2)

    return {
        "style_summary": f"@{username} 的样本已做基础统计画像；建议配置 OPENAI_API_KEY 后生成更细的 LLM 风格画像。",
        "tone": ["direct", "observation-driven"],
        "opening_patterns": ["lead with a clear claim", "use a question when useful"],
        "structure_patterns": ["claim", "context", "reasoning", "takeaway"],
        "sentence_rhythm": f"Average sentence length in sample is about {avg_sentence_length} characters.",
        "argument_moves": ["state a view", "contrast with common belief", "end with a practical takeaway"],
        "vocabulary": ["plain language", "topic-specific nouns"],
        "platform_tactics": [f"Question-mark post ratio: {question_ratio}"],
        "do": ["write clearly", "keep claims concrete", "use original wording"],
        "avoid": [
            f"Do not claim to be @{username}.",
            "Do not copy exact phrases, slogans, or distinctive catchphrases.",
            "Do not reuse source post wording.",
        ],
        "generation_template": "Start with a strong claim, explain the hidden mechanism, give examples, close with an actionable takeaway.",
    }


def _format_samples(posts: list[dict[str, Any]], max_chars: int = 60000) -> str:
    chunks: list[str] = []
    total = 0
    for index, post in enumerate(posts, start=1):
        text = re.sub(r"\s+", " ", post.get("text", "")).strip()
        if not text:
            continue
        chunk = f"{index}. {text}"
        if total + len(chunk) > max_chars:
            break
        chunks.append(chunk)
        total += len(chunk)
    return "\n".join(chunks)


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[.!?。！？\n]+", text) if part.strip()]
