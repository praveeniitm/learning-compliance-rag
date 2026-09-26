"""Thin wrapper around an OpenAI-compatible chat endpoint (OpenAI, Ollama, vLLM)."""

from __future__ import annotations

import json
import re
from functools import lru_cache

from openai import OpenAI

from .config import LLMConfig, get_llm_config


@lru_cache(maxsize=4)
def _client(api_key: str, base_url: str | None, timeout_s: float) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)


def chat(messages: list[dict], cfg: LLMConfig | None = None, json_mode: bool = False, **overrides) -> str:
    cfg = cfg or get_llm_config()
    kwargs = dict(
        model=overrides.pop("model", cfg.model),
        messages=messages,
        temperature=overrides.pop("temperature", cfg.temperature),
        max_tokens=overrides.pop("max_tokens", cfg.max_tokens),
        **overrides,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = _client(cfg.api_key, cfg.base_url, cfg.timeout_s).chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output, tolerating code fences or leading prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))
