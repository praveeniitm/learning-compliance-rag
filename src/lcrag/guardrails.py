"""NeMo Guardrails as pre/post checks around the RAG graph (LLMRails.check_async: rails only, no generation).

Input rails:  regex PII block (SSN, card numbers, DOB), then an LLM self-check for jailbreak/prompt
              injection, off-topic requests, requests about a named person's records, and evasion.
Output rails: regex PII block, then an LLM self-check for personal-data disclosure, evasion advice
              and content injected from a retrieved document.
Grounding (hallucination) is checked by the grader node in graph.py, which sees the sources;
NeMo's self-check-facts would duplicate it.

All checks run on one dedicated event loop: the graph executes nodes in worker threads, and the
async OpenAI client inside NeMo must not be shared across event loops. Rails fail closed: if a check
errors or times out, the request is blocked rather than passed through unchecked.
"""

import asyncio
import logging
import threading
from functools import lru_cache
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType

from .config import CHECK_MODEL, chat_model

log = logging.getLogger(__name__)
_LOOP = asyncio.new_event_loop()
threading.Thread(target=_LOOP.run_forever, daemon=True, name="guardrails-loop").start()


@lru_cache(maxsize=1)
def rails() -> LLMRails:
    return LLMRails(RailsConfig.from_path(str(Path(__file__).parent / "rails")), llm=chat_model(CHECK_MODEL))


def _blocked(messages: list[dict], rail_type: RailType) -> str | None:
    try:
        fut = asyncio.run_coroutine_threadsafe(rails().check_async(messages, rail_types=[rail_type]), _LOOP)
        r = fut.result(timeout=60)
    except Exception:  # noqa: BLE001 - fail closed on any rail failure
        log.exception("%s rail failed", rail_type.value)
        return f"{rail_type.value} rail error (fail-closed)"
    return (r.rail or rail_type.value) if r.status == RailStatus.BLOCKED else None


def check_input(text: str) -> str | None:
    """Name of the blocking rail, or None if allowed."""
    return _blocked([{"role": "user", "content": text}], RailType.INPUT)


def check_output(question: str, answer: str) -> str | None:
    return _blocked([{"role": "user", "content": question}, {"role": "assistant", "content": answer}], RailType.OUTPUT)
