"""NeMo Guardrails as pre/post checks around the RAG graph.

Input rails:  regex PII block (SSN, card numbers, DOB), then an LLM self-check for jailbreak/prompt
              injection, off-topic requests, requests about a named person's records, and evasion.
Output rails: regex PII block, then an LLM self-check for personal-data disclosure, evasion advice
              and content injected from a retrieved document.
Grounding (hallucination) is checked separately by the grader node in graph.py, which sees the
sources; NeMo's self-check-facts would duplicate it.
"""

from functools import lru_cache
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType

from .config import chat_model


@lru_cache(maxsize=1)
def rails() -> LLMRails:
    return LLMRails(RailsConfig.from_path(str(Path(__file__).parent / "rails")), llm=chat_model())


async def check_input(text: str) -> str | None:
    """Return the name of the blocking rail, or None if the input is allowed."""
    r = await rails().check_async([{"role": "user", "content": text}], rail_types=[RailType.INPUT])
    return r.rail or "input rail" if r.status == RailStatus.BLOCKED else None


async def check_output(question: str, answer: str) -> str | None:
    r = await rails().check_async([{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                                  rail_types=[RailType.OUTPUT])
    return r.rail or "output rail" if r.status == RailStatus.BLOCKED else None
