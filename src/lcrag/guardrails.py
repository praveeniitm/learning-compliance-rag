"""NeMo Guardrails as pre/post checks around the RAG graph (LLMRails.check: rails only, no generation).

Input rails:  regex PII block (SSN, card numbers, DOB), then an LLM self-check for jailbreak/prompt
              injection, off-topic requests, requests about a named person's records, and evasion.
Output rails: regex PII block, then an LLM self-check for personal-data disclosure, evasion advice
              and content injected from a retrieved document.
Grounding (hallucination) is checked by the grader node in graph.py, which sees the sources;
NeMo's self-check-facts would duplicate it.
"""

from functools import lru_cache
from pathlib import Path

from nemoguardrails import LLMRails, RailsConfig
from nemoguardrails.rails.llm.options import RailStatus, RailType

from .config import chat_model


@lru_cache(maxsize=1)
def rails() -> LLMRails:
    return LLMRails(RailsConfig.from_path(str(Path(__file__).parent / "rails")), llm=chat_model())


def _blocked(messages: list[dict], rail_type: RailType) -> str | None:
    r = rails().check(messages, rail_types=[rail_type])
    return (r.rail or rail_type.value) if r.status == RailStatus.BLOCKED else None


def check_input(text: str) -> str | None:
    """Name of the blocking rail, or None if allowed."""
    return _blocked([{"role": "user", "content": text}], RailType.INPUT)


def check_output(question: str, answer: str) -> str | None:
    return _blocked([{"role": "user", "content": question}, {"role": "assistant", "content": answer}], RailType.OUTPUT)
