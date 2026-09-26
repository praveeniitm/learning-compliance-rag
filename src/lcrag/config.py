"""Central configuration. Everything tunable lives here or in environment variables.

The LLM client is OpenAI-compatible, so the same code talks to api.openai.com,
a local Ollama server (http://localhost:11434/v1) or a vLLM deployment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTH_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = ROOT / "indexes"
EVAL_DIR = ROOT / "eval"
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "") or "not-needed")
    base_url: str | None = field(default_factory=lambda: os.getenv("LLM_BASE_URL") or None)
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4o-mini"))
    temperature: float = 0.0
    max_tokens: int = 700
    timeout_s: float = 120.0


@dataclass(frozen=True)
class RetrievalConfig:
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    rerank_model: str = field(
        default_factory=lambda: os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    )
    dense_k: int = 30
    sparse_k: int = 30
    rrf_k: int = 60  # standard RRF constant (Cormack et al., 2009)
    rerank_candidates: int = 20
    final_k: int = 5


def get_llm_config() -> LLMConfig:
    return LLMConfig()


def get_retrieval_config() -> RetrievalConfig:
    return RetrievalConfig()
