"""Settings (environment variables / .env) and model factories.

OpenAI by default. Any OpenAI-compatible server works via a base URL, e.g. Ollama:
  LLM_BASE_URL=EMBED_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:14b-instruct  EMBED_MODEL=nomic-embed-text
Tracing: set LANGSMITH_TRACING=true and LANGSMITH_API_KEY (no code needed).
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.cache import SQLiteCache
from langchain_core.globals import set_llm_cache
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
CORPUS_DIRS = [DATA_DIR / "regulations", DATA_DIR / "northwind", DATA_DIR / "incoming"]
INDEX_DIR, EVAL_DIR, RESULTS_DIR, LOG_DIR = ROOT / "indexes", ROOT / "eval", ROOT / "results", ROOT / "logs"

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")  # writes answers
CHECK_MODEL = os.getenv("CHECK_MODEL", "gpt-4.1-mini")  # plans, re-ranks, grades, guardrails (nano was unreliable)
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")  # evaluation only
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TOP_K = 5  # chunks returned per search
CANDIDATES = 20  # per retriever, before fusion and re-ranking

# Exact-match LLM cache: at temperature 0, repeated prompts cost nothing.
(ROOT / ".cache").mkdir(exist_ok=True)
if os.getenv("LLM_CACHE", "1") == "1":
    set_llm_cache(SQLiteCache(str(ROOT / ".cache" / "llm.sqlite")))


def _api_key(base_url: str | None) -> str | None:
    return os.getenv("OPENAI_API_KEY") or ("not-needed" if base_url else None)  # local servers ignore the key


def chat_model(model: str = LLM_MODEL) -> ChatOpenAI:
    base = os.getenv("LLM_BASE_URL") or None
    return ChatOpenAI(model=model, base_url=base, api_key=_api_key(base), temperature=0, timeout=120, max_retries=3)


def embedding_model(model: str = EMBED_MODEL) -> OpenAIEmbeddings:
    base = os.getenv("EMBED_BASE_URL") or None
    # non-OpenAI servers need raw strings, not pre-tokenized input
    return OpenAIEmbeddings(model=model, base_url=base, api_key=_api_key(base), check_embedding_ctx_length=base is None)
