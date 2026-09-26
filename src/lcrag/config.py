"""Settings from environment variables / .env.

OpenAI by default. Any OpenAI-compatible server works by setting a base URL:
  Ollama: LLM_BASE_URL=EMBED_BASE_URL=http://localhost:11434/v1, LLM_MODEL=qwen2.5:14b-instruct,
          EMBED_MODEL=nomic-embed-text
  vLLM:   LLM_BASE_URL=http://<host>:8000/v1
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")
DATA_DIR = ROOT / "data"
CORPUS_DIRS = [DATA_DIR / "regulations", DATA_DIR / "synthetic" / "docs", DATA_DIR / "incoming"]
INDEX_DIR, EVAL_DIR, RESULTS_DIR = ROOT / "indexes", ROOT / "eval", ROOT / "results"
LOG_DIR = ROOT / "logs"  # audit.jsonl (every request), feedback.jsonl
(ROOT / ".cache").mkdir(exist_ok=True)

# Exact-match LLM cache (temperature 0 -> identical prompts give identical answers): repeated
# questions cost nothing. Tracing: set LANGSMITH_TRACING=true + LANGSMITH_API_KEY, no code needed.
if os.getenv("LLM_CACHE", "1") == "1":
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    set_llm_cache(SQLiteCache(str(ROOT / ".cache" / "llm.sqlite")))

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-nano")
# Checking tasks (re-ranking, grounding grader, guardrail classifiers) need a stronger model than
# generation: with nano, re-rank recall fell 0.906 -> 0.844 and the grader flagged ~80% of faithful
# answers as unsupported. Generation stays on nano.
CHECK_MODEL = os.getenv("CHECK_MODEL", "gpt-4.1-mini")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gpt-4.1-mini")  # evaluation only; stronger than the generator
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
TOP_K = 5  # chunks given to the generator
CANDIDATES = 20  # per retriever, before fusion and re-ranking


def _key(base_url):  # local servers ignore the key, but the client requires one
    return os.getenv("OPENAI_API_KEY") or ("not-needed" if base_url else None)


def chat_model(model: str = LLM_MODEL) -> ChatOpenAI:
    base = os.getenv("LLM_BASE_URL") or None
    return ChatOpenAI(model=model, base_url=base, api_key=_key(base), temperature=0, timeout=120, max_retries=3)


def embedding_model(model: str = EMBED_MODEL) -> OpenAIEmbeddings:
    base = os.getenv("EMBED_BASE_URL") or None
    # check_embedding_ctx_length=False sends raw strings, which non-OpenAI servers (Ollama) require
    return OpenAIEmbeddings(model=model, base_url=base, api_key=_key(base), check_embedding_ctx_length=base is None)
