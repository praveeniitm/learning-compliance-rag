"""Hybrid retrieval with access control.

One search = BM25 (keywords) + FAISS (meaning) -> reciprocal rank fusion -> LLM re-rank -> top 5.

Filters run inside retrieval, before ranking, so restricted text never reaches the LLM:
  role    front matter `access: all | compliance,ehs,...` (the compliance role sees everything)
  as_of   only documents in force on that date (already effective, and not yet superseded)
  source  any | regulation (laws) | internal (Northwind documents)
"""

import re

from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from .config import CANDIDATES, CHECK_MODEL, TOP_K, chat_model

ROLES = ["employee", "manager", "ehs", "lms_admin", "compliance"]
LAW = ("regulation", "statute")
CODE = re.compile(r"\d{2,5}\.\d+(?:\([^)\s]{1,5}\))*|\b[A-Z]{1,4}(?:-[A-Z0-9]{1,6})+\b")


def tokenize(text: str) -> list[str]:
    """BM25 tokens that keep citations and course codes whole.

    '1910.178(l)(4)' -> '1910.178', '1910.178(l)', '1910.178(l)(4)';  'EHS-FL-201' -> 'ehs-fl-201'.
    """
    codes = []
    for m in CODE.findall(text):
        codes.append(m.split("(")[0])
        codes += [m[: i + 1] for i, ch in enumerate(m) if ch == ")"]
    words = re.findall(r"[a-z0-9]+", CODE.sub(" ", text).lower())
    return [c.lower() for c in codes] + [w.rstrip("s") if len(w) > 4 else w for w in words]


def allowed(meta: dict, role: str = "compliance", as_of: str | None = None, source: str = "any") -> bool:
    if source != "any" and (meta.get("doc_type") in LAW) != (source == "regulation"):
        return False
    access = meta.get("access", "all")
    if role != "compliance" and access != "all" and role not in access.split(","):
        return False
    if as_of and (meta.get("effective_date", "0000") > as_of or meta.get("superseded_on", "9999") <= as_of):
        return False
    return True


def make_retriever(vs: FAISS, mode: str = "hybrid_rerank", role: str = "compliance", as_of: str | None = None,
                   source: str = "any", k: int = TOP_K):
    """mode: dense | bm25 | hybrid | hybrid_rerank (the default used by the agent)."""
    keep = lambda meta: allowed(meta, role, as_of, source)  # noqa: E731
    n = k if mode in ("dense", "bm25") else CANDIDATES
    dense = vs.as_retriever(search_kwargs={"k": n, "fetch_k": 50 * n, "filter": keep})  # over-fetch, then filter
    bm25 = BM25Retriever.from_documents([d for d in vs.docstore._dict.values() if keep(d.metadata)],
                                        k=n, preprocess_func=tokenize)
    if mode == "dense":
        return dense
    if mode == "bm25":
        return bm25
    fused = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])  # reciprocal rank fusion
    top_fused = fused | (lambda docs: docs[:k])
    if mode == "hybrid":
        return top_fused
    reranked = ContextualCompressionRetriever(
        base_retriever=fused, base_compressor=LLMListwiseRerank.from_llm(chat_model(CHECK_MODEL), top_n=k))
    return reranked.with_fallbacks([top_fused])  # an invalid ranking from the LLM falls back to fusion order
