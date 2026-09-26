"""FAISS index with incremental sync, and hybrid retrieval with access control.

Filters run inside retrieval, before ranking, so restricted text never reaches the LLM:
  role    document front matter `access: all | compliance,ehs,...` (compliance sees everything)
  as_of   only documents in force on that date (effective, and not yet superseded)
  source  any | regulation (laws) | internal (Northwind documents)

Usage: python -m lcrag.store   # build the index, or sync it after corpus changes
"""

import re

import faiss
from langchain_classic.indexes import SQLRecordManager, index
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from .config import CANDIDATES, CHECK_MODEL, EMBED_MODEL, INDEX_DIR, TOP_K, chat_model, embedding_model
from .corpus import load_documents, split_documents

ROLES = ["employee", "manager", "ehs", "lms_admin", "compliance"]
LAW = ("regulation", "statute")


def sync_index(strategy: str = "structure", model: str = EMBED_MODEL) -> tuple[FAISS, dict]:
    """LangChain Indexing API: only new or changed chunks are embedded; chunks of removed files are deleted."""
    path = INDEX_DIR / f"{strategy}__{model}"
    emb = embedding_model(model)
    if (path / "index.faiss").exists():
        vs = FAISS.load_local(str(path), emb, allow_dangerous_deserialization=True)  # our own file
    else:
        vs = FAISS(emb, faiss.IndexFlatL2(len(emb.embed_query("x"))), InMemoryDocstore(), {})
    path.mkdir(parents=True, exist_ok=True)
    records = SQLRecordManager(f"lcrag/{path.name}", db_url=f"sqlite:///{path / 'records.sqlite'}")
    records.create_schema()
    stats = index(split_documents(load_documents(), strategy), records, vs, cleanup="full", source_id_key="source")
    vs.save_local(str(path))
    return vs, stats


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
    """mode: dense | bm25 | hybrid | hybrid_rerank (default: BM25 + dense fused by RRF, then LLM re-rank)."""
    keep = lambda meta: allowed(meta, role, as_of, source)  # noqa: E731
    n = k if mode in ("dense", "bm25") else CANDIDATES
    dense = vs.as_retriever(search_kwargs={"k": n, "fetch_k": 50 * n, "filter": keep})  # over-fetch, then filter
    bm25 = BM25Retriever.from_documents([d for d in vs.docstore._dict.values() if keep(d.metadata)],
                                        k=n, preprocess_func=tokenize)
    if mode == "dense":
        return dense
    if mode == "bm25":
        return bm25
    hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5]) | (lambda docs: docs[:k])
    if mode == "hybrid":
        return hybrid
    reranked = ContextualCompressionRetriever(
        base_retriever=EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5]),
        base_compressor=LLMListwiseRerank.from_llm(chat_model(CHECK_MODEL), top_n=k))
    return reranked.with_fallbacks([hybrid])  # an invalid ranking from the LLM falls back to RRF order


if __name__ == "__main__":
    print(sync_index()[1])
