"""Index (FAISS + LangChain Indexing API) and retrievers (BM25, dense, hybrid RRF, LLM re-rank).

Access control and point-in-time filtering are applied inside retrieval, *before* ranking and
before the LLM sees anything, so a restricted chunk can never leak through the prompt, the
re-ranker or a citation. Metadata comes from each document's front matter:
  access: all | comma-separated roles (e.g. "compliance,ehs")
  effective_date / superseded_on: used when the caller asks for an as-of date (audit questions such
  as "what applied in March 2025?")

Usage: python -m lcrag.store   # build the index, or incrementally sync it with the corpus
"""

import re
from functools import lru_cache

import faiss
from langchain_classic.indexes import SQLRecordManager, index
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from .config import CANDIDATES, EMBED_MODEL, INDEX_DIR, CHECK_MODEL, TOP_K, chat_model, embedding_model
from .corpus import load_documents, split_documents

ROLES = ["employee", "manager", "ehs", "lms_admin", "compliance"]


def sync_index(strategy: str = "structure", model: str = EMBED_MODEL) -> tuple[FAISS, dict]:
    """Only new or changed chunks are embedded. Chunks of edited or deleted files are removed."""
    path = INDEX_DIR / f"{strategy}__{model.replace('/', '_')}"
    emb = embedding_model(model)
    if (path / "index.faiss").exists():
        vs = FAISS.load_local(str(path), emb, allow_dangerous_deserialization=True)  # file written by us
    else:
        vs = FAISS(emb, faiss.IndexFlatL2(len(emb.embed_query("x"))), InMemoryDocstore(), {})
    path.mkdir(parents=True, exist_ok=True)
    rm = SQLRecordManager(f"lcrag/{path.name}", db_url=f"sqlite:///{path / 'records.sqlite'}")
    rm.create_schema()
    stats = index(split_documents(load_documents(), strategy), rm, vs, cleanup="full", source_id_key="source")
    vs.save_local(str(path))
    return vs, stats


_ID = re.compile(r"\d{2,5}\.\d+(?:\([^)\s]{1,5}\))*|\b[A-Z]{1,4}(?:-[A-Z0-9]{1,6})+\b")


def tokenize(text: str) -> list[str]:
    """BM25 tokens. Citations and codes stay whole: '1910.178(l)(4)' -> '1910.178', '1910.178(l)',
    '1910.178(l)(4)'; 'EHS-FL-201' stays one token. Default tokenizers split them into noise."""
    ids = [p for m in _ID.findall(text) for p in [m.split("(")[0]] + [m[: i + 1] for i, c in enumerate(m) if c == ")"]]
    words = re.findall(r"[a-z0-9]+", _ID.sub(" ", text).lower())
    return [i.lower() for i in ids] + [w.rstrip("s") if len(w) > 4 else w for w in words]


def allowed(meta: dict, role: str = "compliance", as_of: str | None = None) -> bool:
    """ACL + point-in-time filter. Compliance sees everything (audit history included)."""
    acl = meta.get("access", "all")
    if role != "compliance" and acl != "all" and role not in acl.split(","):
        return False
    if as_of:  # in force on that date: already effective, and not yet superseded
        if meta.get("effective_date", "0000") > as_of or meta.get("superseded_on", "9999") <= as_of:
            return False
    return True


@lru_cache(maxsize=64)
def _bm25(vs_id: int, role: str, as_of: str | None, k: int) -> BM25Retriever:
    docs = [d for d in _VS[vs_id].docstore._dict.values() if allowed(d.metadata, role, as_of)]
    return BM25Retriever.from_documents(docs, k=k, preprocess_func=tokenize)


_VS: dict[int, FAISS] = {}


def make_retriever(vs: FAISS, mode: str = "hybrid_rerank", k: int = TOP_K, role: str = "compliance",
                   as_of: str | None = None, rerank_model: str = CHECK_MODEL):
    """mode: dense | bm25 | hybrid | hybrid_rerank"""
    _VS[id(vs)] = vs
    n = CANDIDATES if mode not in ("dense", "bm25") else k
    dense = vs.as_retriever(search_kwargs={"k": n, "fetch_k": 50 * n,  # over-fetch so filtering keeps n hits
                                           "filter": lambda m: allowed(m, role, as_of)})
    bm25 = _bm25(id(vs), role, as_of, n)
    if mode in ("dense", "bm25"):
        return {"dense": dense, "bm25": bm25}[mode]
    hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])  # reciprocal rank fusion
    if mode == "hybrid":
        return hybrid | (lambda docs: docs[:k])
    reranked = ContextualCompressionRetriever(base_retriever=hybrid,
                                              base_compressor=LLMListwiseRerank.from_llm(chat_model(rerank_model), top_n=k))
    # Small models occasionally return an invalid ranking (out-of-range index): fall back to RRF order.
    return reranked.with_fallbacks([hybrid | (lambda docs: docs[:k])])


if __name__ == "__main__":
    print(sync_index()[1])
