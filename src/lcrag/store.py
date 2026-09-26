"""Index (FAISS + LangChain Indexing API) and retrievers (BM25, dense, hybrid RRF, LLM re-rank).

Usage: python -m lcrag.store   # build the index, or incrementally sync it with the corpus
"""

import re

import faiss
from langchain_classic.indexes import SQLRecordManager, index
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import LLMListwiseRerank
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS

from .config import CANDIDATES, EMBED_MODEL, INDEX_DIR, TOP_K, chat_model, embedding_model
from .corpus import load_documents, split_documents


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


def make_retriever(vs: FAISS, mode: str = "hybrid_rerank", k: int = TOP_K):
    """mode: dense | bm25 | hybrid | hybrid_rerank"""
    dense = vs.as_retriever(search_kwargs={"k": CANDIDATES if mode != "dense" else k})
    bm25 = BM25Retriever.from_documents(list(vs.docstore._dict.values()), k=CANDIDATES if mode != "bm25" else k,
                                        preprocess_func=tokenize)
    if mode in ("dense", "bm25"):
        return {"dense": dense, "bm25": bm25}[mode]
    hybrid = EnsembleRetriever(retrievers=[bm25, dense], weights=[0.5, 0.5])  # reciprocal rank fusion
    if mode == "hybrid":
        return hybrid | (lambda docs: docs[:k])
    return ContextualCompressionRetriever(base_retriever=hybrid,
                                          base_compressor=LLMListwiseRerank.from_llm(chat_model(), top_n=k))


if __name__ == "__main__":
    print(sync_index()[1])
