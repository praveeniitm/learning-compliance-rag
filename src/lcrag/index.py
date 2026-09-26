"""Vector + keyword indexes with incremental synchronisation.

Vector store: FAISS `IndexIDMap2(IndexFlatIP)` on L2-normalised embeddings (inner product = cosine).
  * Exact search is deliberate: at ~1k chunks a flat scan takes well under 1 ms, so an approximate
    index (HNSW/IVF) would add recall loss and tuning for no latency gain. The ID map allows
    `remove_ids` + `add_with_ids`, which is what makes incremental updates possible.
  * Why FAISS rather than a database-backed store (pgvector, Weaviate, Chroma): the system is
    single-node and read-mostly, and FAISS keeps the retrieval layer embeddable and dependency-free
    for reproducible experiments. Metadata filtering (status, doc_type) is done in Python on the
    candidate set. With multi-tenant data or concurrent writers, pgvector would be preferred, since
    vectors, metadata and ACLs then live in one transactional store.

Incremental sync: every chunk carries a content hash. `sync()` diffs the new chunk set against the
stored state, removes vectors for deleted or changed chunks, and adds vectors for new or changed
ones. Embeddings come from a content-addressed cache, so only genuinely new text is embedded.
BM25 statistics are global, so the keyword index is rebuilt; that takes milliseconds at this size.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import faiss
import numpy as np

from .config import INDEX_DIR
from .embeddings import Embedder
from .schema import Chunk
from .sparse import BM25Index


def index_dir(chunks_name: str, model: str) -> Path:
    return INDEX_DIR / f"{chunks_name}__{re.sub(r'[^A-Za-z0-9.-]+', '_', model)}"


class VectorIndex:
    def __init__(self, embedder: Embedder, dim: int) -> None:
        self.embedder = embedder
        self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))
        self.state: dict[str, dict] = {}  # chunk_id -> {"id": int, "hash": str}
        self.next_id = 0

    # ---- persistence -------------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "faiss.index"))
        (path / "state.json").write_text(json.dumps({"next_id": self.next_id, "state": self.state,
                                                     "model": self.embedder.name}))

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> "VectorIndex":
        obj = cls.__new__(cls)
        obj.embedder = embedder
        obj.index = faiss.read_index(str(path / "faiss.index"))
        s = json.loads((path / "state.json").read_text())
        obj.state, obj.next_id = s["state"], s["next_id"]
        return obj

    # ---- updates -----------------------------------------------------------------------------
    def sync(self, chunks: list[Chunk]) -> dict:
        t0 = time.perf_counter()
        before = dict(self.embedder.stats)
        new_state = {c.chunk_id: c.meta["hash"] for c in chunks}
        to_remove = [v["id"] for cid, v in self.state.items() if new_state.get(cid) != v["hash"]]
        to_add = [c for c in chunks if self.state.get(c.chunk_id, {}).get("hash") != c.meta["hash"]]
        unchanged = len(chunks) - len(to_add)
        removed_ids = {cid for cid, v in self.state.items() if new_state.get(cid) != v["hash"]}

        if to_remove:
            self.index.remove_ids(np.asarray(to_remove, dtype=np.int64))
        for cid in removed_ids:
            del self.state[cid]
        if to_add:
            vecs = self.embedder.embed_passages([c.text for c in to_add])
            ids = np.arange(self.next_id, self.next_id + len(to_add), dtype=np.int64)
            self.index.add_with_ids(vecs, ids)
            for c, i in zip(to_add, ids):
                self.state[c.chunk_id] = {"id": int(i), "hash": c.meta["hash"]}
            self.next_id += len(to_add)
        return {
            "added_or_updated": len(to_add),
            "removed_or_replaced": len(to_remove),
            "unchanged": unchanged,
            "newly_embedded": self.embedder.stats["embedded"] - before["embedded"],
            "embedding_cache_hits": self.embedder.stats["cache_hits"] - before["cache_hits"],
            "seconds": round(time.perf_counter() - t0, 2),
            "total_vectors": int(self.index.ntotal),
        }

    # ---- search ------------------------------------------------------------------------------
    def search_vec(self, qvec: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        scores, ids = self.index.search(qvec.reshape(1, -1).astype(np.float32), min(k, self.index.ntotal))
        by_int = {v["id"]: cid for cid, v in self.state.items()}
        return [(by_int[int(i)], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        return self.search_vec(self.embedder.embed_query(query), k)


class CorpusIndex:
    """Chunk store + dense index + sparse index for one (chunking, embedding model) pair."""

    def __init__(self, chunks: list[Chunk], dense: VectorIndex) -> None:
        self.dense = dense
        self.set_chunks(chunks)

    def set_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks = {c.chunk_id: c for c in chunks}
        self.sparse = BM25Index(chunks)

    @classmethod
    def build(cls, chunks: list[Chunk], chunks_name: str, model: str, save: bool = True) -> "CorpusIndex":
        path = index_dir(chunks_name, model)
        embedder = Embedder(model)
        if (path / "faiss.index").exists():
            dense = VectorIndex.load(path, embedder)
        else:
            dim = embedder.embed_passages(["dimension probe"]).shape[1]
            dense = VectorIndex(embedder, dim)
        stats = dense.sync(chunks)
        obj = cls(chunks, dense)
        if save:
            dense.save(path)
            _write_chunks(path, chunks)
        obj.last_sync = stats
        return obj

    @classmethod
    def load(cls, chunks_name: str, model: str) -> "CorpusIndex":
        path = index_dir(chunks_name, model)
        chunks = [Chunk.from_dict(json.loads(l)) for l in (path / "chunks.jsonl").read_text().splitlines()]
        return cls(chunks, VectorIndex.load(path, Embedder(model)))

    def update(self, chunks: list[Chunk], chunks_name: str) -> dict:
        """Incrementally bring the index in line with a new chunk set and persist it."""
        stats = self.dense.sync(chunks)
        self.set_chunks(chunks)
        path = index_dir(chunks_name, self.dense.embedder.name)
        self.dense.save(path)
        _write_chunks(path, chunks)
        return stats


def _write_chunks(path: Path, chunks: list[Chunk]) -> None:
    with (path / "chunks.jsonl").open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")


def main() -> None:
    import argparse

    from .config import get_retrieval_config
    from .ingest import load_chunks

    ap = argparse.ArgumentParser(description="Build or incrementally update an index")
    ap.add_argument("--chunks", default="structure350")
    ap.add_argument("--model", default=get_retrieval_config().embed_model)
    args = ap.parse_args()
    idx = CorpusIndex.build(load_chunks(args.chunks), args.chunks, args.model)
    print(json.dumps(idx.last_sync, indent=2))


if __name__ == "__main__":
    main()
