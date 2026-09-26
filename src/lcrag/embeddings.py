"""Embedding models behind one interface, with a content-addressed cache.

Asymmetric retrieval models need different prefixes for queries and passages; getting this wrong
silently costs several points of recall, so the prefixes are declared per model here.

The cache is keyed by (model, sha1(text)). Re-indexing after a corpus change therefore only
embeds chunks whose text actually changed, which is what makes incremental updates cheap.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import ROOT


@dataclass(frozen=True)
class EmbedSpec:
    name: str
    query_prefix: str = ""
    passage_prefix: str = ""
    max_seq_len: int = 512
    provider: str = "st"  # "st" (sentence-transformers, local) | "openai"


MODELS: dict[str, EmbedSpec] = {
    # General-purpose baseline many RAG stacks start with. 256-token limit truncates long chunks.
    "sentence-transformers/all-MiniLM-L6-v2": EmbedSpec("sentence-transformers/all-MiniLM-L6-v2", max_seq_len=256),
    # Retrieval-trained, 33M params, strong on BEIR for its size; instruction prefix for queries.
    "BAAI/bge-small-en-v1.5": EmbedSpec(
        "BAAI/bge-small-en-v1.5", query_prefix="Represent this sentence for searching relevant passages: "
    ),
    # 110M params, same family: tests whether 3x the compute buys recall on this corpus.
    "BAAI/bge-base-en-v1.5": EmbedSpec(
        "BAAI/bge-base-en-v1.5", query_prefix="Represent this sentence for searching relevant passages: "
    ),
    # Different training recipe (weakly supervised contrastive) with symmetric "query:/passage:" prefixes.
    "intfloat/e5-base-v2": EmbedSpec("intfloat/e5-base-v2", query_prefix="query: ", passage_prefix="passage: "),
    # Hosted option, used only when an API key is configured.
    "text-embedding-3-small": EmbedSpec("text-embedding-3-small", max_seq_len=8191, provider="openai"),
}


class _Cache:
    def __init__(self, model: str) -> None:
        path = ROOT / ".cache" / "embeddings.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS emb (model TEXT, h TEXT, v BLOB, PRIMARY KEY (model, h))")

    def get(self, hashes: list[str]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(hashes), 500):
            batch = hashes[i : i + 500]
            q = f"SELECT h, v FROM emb WHERE model=? AND h IN ({','.join('?' * len(batch))})"
            for h, v in self.db.execute(q, [self.model, *batch]):
                out[h] = np.frombuffer(v, dtype=np.float32)
        return out

    def put(self, items: dict[str, np.ndarray]) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO emb VALUES (?, ?, ?)",
            [(self.model, h, v.astype(np.float32).tobytes()) for h, v in items.items()],
        )
        self.db.commit()


class Embedder:
    def __init__(self, model: str) -> None:
        self.spec = MODELS.get(model, EmbedSpec(model))
        self._cache = _Cache(self.spec.name)
        self.stats = {"cache_hits": 0, "embedded": 0}

    @property
    def name(self) -> str:
        return self.spec.name

    def _encode(self, texts: list[str]) -> np.ndarray:
        if self.spec.provider == "openai":
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("EMBED_BASE_URL") or None)
            vecs = []
            for i in range(0, len(texts), 128):
                resp = client.embeddings.create(model=self.spec.name, input=texts[i : i + 128])
                vecs.extend(d.embedding for d in resp.data)
            arr = np.asarray(vecs, dtype=np.float32)
            return arr / np.linalg.norm(arr, axis=1, keepdims=True)
        model = _st_model(self.spec.name, self.spec.max_seq_len)
        return model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=len(texts) > 200,
                            convert_to_numpy=True).astype(np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        prefixed = [self.spec.passage_prefix + t for t in texts]
        hashes = [hashlib.sha1(t.encode()).hexdigest() for t in prefixed]
        cached = self._cache.get(list(set(hashes)))
        missing = sorted({h: t for h, t in zip(hashes, prefixed) if h not in cached}.items())
        if missing:
            vecs = self._encode([t for _, t in missing])
            new = {h: v for (h, _), v in zip(missing, vecs)}
            self._cache.put(new)
            cached.update(new)
        self.stats["embedded"] += len(missing)
        self.stats["cache_hits"] += len(hashes) - len(missing)
        return np.stack([cached[h] for h in hashes])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.spec.query_prefix + t for t in texts])


def torch_device() -> str:
    """EMBED_DEVICE overrides; otherwise CUDA, then Apple MPS, then CPU."""
    if os.getenv("EMBED_DEVICE"):
        return os.environ["EMBED_DEVICE"]
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@lru_cache(maxsize=4)
def _st_model(name: str, max_seq_len: int):
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(name, device=torch_device())
    m.max_seq_length = max_seq_len
    return m
