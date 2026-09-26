"""Retrieval pipeline: dense and sparse candidates -> fusion -> cross-encoder re-ranking -> source policy.

Fusion options
  * RRF (reciprocal rank fusion, Cormack et al. 2009): score = sum over retrievers of 1/(k + rank).
    It uses ranks only, so no score calibration is needed between cosine similarities (about 0.3-0.9)
    and unbounded BM25 scores. This is the default.
  * Weighted score fusion ("convex"): min-max normalise each list and combine with weight alpha.
    Kept as the alternative that needs tuning; it is compared against RRF in the experiments.

Re-ranking
  A cross-encoder reads the query and chunk together, which bi-encoders cannot do. It is the main
  precision lever at top-k, and it costs one forward pass per candidate, so only the fused top-N
  candidates (default 20) are re-scored.

Source policy
  Superseded documents stay in the index because "what did the old policy say?" is a legitimate
  audit question. They are demoted below current sources unless the query asks about history.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass, field
from functools import lru_cache

from .config import RetrievalConfig, get_retrieval_config
from .embeddings import torch_device
from .index import CorpusIndex
from .query import expand
from .schema import Chunk

MODES = ("dense", "sparse", "hybrid", "hybrid_convex", "dense_rerank", "hybrid_rerank")
_HISTORY_RE = re.compile(r"\b(v?3\.1|version 3\.1|previous|prior|old|older|before 2026|superseded|history|used to|changed?)\b",
                         re.I)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk.chunk_id, "cite": self.chunk.cite, "score": round(self.score, 4),
                "dense_rank": self.dense_rank, "sparse_rank": self.sparse_rank,
                "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 4)}


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    timings_ms: dict = field(default_factory=dict)

    @property
    def top_rerank_score(self) -> float | None:
        scores = [h.rerank_score for h in self.hits if h.rerank_score is not None]
        return max(scores) if scores else None


@lru_cache(maxsize=2)
def _cross_encoder(name: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(name, device=torch_device(), max_length=512)


def rrf(rank_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranks in rank_lists:
        for r, cid in enumerate(ranks, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + r)
    return scores


def convex(dense: list[tuple[str, float]], sparse: list[tuple[str, float]], alpha: float = 0.5) -> dict[str, float]:
    def norm(pairs: list[tuple[str, float]]) -> dict[str, float]:
        if not pairs:
            return {}
        vals = [s for _, s in pairs]
        lo, hi = min(vals), max(vals)
        return {c: (s - lo) / (hi - lo) if hi > lo else 1.0 for c, s in pairs}

    d, s = norm(dense), norm(sparse)
    return {c: alpha * d.get(c, 0.0) + (1 - alpha) * s.get(c, 0.0) for c in set(d) | set(s)}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class Retriever:
    def __init__(self, index: CorpusIndex, mode: str = "hybrid_rerank", cfg: RetrievalConfig | None = None,
                 source_policy: bool = True, alpha: float = 0.5, expand_query: bool = True) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.index, self.mode, self.cfg = index, mode, cfg or get_retrieval_config()
        self.source_policy, self.alpha, self.expand_query = source_policy, alpha, expand_query

    def rerank(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not hits:
            return hits
        model = _cross_encoder(self.cfg.rerank_model)
        raw = model.predict([(query, h.chunk.text) for h in hits], batch_size=32, show_progress_bar=False)
        for h, s in zip(hits, raw):
            s = float(s)
            h.rerank_score = s if 0.0 <= s <= 1.0 else _sigmoid(s)  # model-agnostic [0, 1] relevance
            h.score = h.rerank_score
        return sorted(hits, key=lambda h: -h.score)

    def _apply_source_policy(self, query: str, hits: list[Hit]) -> list[Hit]:
        if not self.source_policy or _HISTORY_RE.search(query):
            return hits
        current = [h for h in hits if h.chunk.meta.get("status") != "superseded"]
        old = [h for h in hits if h.chunk.meta.get("status") == "superseded"]
        return current + old

    def retrieve(self, query: str, k: int | None = None) -> RetrievalResult:
        k = k or self.cfg.final_k
        t: dict[str, float] = {}
        chunks = self.index.chunks
        q_search = expand(query) if self.expand_query else query  # retrievers see the expanded query

        t0 = time.perf_counter()
        dense = self.index.dense.search(q_search, self.cfg.dense_k) if self.mode != "sparse" else []
        t["dense"] = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        sparse = self.index.sparse.search(q_search, self.cfg.sparse_k) if self.mode not in ("dense", "dense_rerank") else []
        t["sparse"] = (time.perf_counter() - t0) * 1000

        d_rank = {c: i for i, (c, _) in enumerate(dense, start=1)}
        s_rank = {c: i for i, (c, _) in enumerate(sparse, start=1)}

        if self.mode in ("dense", "dense_rerank"):
            fused = {c: s for c, s in dense}
        elif self.mode == "sparse":
            fused = {c: s for c, s in sparse}
        elif self.mode == "hybrid_convex":
            fused = convex(dense, sparse, self.alpha)
        else:
            fused = rrf([[c for c, _ in dense], [c for c, _ in sparse]], k=self.cfg.rrf_k)

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])
        hits = [Hit(chunks[c], s, d_rank.get(c), s_rank.get(c)) for c, s in ordered]

        if self.mode.endswith("rerank"):
            t0 = time.perf_counter()
            hits = self.rerank(q_search, hits[: self.cfg.rerank_candidates])
            t["rerank"] = (time.perf_counter() - t0) * 1000

        hits = self._apply_source_policy(query, hits)
        return RetrievalResult(query, hits[:k], {k_: round(v, 1) for k_, v in t.items()})
